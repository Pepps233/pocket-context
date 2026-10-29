import json
import shlex
import shutil
import subprocess
import sys
import tarfile
import threading
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from pocketcontextagent.comparisons.registry import METHODS
from pocketcontextagent.comparisons.report import compare_runs
from pocketcontextagent.environments.swebench_verified import SWEBenchVerifiedEnvironment, subprocess_backend
from pocketcontextagent.models.test_models import make_toolcall_output
from pocketcontextagent.run.benchmarks.benchmark_grading import export_evaluation, import_evaluation
from pocketcontextagent.run.benchmarks.benchmark_matrix import (
    archive_trajectories,
    benchmark_config,
    bounded_command,
    export_predictions,
    load_instances,
    run_instance,
    run_manifest,
    sandbox_params,
)
from pocketcontextagent.run.benchmarks.browsecomp_plus import app
from pocketcontextagent.run.benchmarks.browsecomp_plus_daytona import publish_matrix_task, run_matrix_workers
from pocketcontextagent.run.benchmarks.utils.swebench_daytona import PreparedSnapshot, RuntimeConfig


def test_partial_trajectory_archive_is_not_treated_as_successful_collection(tmp_path):
    with pytest.raises(FileNotFoundError):
        archive_trajectories(tmp_path)
    assert not (tmp_path / "trajectories.tar.gz").exists()
    assert (tmp_path / "trajectories.tar.gz.tmp").is_file()
    source = tmp_path / "trajectories" / "instance"
    source.mkdir(parents=True)
    (source / "trajectory.json").write_text('{"complete": true}')
    (source / "context.sqlite").write_bytes(b"context-state")
    archive_trajectories(tmp_path)
    with tarfile.open(tmp_path / "trajectories.tar.gz") as archive:
        assert archive.extractfile("trajectories/instance/trajectory.json").read() == b'{"complete": true}'
        assert archive.extractfile("trajectories/instance/context.sqlite").read() == b"context-state"
    assert not (tmp_path / "trajectories.tar.gz.tmp").exists()


def git(directory: Path, *arguments: str, input: str | None = None) -> str:
    return subprocess.run(
        ["git", "-C", str(directory), *arguments], input=input, text=True, capture_output=True, check=True
    ).stdout


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    repo = tmp_path / "source"
    repo.mkdir()
    git(repo, "init", "--quiet")
    git(repo, "config", "user.email", "test@example.com")
    git(repo, "config", "user.name", "Test")
    (repo / "solver.py").write_text("answer = 0\n")
    git(repo, "add", "solver.py")
    git(repo, "commit", "--quiet", "-m", "base")
    return repo


def public_rows(base: str, count: int = 2) -> list[dict]:
    return [
        {
            "instance_id": f"fixture__repo-{i}",
            "problem_statement": "Set answer to 42.",
            "base_commit": base,
            "repo": "fixture/repo",
            "version": "1",
            "patch": "SECRET_GOLD",
            "test_patch": "SECRET_TEST",
            "FAIL_TO_PASS": ["SECRET_TEST_NAME"],
        }
        for i in range(count)
    ]


def native_call(name: str, command: str) -> dict:
    return make_toolcall_output(
        "Work on the repository.",
        [{"id": name, "type": "function", "function": {"name": "bash", "arguments": json.dumps({"command": command})}}],
        [{"command": command, "tool_call_id": name}],
    )


@pytest.mark.parametrize(("step_limit", "edit"), [(1, True), (5, True), (5, False)])
def test_four_method_coding_matrix_exports_actual_patches_and_native_grades(
    repository: Path, tmp_path: Path, step_limit: int, edit: bool
):
    data = tmp_path / "data"
    data.mkdir()
    (data / "queries.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in public_rows(git(repository, "rev-parse", "HEAD").strip()))
    )
    (data / "grading_dataset.jsonl").write_text("PRIVATE_GRADING_FIXTURE\n")
    instances = load_instances(data, query_ids=[], limit=2, seed=42)
    assert all(
        set(row) == {"query_id", "instance_id", "query", "problem_statement", "repo", "base_commit", "version", "image"}
        for row in instances
    )
    replay = tmp_path / "replay.yaml"
    outputs = ([native_call("edit", "printf 'answer = 42\\n' > solver.py")] if edit else []) + [
        native_call("submit", "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT")
    ]
    replay.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "model_class": "pocketcontextagent.models.test_models.DeterministicToolcallModel",
                    "model_name": "deepseek/replay",
                    "outputs": outputs,
                },
                "agent": {"cost_limit": 0, "step_limit": step_limit},
            }
        )
    )
    config = benchmark_config([str(replay)])
    manifests = {
        method: run_manifest(config, method, instances, 2, data / "queries.jsonl", "test") for method in METHODS
    }
    root = tmp_path / "runs"
    locks = {method: threading.Lock() for method in METHODS}

    def execute(method: str, worker: dict, instance: dict) -> None:
        checkout = tmp_path / "sandboxes" / worker["id"] / instance["query_id"]
        checkout.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--quiet", str(repository), str(checkout)], check=True)
        env = SWEBenchVerifiedEnvironment(
            subprocess_backend,
            cwd=str(checkout),
            base_commit=instance["base_commit"],
            context_tools=manifests[method]["config"]["environment"]["context_tools"],
        )
        env.prepare(fresh_isolated=True)
        task = root / method / "tasks" / instance["query_id"]
        run_instance(instance, task, manifests[method]["config"], env, manifests[method])
        with locks[method]:
            publish_matrix_task(task, root / method, instance["query_id"], manifests[method])
        assert "SECRET_" not in (task / "trajectories" / instance["query_id"] / "trajectory.json").read_text()

    run_matrix_workers({method: [{"id": f"{method}-{i}"} for i in range(2)] for method in METHODS}, instances, execute)
    assert all(row["success_rate"] is None for row in compare_runs(root)["methods"].values())
    export_predictions(root)
    commands = export_evaluation(root, data, "fixture-run")
    assert all("--dataset_name" in command and "grading_dataset.jsonl" in command for command in commands.values())
    original_grading = (data / "grading_dataset.jsonl").read_bytes()
    (data / "grading_dataset.jsonl").write_text("CHANGED_PRIVATE_SNAPSHOT")
    with pytest.raises(ValueError, match="frozen public/private snapshot"):
        export_evaluation(root, data, "fixture-run")
    (data / "grading_dataset.jsonl").write_bytes(original_grading)
    with pytest.raises(ValueError, match="matching exported run prefix"):
        import_evaluation(root, tmp_path / "no-logs-needed", "wrong-prefix", tmp_path / "wrong-grades")
    logs = tmp_path / "logs"
    for method in METHODS:
        predictions = [json.loads(line) for line in (root / method / "predictions.jsonl").read_text().splitlines()]
        assert len(predictions) == 2 and all(bool(row["model_patch"]) == edit for row in predictions)
        for row in predictions:
            original = root / method / "tasks" / row["instance_id"]
            assert (original / f"{row['instance_id']}.json").read_bytes() == (
                root / method / f"{row['instance_id']}.json"
            ).read_bytes()
            if not edit:
                continue
            check = tmp_path / "grading" / method / row["instance_id"]
            check.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "clone", "--quiet", str(repository), str(check)], check=True)
            git(check, "apply", "-", input=row["model_patch"])
            subprocess.run([sys.executable, "-c", "import solver; assert solver.answer == 42"], cwd=check, check=True)
            folder = logs / f"fixture-run-{method}" / row["model_name_or_path"].replace("/", "__") / row["instance_id"]
            folder.mkdir(parents=True)
            (folder / "patch.diff").write_text(row["model_patch"])
            (folder / "report.json").write_text(
                json.dumps(
                    {
                        row["instance_id"]: {
                            "resolved": True,
                            "patch_exists": True,
                            "patch_successfully_applied": True,
                            "tests_status": {
                                "FAIL_TO_PASS": {"success": ["solver.answer"], "failure": []},
                                "PASS_TO_PASS": {"success": [], "failure": []},
                            },
                        }
                    }
                )
            )
    graded = import_evaluation(root, logs, "fixture-run", tmp_path / "grades")
    for row in graded["methods"].values():
        assert row["tasks"] == row["graded_tasks"] == 2
        assert row["completed"] == (0 if step_limit == 1 else 2)
        assert row["success_rate"] == float(edit)
    grade = json.loads((tmp_path / "grades" / "react" / f"{instances[0]['query_id']}_eval.json").read_text())
    assert grade["grading_basis"] == ("official_harness" if edit else "confirmed_empty_patch")
    coverage = yaml.safe_load((tmp_path / "grades/coverage.yaml").read_text())
    assert coverage["react"]["known_outcome_coverage"] == 1.0
    assert coverage["react"]["test_execution_coverage"] == float(edit)
    if edit:
        first = instances[0]["query_id"]
        (logs / "fixture-run-pocket_context" / "deepseek__replay" / first / "report.json").unlink()
        partial = import_evaluation(root, logs, "fixture-run", tmp_path / "partial-grades")
        assert partial["methods"]["pocket_context"]["success_rate"] is None
        assert partial["methods"]["pocket_context"]["ungraded_tasks"] == 1
    path = root / "pocket_context" / f"{instances[0]['query_id']}.json"
    failed_capture = json.loads(path.read_text())
    failed_capture.update(model_patch="", patch_capture_status="failed")
    path.write_text(json.dumps(failed_capture))
    with pytest.raises(ValueError, match="incomplete patch captures"):
        export_predictions(root)


def test_runtime_error_retains_actual_patch_and_explicit_error_status(repository: Path, tmp_path: Path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "queries.jsonl").write_text(json.dumps(public_rows(git(repository, "rev-parse", "HEAD").strip(), 1)[0]))
    instance = load_instances(data, query_ids=[], limit=1, seed=42)[0]
    replay = tmp_path / "error.yaml"
    replay.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "model_class": "pocketcontextagent.models.test_models.DeterministicToolcallModel",
                    "model_name": "deepseek/replay",
                    "outputs": [
                        native_call("edit", "printf 'answer = 42\\n' > solver.py"),
                        native_call("slow", "sleep 3"),
                    ],
                },
                "agent": {"cost_limit": 0},
            }
        )
    )
    config = benchmark_config([str(replay)])
    manifest = run_manifest(config, "pocket_context", [instance], 1, data / "queries.jsonl", "test")
    env = SWEBenchVerifiedEnvironment(
        subprocess_backend, cwd=str(repository), base_commit=instance["base_commit"], timeout=1
    )
    env.prepare(fresh_isolated=True)
    output = tmp_path / "attempt"
    with pytest.raises(subprocess.TimeoutExpired):
        run_instance(instance, output, manifest["config"], env, manifest)
    result = json.loads((output / f"{instance['query_id']}.json").read_text())
    assert result["status"] == result["execution_error"] == "TimeoutExpired"
    assert result["patch_capture_status"] == "captured" and "+answer = 42" in result["model_patch"]
    assert (output / "trajectories.tar.gz").is_file()
    assert (
        json.loads((output / "trajectories" / instance["query_id"] / "trajectory.json").read_text())["info"][
            "execution_error"
        ]
        == "TimeoutExpired"
    )


def test_coding_cli_dry_run_needs_no_sandbox_or_provider_credentials(repository: Path, tmp_path: Path):
    (tmp_path / "queries.jsonl").write_text(
        "\n".join(json.dumps(row) for row in public_rows(git(repository, "rev-parse", "HEAD").strip()))
    )
    result = CliRunner().invoke(
        app,
        [
            "daytona",
            "matrix",
            "--benchmark",
            "swebench-verified",
            "--data-dir",
            str(tmp_path),
            "--limit",
            "1",
            "--workers-per-method",
            "30",
            "--seed",
            "42",
            "--output",
            str(tmp_path / "output"),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    plan = yaml.safe_load(result.output)
    assert plan["attempts"] == plan["maximum_active_sandboxes"] == 4
    assert plan["active_workers_per_method"] == 1 and not (tmp_path / "output").exists()
    assert "SECRET_" not in result.output
    preflight = CliRunner().invoke(
        app,
        [
            "daytona",
            "matrix",
            "--benchmark",
            "swebench-verified",
            "--data-dir",
            str(tmp_path),
            "--limit",
            "1",
            "--preflight",
            "--dry-run",
        ],
    )
    assert preflight.exit_code == 0, preflight.output
    infra = yaml.safe_load(preflight.output)
    assert infra["preflight_only"] is True and infra["attempts"] == 0
    assert infra["planned_method_tasks"] == 4 and infra["sandbox_idle_stop_minutes"] == 0
    assert infra["sandbox_ttl_minutes"] == 240 and infra["task_wall_time_limit_seconds"] == 7200
    assert load_instances(tmp_path, query_ids=[], limit=1, seed=42) == load_instances(
        tmp_path, query_ids=[], limit=1, seed=42
    )
    with pytest.raises(ValueError, match="absent"):
        load_instances(tmp_path, query_ids=["missing"], limit=1, seed=42)
    with pytest.raises(ValueError, match="test split"):
        load_instances(tmp_path, query_ids=[], limit=1, seed=42, split="dev")
    with pytest.raises(ValueError, match="coding provider"):
        benchmark_config(["browsecomp_plus_deepseek"])


def test_sandbox_spec_uses_official_image_without_control_or_model_keys():
    spec = sandbox_params(
        PreparedSnapshot("snapshot-name", "docker.io/swebench/fixture@sha256:" + "a" * 64),
        "pocket_context",
        RuntimeConfig(),
    )
    assert spec.snapshot == "snapshot-name" and spec.os_user == "root"
    assert "image" not in spec.model_dump() and "resources" not in spec.model_dump()
    assert spec.network_block_all is True and spec.public is False
    assert set(spec.env_vars) == {"BASH_ENV", "PAGER", "GIT_TERMINAL_PROMPT"}
    assert spec.ephemeral is False and spec.auto_stop_interval == spec.auto_pause_interval == 0
    assert spec.auto_delete_interval == -1 and spec.ttl_minutes == 240
    assert not any("API_KEY" in key for key in spec.env_vars)
    assert spec.volumes is None and spec.secrets is None


def test_bounded_command_preserves_shell_quoting_and_rejects_disabled_timeout():
    command = bounded_command("bash -c 'printf hello; sleep 3'", 1)
    assert shlex.split(command) == [
        "timeout",
        "--signal=TERM",
        "--kill-after=5s",
        "1s",
        "bash",
        "-c",
        "printf hello; sleep 3",
    ]
    with pytest.raises(ValueError, match="positive"):
        bounded_command("echo hello", 0)


@pytest.mark.skipif(shutil.which("timeout") is None, reason="GNU timeout is supplied by the Linux instance image")
def test_shell_timeout_is_a_nonzero_tool_result_not_a_transport_exception():
    result = subprocess.run(
        shlex.split(bounded_command("bash -c 'printf started; sleep 3'", 1)), capture_output=True, text=True, timeout=8
    )
    assert result.returncode == 124 and result.stdout == "started"
