import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from benchmarks.benchmark_matrix import archive_trajectories
from benchmarks.browsecomp_plus import implementation_hash, run_instance
from benchmarks.browsecomp_plus_daytona import matrix_query_bytes, publish_matrix_task
from benchmarks.utils.controller_artifacts import ArtifactStore
from benchmarks.utils.controller_browsecomp import (
    PUBLIC_ENV,
    RETRIEVAL_CODE,
    RemoteRetriever,
    app,
    plan,
    run_manifest,
    run_matrix,
    worker_params,
)
from benchmarks.utils.swebench_daytona import RuntimeConfig
from pocketcontextagent.comparisons.registry import METHODS
from pocketcontextagent.comparisons.report import compare_runs
from pocketcontextagent.config import builtin_config_dir
from pocketcontextagent.models.test_models import make_toolcall_output
from tests.run.test_browsecomp_plus import bm25_index  # noqa: F401


def subprocess_backend(command: str, *, timeout: int) -> dict:
    environment = os.environ | {
        "PATH": f"{Path(sys.executable).parent}:{os.environ['PATH']}",
        "MSWEA_SILENT_STARTUP": "1",
    }
    if sys.platform == "darwin" and not environment.get("JAVA_HOME"):
        environment["JAVA_HOME"] = subprocess.run(
            ["/usr/libexec/java_home"], capture_output=True, text=True, check=True
        ).stdout.strip()
    response = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=environment,
    )
    output = response.stdout + (response.stderr if response.returncode else "")
    for key, value in environment.items():
        if key.endswith("_API_KEY") and value:
            output = output.replace(value, "[REDACTED]")
    return {"output": output, "returncode": response.returncode}


@pytest.fixture
def public_data(tmp_path: Path) -> Path:
    ids = json.loads((builtin_config_dir / "benchmarks/browsecomp_plus_eval150.json").read_text())["query_ids"]
    (tmp_path / "queries.jsonl").write_text(
        "".join(
            json.dumps({"query_id": iid, "query": "Which city?", "answer": "PRIVATE_GOLD"}) + "\n" for iid in ids[:4]
        )
    )
    return tmp_path


def test_plan_and_cli_are_public_reproducible_and_read_only(public_data: Path):
    config, selection = plan(public_data, [], 3, 42, 12, [])
    assert plan(public_data, [], 3, 42, 12, [])[1] == selection
    assert selection["maximum_active_sandboxes"] == 12
    assert selection["sandbox_ttl_minutes"] == 240
    assert selection["planned_attempts"] == selection["attempts"] == 12
    assert selection["limit"] == 3 and selection["query_ids"] == []
    assert selection["implementation_sha256"] == implementation_hash()
    assert selection["queries_sha256"] == hashlib.sha256(matrix_query_bytes(selection["instances"])).hexdigest()
    assert all(set(row) == {"query_id", "query"} for row in selection["instances"])
    assert "PRIVATE_GOLD" not in json.dumps(selection)
    assert config["model"]["model_name"].startswith("deepseek/")
    output = public_data / "not-created"
    before = sorted(public_data.iterdir())
    result = CliRunner().invoke(
        app,
        [
            "--data-dir",
            str(public_data),
            "--output",
            str(output),
            "-c",
            "browsecomp_plus_deepseek",
            "--limit",
            "3",
            "--seed",
            "42",
            "--workers-per-method",
            "12",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert yaml.safe_load(result.output)["instances"] == selection["instances"]
    assert sorted(public_data.iterdir()) == before and not output.exists()
    assert "PRIVATE_GOLD" not in result.output
    preflight = CliRunner().invoke(
        app,
        [
            "--data-dir",
            str(public_data),
            "--output",
            str(output),
            "--limit",
            "3",
            "--preflight",
            "--dry-run",
        ],
    )
    assert preflight.exit_code == 0, preflight.output
    assert yaml.safe_load(preflight.output)["attempts"] == 0
    assert yaml.safe_load(preflight.output)["planned_attempts"] == 12
    assert sorted(public_data.iterdir()) == before
    with pytest.raises(ValueError, match="1 to 30"):
        plan(public_data, [], 1, 42, 31, [])
    with pytest.raises(ValueError, match="outside saved"):
        plan(public_data, ["browsecomp_plus_deepseek", "model.model_kwargs.api_key=private"], 1, 42, 1, [])


def test_plan_ttl_uses_wall_time_and_preserves_requested_ids_and_tokenizer_revision(public_data: Path):
    ids = [row["query_id"] for row in plan(public_data, [], 0, 42, 1, [])[1]["instances"]]
    config, selection = plan(
        public_data,
        [
            "browsecomp_plus_deepseek",
            "agent.wall_time_limit_seconds=3661",
            "environment.tokenizer_revision=c202fixture",
        ],
        0,
        42,
        1,
        ids,
    )
    assert selection["sandbox_ttl_minutes"] == 4 * 62 + 120
    assert selection["query_ids"] == ids and selection["limit"] == 0
    assert selection["planned_attempts"] == selection["attempts"] == 16
    assert config["environment"]["tokenizer_revision"] == "c202fixture"


@pytest.mark.parametrize(("method",), [(method,) for method in METHODS])  # noqa: PT006
def test_single_method_cli_defaults_to_six_workers_and_keeps_matched_cohort(public_data: Path, method: str):
    ids = json.loads((builtin_config_dir / "benchmarks/browsecomp_plus_eval150.json").read_text())["query_ids"][:8]
    (public_data / "queries.jsonl").write_text(
        "".join(json.dumps({"query_id": iid, "query": "Which city?"}) + "\n" for iid in ids)
    )
    arguments = [
        "--data-dir",
        str(public_data),
        "--output",
        str(public_data / "output"),
        "--method",
        method,
        "--dry-run",
    ]
    result = CliRunner().invoke(app, arguments)
    assert result.exit_code == 0, result.output
    selection = yaml.safe_load(result.stdout)
    assert selection["methods"] == [method]
    assert selection["workers_per_method"] == selection["active_workers_per_method"] == 6
    assert selection["maximum_active_sandboxes"] == 6
    assert selection["attempts"] == selection["planned_attempts"] == 8
    matched = plan(public_data, [], 0, 42, 6, [])[1]
    assert selection["instances"] == matched["instances"]
    assert selection["queries_sha256"] == matched["queries_sha256"]
    preflight = CliRunner().invoke(app, [*arguments, "--preflight", "--workers-per-method", "2"])
    assert preflight.exit_code == 0, preflight.output
    assert yaml.safe_load(preflight.stdout)["attempts"] == 0
    assert yaml.safe_load(preflight.stdout)["planned_attempts"] == 8
    assert yaml.safe_load(preflight.stdout)["maximum_active_sandboxes"] == 2
    assert not (public_data / "output").exists()


def test_unknown_method_is_rejected_before_data_or_cloud_access(tmp_path: Path):
    with pytest.raises(ValueError, match="Unknown method"):
        plan(tmp_path / "missing", [], 0, 42, 6, [], method="typo")
    assert not list(tmp_path.iterdir())


def test_corpus_params_use_only_public_environment_and_no_mounts():
    pytest.importorskip("daytona")
    params = worker_params("worker", "existing", "react", RuntimeConfig(cpu=1, memory=4, disk=10, ttl_minutes=600))
    assert params.env_vars == PUBLIC_ENV and params.volumes is None and params.secrets is None
    assert not any(key.endswith("_API_KEY") for key in params.env_vars)
    assert params.network_block_all and params.ephemeral is False and params.public is False
    assert (params.auto_stop_interval, params.auto_pause_interval, params.auto_delete_interval) == (0, 0, -1)
    assert params.ttl_minutes == 600 and params.labels["role"] == "corpus-worker"


def test_fresh_output_is_required_before_api_access(public_data: Path):
    output = public_data / "existing"
    output.mkdir()
    (output / "result.json").write_text("recorded")
    with pytest.raises(ValueError, match="fresh output"):
        run_matrix(data_dir=public_data, output=output, config_spec=[], limit=1, seed=42, workers_per_method=1)
    assert (output / "result.json").read_text() == "recorded"


@pytest.mark.parametrize(
    ("payload", "exit_code", "operation", "arguments", "error"),
    [
        ("[]", 7, "search", ["jazz", 1], RuntimeError),
        ("not json", 0, "search", ["jazz", 1], ValueError),
        ('[{"docid":"city","text":"x","score":"bad"}]', 0, "search", ["jazz", 1], ValueError),
        ('[{"docid":"city","text":"x","score":NaN}]', 0, "search", ["jazz", 1], ValueError),
        ('{"docid":"other","text":"x"}', 0, "get_document", ["city"], ValueError),
    ],
)
def test_command_validates_actual_process_exit_and_json(payload, exit_code, operation, arguments, error):
    def backend(command: str, *, timeout: int) -> dict:
        assert timeout == 120
        assert shlex.split(command)[-2] == RETRIEVAL_CODE
        return subprocess_backend(
            shlex.join(
                [
                    sys.executable,
                    "-c",
                    "import sys; print(sys.argv[1]); sys.exit(int(sys.argv[2]))",
                    payload,
                    str(exit_code),
                ]
            ),
            timeout=timeout,
        )

    with pytest.raises(error):
        RemoteRetriever(backend)._command(operation, arguments)
    with pytest.raises(ValueError, match="Unsupported"):
        RemoteRetriever(backend)._command("__import__", [])


def native_output(command: str) -> dict:
    return make_toolcall_output(
        "Evidence",
        [
            {
                "id": "call",
                "type": "function",
                "function": {"name": "bash", "arguments": json.dumps({"command": command})},
            }
        ],
        [{"command": command, "tool_call_id": "call"}],
    )


def deterministic_config() -> dict:
    return {
        "agent": {"step_limit": 0, "cost_limit": 0},
        "environment": {"tokenizer_name": "", "snippet_tokens": 0, "seen_snippet_tokens": 0, "document_tokens": 0},
        "model": {
            "model_name": "deepseek/replay",
            "model_class": "pocketcontextagent.models.test_models.DeterministicToolcallModel",
            "outputs": [native_output("submit Explanation: Jazz [city].\nExact Answer: Montréal\nConfidence: 100%")],
        },
    }


def test_real_bm25_transport_and_separate_method_archives_combine(bm25_index: Path, public_data: Path):
    def corpus_backend(command: str, *, timeout: int) -> dict:
        response = subprocess_backend(command, timeout=timeout)
        assert response["returncode"] == 0, response["output"]
        return response

    retriever = RemoteRetriever(corpus_backend, bm25_index)
    hits = retriever.search("jazz festival", 1)
    assert len(hits) == 1 and hits[0]["docid"] == "city" and "Montréal" in hits[0]["text"]
    assert retriever.get_document("city") == {"docid": "city", "text": hits[0]["text"]}
    assert retriever.get_document("missing") is None
    marker = public_data / "SHELL_INJECTION"
    assert retriever.get_document(f"missing'; touch {marker}; #") is None
    assert not marker.exists()
    with pytest.raises(ValueError, match="positive k"):
        retriever.search("jazz", 0)
    replay = public_data / "replay.yaml"
    replay_config = deterministic_config()
    replay_config["model"]["outputs"][:0] = [native_output("search jazz festival"), native_output("get_document city")]
    replay.write_text(yaml.safe_dump(replay_config))
    config, selection = plan(public_data, [str(replay)], 1, 42, 1, [])
    iid = selection["selected_query_ids"][0]
    root = public_data / "runs"
    root.mkdir()
    store = ArtifactStore(public_data / "store")
    for method in METHODS:
        config, selection = plan(public_data, [str(replay)], 1, 42, 6, [], method=method)
        manifest = run_manifest(config, selection, method, {"name": "public-index", "id": "immutable-id"})
        controller = public_data / f"controller-{method}"
        task = controller / method / "tasks" / iid
        task.mkdir(parents=True)
        (task / "run.yaml").write_text(yaml.safe_dump(manifest))
        fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        assert run_instance(selection["instances"][0], task, manifest["config"], retriever, fingerprint)
        result = json.loads((task / f"{iid}.json").read_text())
        assert result["tool_call_counts"] == {"search": 1, "get_document": 1}
        assert result["retrieved_docids"] == ["city"] and result["execution_error"] is None
        archive_trajectories(task)
        publish_matrix_task(task, controller / method, iid, manifest)
        store.archive_task(task, f"job-{method}", method, iid, prune=True)
        assert store.task_receipt(f"job-{method}", method, iid)
        assert not (task / "trajectories").exists() and (task / f"{iid}.json").exists()
        report = compare_runs(controller, methods=selection["methods"])
        assert list(report["methods"]) == list(report["method_provenance"]) == [method]
        assert report["methods"][method]["completed"] == 1
        (root / method).symlink_to(controller / method, target_is_directory=True)
    report = compare_runs(root)
    assert all(row["tasks"] == row["completed"] == 1 for row in report["methods"].values())
    assert "PRIVATE_GOLD" not in "".join(path.read_text() for path in root.glob("*/run.yaml"))
    retriever.close()


def test_runtime_failure_has_result_and_trajectory_without_invented_usage(public_data: Path):
    replay_config = deterministic_config()
    replay_config["model"]["outputs"] = []
    replay = public_data / "failure.yaml"
    replay.write_text(yaml.safe_dump(replay_config))
    config, selection = plan(public_data, [str(replay)], 1, 42, 1, [])
    instance = selection["instances"][0]
    manifest = run_manifest(config, selection, "pocket_context", {"name": "public", "id": "id"})
    task = public_data / "attempt"
    task.mkdir()
    (task / "run.yaml").write_text(yaml.safe_dump(manifest))
    fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    with pytest.raises(IndexError):
        run_instance(instance, task, manifest["config"], RemoteRetriever(subprocess_backend), fingerprint)
    result = json.loads((task / f"{instance['query_id']}.json").read_text())
    trajectory = json.loads((task / "trajectories" / instance["query_id"] / "trajectory.json").read_text())
    assert result["execution_error"] == result["status"] == "IndexError"
    assert trajectory["info"]["execution_error"] == "IndexError"
    assert result["metrics"]["cost_usd"] is None and result["model_stats"]["cost_available"] is False
    archive_trajectories(task)
    publish_matrix_task(task, public_data / "pocket_context", instance["query_id"], manifest)
    store = ArtifactStore(public_data / "store")
    receipt = store.archive_task(task, "failed", "pocket_context", instance["query_id"], prune=True)
    store.verify(receipt)
    assert (
        not (task / "trajectories").exists()
        and (public_data / "pocket_context" / f"{instance['query_id']}.json").is_file()
    )
