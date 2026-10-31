import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from pocketcontextagent.comparisons.registry import METHODS
from pocketcontextagent.config import builtin_config_dir
from pocketcontextagent.run.benchmarks.benchmark_controller import app, submission_plan, validate_remote_plan
from pocketcontextagent.run.benchmarks.browsecomp_plus_daytona import app as matrix_app
from pocketcontextagent.run.benchmarks.utils import swebench_verified as swe
from pocketcontextagent.run.benchmarks.utils.controller_browsecomp import app as corpus_app

PRIVATE = "PRIVATE_GRADING_MARKER_92831"


@pytest.fixture
def prepared(tmp_path: Path) -> Path:
    data = tmp_path / "prepared"
    data.mkdir()
    rows = [
        {
            "instance_id": f"fixture__repo-{index}",
            "repo": "fixture/repo",
            "base_commit": "a" * 40,
            "version": "1",
            "problem_statement": f"Public issue {index}",
            "patch": PRIVATE,
            "test_patch": PRIVATE,
            "FAIL_TO_PASS": [PRIVATE],
            "PASS_TO_PASS": [PRIVATE],
        }
        for index in range(9)
    ]
    swe.write_public_tasks(rows, data / "queries.jsonl")
    (data / "grading_dataset.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (data / "manifest.json").write_text(
        json.dumps(
            {
                "dataset": swe.DATASET_NAME,
                "dataset_revision": swe.DATASET_REVISION,
                "harness_revision": swe.HARNESS_REVISION,
                "split": "test",
                "queries_sha256": hashlib.sha256((data / "queries.jsonl").read_bytes()).hexdigest(),
                "grading_dataset_sha256": hashlib.sha256((data / "grading_dataset.jsonl").read_bytes()).hexdigest(),
            }
        )
    )
    return data


def credential_free_env() -> dict:
    return {key: None for key in os.environ if key.endswith("_API_KEY")} | {"MSWEA_SILENT_STARTUP": "1"}


def test_context_window_survives_frozen_config_upload(prepared: Path, tmp_path: Path):
    config, plan = submission_plan(prepared, [], 4, 42, 6, [], context_window=64000)
    frozen = tmp_path / "frozen.yaml"
    frozen.write_text(yaml.safe_dump(config))
    result = CliRunner().invoke(
        matrix_app,
        [
            "matrix",
            "--benchmark",
            "swebench-verified",
            "--data-dir",
            str(prepared),
            "--limit",
            "4",
            "--seed",
            "42",
            "--workers-per-method",
            "6",
            "-c",
            str(frozen),
            "--dry-run",
        ],
        env=credential_free_env(),
    )
    assert result.exit_code == 0, result.output
    remote = yaml.safe_load(result.stdout)
    assert remote["context_token_limit"] == 64000
    plan.update(attempts=plan["planned_attempts"], preflight_only=False)
    validate_remote_plan(plan, remote)
    with pytest.raises(ValueError, match="context_token_limit"):
        validate_remote_plan(plan, remote | {"context_token_limit": 128000})


@pytest.fixture
def prepared_corpus(tmp_path: Path) -> Path:
    ids = json.loads((builtin_config_dir / "benchmarks/browsecomp_plus_eval150.json").read_text())["query_ids"][:8]
    (tmp_path / "queries.jsonl").write_text(
        "".join(json.dumps({"query_id": iid, "query": "Which city?", "answer": PRIVATE}) + "\n" for iid in ids)
    )
    return tmp_path


@pytest.mark.parametrize(
    ("method", "preflight"), [(method, preflight) for method in METHODS for preflight in (True, False)]
)
def test_single_method_submission_matches_cloud_cli_without_resources(
    prepared_corpus: Path, method: str, preflight: bool
):
    arguments = [
        "--data-dir",
        str(prepared_corpus),
        "--limit",
        "0",
        "--seed",
        "42",
        "--method",
        method,
        "--context-window",
        "64000",
        "--dry-run",
        *(["--preflight"] if preflight else []),
    ]
    submitted = CliRunner().invoke(
        app,
        ["submit", "--benchmark", "browsecomp-plus", "--manifest", str(prepared_corpus / "missing.json"), *arguments],
        env=credential_free_env(),
    )
    remote = CliRunner().invoke(
        corpus_app, ["--output", str(prepared_corpus / "output"), *arguments], env=credential_free_env()
    )
    assert submitted.exit_code == remote.exit_code == 0, submitted.output + remote.output
    local_plan, remote_plan = yaml.safe_load(submitted.stdout), yaml.safe_load(remote.stdout)
    validate_remote_plan(local_plan, remote_plan)
    assert local_plan["context_token_limit"] == remote_plan["context_token_limit"] == 64000
    assert local_plan["methods"] == [method]
    assert local_plan["workers_per_method"] == local_plan["active_workers_per_method"] == 6
    assert local_plan["maximum_active_sandboxes"] == 6
    assert local_plan["planned_attempts"] == 8 and local_plan["attempts"] == (0 if preflight else 8)
    assert PRIVATE not in submitted.stdout + remote.stdout
    assert [path.name for path in prepared_corpus.iterdir()] == ["queries.jsonl"]
    for key in (
        "methods",
        "workers_per_method",
        "maximum_active_sandboxes",
        "queries_sha256",
        "implementation_sha256",
        "context_token_limit",
    ):
        with pytest.raises(ValueError, match=key):
            validate_remote_plan(local_plan, remote_plan | {key: None})


def test_corpus_submission_default_and_override_preserve_existing_matrix(prepared_corpus: Path):
    arguments = [
        "submit",
        "--benchmark",
        "browsecomp-plus",
        "--data-dir",
        str(prepared_corpus),
        "--limit",
        "0",
        "--dry-run",
    ]
    all_methods = CliRunner().invoke(app, arguments)
    single = CliRunner().invoke(app, [*arguments, "--method", "react", "--workers-per-method", "2"])
    assert all_methods.exit_code == single.exit_code == 0, all_methods.output + single.output
    matrix, isolated = yaml.safe_load(all_methods.stdout), yaml.safe_load(single.stdout)
    assert matrix["methods"] == list(METHODS) and matrix["workers_per_method"] == 1
    assert matrix["maximum_active_sandboxes"] == 4 and matrix["planned_attempts"] == 32
    assert isolated["methods"] == ["react"] and isolated["maximum_active_sandboxes"] == 2
    assert isolated["planned_attempts"] == 8
    assert matrix["selected_query_ids"] == isolated["selected_query_ids"]
    with pytest.raises(ValueError, match="methods"):
        validate_remote_plan(isolated, matrix)


@pytest.mark.parametrize(
    ("benchmark", "method", "error"),
    [
        ("browsecomp-plus", "typo", "Unknown method"),
        ("swebench-verified", "typo", "Unknown method"),
    ],
)
def test_invalid_method_submission_fails_before_data_or_cloud_access(
    tmp_path: Path, benchmark: str, method: str, error: str
):
    result = CliRunner().invoke(
        app,
        ["submit", "--benchmark", benchmark, "--method", method, "--data-dir", str(tmp_path / "missing")],
        env=credential_free_env(),
    )
    assert isinstance(result.exception, ValueError) and error in str(result.exception)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(("preflight", "workers"), [(False, 30), (True, 2)])
def test_submit_dry_run_matches_seeded_matrix_without_controller_or_credentials(
    prepared: Path, tmp_path: Path, preflight: bool, workers: int
) -> None:
    original = {path.name: path.read_bytes() for path in prepared.iterdir()}
    manifest = tmp_path / "missing-controller.json"
    arguments = [
        "--data-dir",
        str(prepared),
        "--limit",
        "4",
        "--workers-per-method",
        str(workers),
        "--context-window",
        "64000",
        "--dry-run",
        *(["--preflight"] if preflight else []),
    ]
    result = CliRunner().invoke(app, ["submit", "--manifest", str(manifest), *arguments], env=credential_free_env())
    assert result.exit_code == 0, result.output
    plan = yaml.safe_load(result.stdout)
    matrix = CliRunner().invoke(
        matrix_app,
        ["matrix", "--benchmark", "swebench-verified", "--seed", "42", *arguments],
        env=credential_free_env(),
    )
    assert matrix.exit_code == 0, matrix.output
    selection = yaml.safe_load(matrix.stdout)
    validate_remote_plan(plan, selection)
    assert plan["context_token_limit"] == selection["context_token_limit"] == 64000
    assert (
        plan["selected_query_ids"]
        == selection["selected_query_ids"]
        == random.Random(42).sample([f"fixture__repo-{index}" for index in range(9)], 4)
    )
    assert plan["seed"] == 42 and plan["workers_per_method"] == workers
    assert plan["maximum_active_coding_sandboxes"] == selection["maximum_active_sandboxes"] == min(workers, 4) * 4
    assert plan["attempts"] == selection["attempts"] == (0 if preflight else 16)
    assert plan["planned_attempts"] == selection["planned_method_tasks"] == 16
    assert plan["preflight_only"] is preflight
    assert len(plan["implementation_sha256"]) == 64
    assert PRIVATE not in result.stdout + matrix.stdout
    assert not manifest.exists()
    assert {path.name: path.read_bytes() for path in prepared.iterdir()} == original
    assert {path.name for path in tmp_path.iterdir()} == {"prepared"}


@pytest.mark.parametrize(("changed",), [("queries.jsonl",), ("grading_dataset.jsonl",), ("revision",)])  # noqa: PT006
def test_changed_prepared_snapshot_fails_before_manifest_or_api(prepared: Path, tmp_path: Path, changed: str) -> None:
    if changed == "revision":
        path = prepared / "manifest.json"
        path.write_text(json.dumps(json.loads(path.read_text()) | {"dataset_revision": "b" * 40}))
    else:
        path = prepared / changed
        path.write_bytes(path.read_bytes() + b"\n")
    original = {path.name: path.read_bytes() for path in prepared.iterdir()}
    result = CliRunner().invoke(
        app,
        ["submit", "--manifest", str(tmp_path / "missing.json"), "--data-dir", str(prepared)],
        env=credential_free_env(),
    )
    assert isinstance(result.exception, ValueError)
    assert "Prepared public/private task snapshots changed" in str(result.exception)
    assert PRIVATE not in result.stdout
    assert {path.name: path.read_bytes() for path in prepared.iterdir()} == original
    assert not (tmp_path / "missing.json").exists()


@pytest.mark.parametrize(("workers",), [(0,), (31,)])  # noqa: PT006
def test_submit_worker_bounds_are_cli_errors_before_io(tmp_path: Path, workers: int) -> None:
    result = CliRunner().invoke(
        app, ["submit", "--data-dir", str(tmp_path / "missing"), "--workers-per-method", str(workers), "--dry-run"]
    )
    assert result.exit_code == 2
    assert "workers-per-method" in result.output
    assert not list(tmp_path.iterdir())


def test_mini_extra_root_help_is_lazy_and_controller_alias_has_commands(tmp_path: Path) -> None:
    for arguments in (["--help"], ["benchmark-controller", "--help"]):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import runpy,sys; "
                "sys.argv=['mini-extra',*sys.argv[1:]]; "
                "runpy.run_module('pocketcontextagent.run.utilities.mini_extra',run_name='__main__'); "
                "assert sys.argv[1:] != ['--help'] or not "
                "({'daytona','litellm','pocketcontextagent.run.benchmarks.benchmark_controller'} & sys.modules.keys())",
                *arguments,
            ],
            cwd=tmp_path,
            env={key: value for key, value in os.environ.items() if not key.endswith("_API_KEY")}
            | {"MSWEA_SILENT_STARTUP": "1", "LITELLM_LOCAL_MODEL_COST_MAP": "True", "NO_COLOR": "1"},
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "benchmark-controller" in result.stdout
        if arguments[0] == "benchmark-controller":
            assert all(command in result.stdout for command in ("submit", "status", "logs", "fetch", "cancel"))
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    ("method", "preflight"), [(method, preflight) for method in METHODS for preflight in (True, False)]
)
def test_swe_single_method_plan_matches_worker_cli(prepared, method, preflight):
    args = [
        "--data-dir",
        str(prepared),
        "--limit",
        "6",
        "--seed",
        "42",
        "--workers-per-method",
        "6",
        "--method",
        method,
        "--dry-run",
        *(["--preflight"] if preflight else []),
    ]
    submitted = CliRunner().invoke(
        app, ["submit", "--benchmark", "swebench-verified", *args], env=credential_free_env()
    )
    worker = CliRunner().invoke(
        matrix_app, ["matrix", "--benchmark", "swebench-verified", *args], env=credential_free_env()
    )
    assert submitted.exit_code == worker.exit_code == 0, submitted.output + worker.output
    plan, actual = yaml.safe_load(submitted.stdout), yaml.safe_load(worker.stdout)
    validate_remote_plan(plan, actual)
    assert actual["methods"] == [method]
    assert actual["maximum_active_sandboxes"] == plan["maximum_active_coding_sandboxes"] == 6
    assert actual["attempts"] == (0 if preflight else 6)
    assert actual["planned_method_tasks"] == plan["planned_attempts"] == 6
    assert PRIVATE not in submitted.stdout + worker.stdout
    with pytest.raises(ValueError, match="methods"):
        validate_remote_plan(plan, actual | {"methods": list(METHODS)})
