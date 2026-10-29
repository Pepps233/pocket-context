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

from pocketcontextagent.run.benchmarks.benchmark_controller import app
from pocketcontextagent.run.benchmarks.browsecomp_plus_daytona import app as matrix_app
from pocketcontextagent.run.benchmarks.utils import swebench_verified as swe

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
