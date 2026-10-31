import hashlib
import json
import subprocess

import pytest

from pocketcontextagent.run.benchmarks.swebench_daytona_evaluate import (
    grading_preflight_command,
    load_inputs,
    publish_journal,
    publish_report,
)
from pocketcontextagent.run.benchmarks.utils.swebench_daytona import ResourceJournal
from pocketcontextagent.run.benchmarks.utils.swebench_verified import import_official_results


def prediction() -> dict:
    return {"instance_id": "psf__requests-1", "model_name_or_path": "org/model", "model_patch": "a patch\n"}


def test_grading_preserves_official_setup_commit_and_rejects_other_states(tmp_path):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(tmp_path), *args], text=True).strip()

    def preflight(base):
        return subprocess.run(["bash", "-c", grading_preflight_command(base)], cwd=tmp_path).returncode

    git("init")
    git("config", "user.name", "SWE-bench")
    git("config", "user.email", "setup@swebench.config")
    (tmp_path / "tox.ini").write_text("pytest\n")
    git("add", "tox.ini")
    git("commit", "-m", "base")
    base = git("rev-parse", "HEAD")
    assert preflight(base) == 0
    (tmp_path / "tox.ini").write_text("pytest -rA\n")
    assert preflight(base) != 0
    git("commit", "-am", "SWE-bench")
    assert preflight(base) == 0
    assert (tmp_path / "tox.ini").read_text() == "pytest -rA\n"
    assert preflight("a" * 40) != 0
    git("commit", "--allow-empty", "-m", "unexpected")
    assert preflight(base) != 0


def test_published_journal_preserves_local_recovery_evidence(tmp_path):
    directory = tmp_path / "volume"
    directory.mkdir()
    journal = ResourceJournal(tmp_path / "local" / "sandbox-cleanup.json")
    publish_journal(journal, directory)
    assert not (directory / "sandbox-cleanup.json").exists()
    journal.record("snapshots", "snapshot", phase="creating")
    publish_journal(journal, directory)
    journal.record("sandboxes", "sandbox", id="sandbox-id", phase="running")
    assert json.loads((directory / "sandbox-cleanup.json").read_text())["sandboxes"] == []
    publish_journal(journal, directory)
    assert json.loads((directory / "sandbox-cleanup.json").read_text()) == journal.document()
    assert journal.path.read_bytes() == (directory / "sandbox-cleanup.json").read_bytes()


def test_grading_inputs_bind_exact_bytes_and_reject_duplicates_and_unsafe_paths(tmp_path):
    dataset, predictions = tmp_path / "private.jsonl", tmp_path / "predictions.jsonl"
    instance = {
        "instance_id": "psf__requests-1",
        "repo": "psf/requests",
        "base_commit": "a" * 40,
        "problem_statement": "An issue",
        "version": "2.19",
        "patch": "PRIVATE GOLD",
    }
    dataset.write_text(json.dumps(instance) + "\n")
    predictions.write_text(json.dumps(prediction()) + "\n")
    dataset_hash = hashlib.sha256(dataset.read_bytes()).hexdigest()
    prediction_hash = hashlib.sha256(predictions.read_bytes()).hexdigest()
    assert load_inputs(dataset, predictions, dataset_hash, prediction_hash) == [(instance, prediction())]
    predictions.write_text(predictions.read_text() * 2)
    with pytest.raises(ValueError, match="checksum"):
        load_inputs(dataset, predictions, dataset_hash, prediction_hash)
    with pytest.raises(ValueError, match="Duplicate"):
        load_inputs(dataset, predictions, dataset_hash, hashlib.sha256(predictions.read_bytes()).hexdigest())
    predictions.write_text(json.dumps(prediction() | {"model_name_or_path": ".."}))
    with pytest.raises(ValueError, match="Unsafe"):
        load_inputs(dataset, predictions, dataset_hash, hashlib.sha256(predictions.read_bytes()).hexdigest())


@pytest.fixture
def spec():
    pytest.importorskip("swebench")
    from swebench.harness.test_spec.test_spec import TestSpec

    return TestSpec(
        instance_id="psf__requests-1",
        repo="psf/requests",
        version="2.19",
        repo_script_list=[],
        eval_script_list=[],
        env_script_list=[],
        arch="x86_64",
        FAIL_TO_PASS=["test_regression"],
        PASS_TO_PASS=["test_existing"],
        language="py",
        docker_specs={},
        namespace="swebench",
    )


@pytest.mark.parametrize(
    ("test_output", "resolved", "executed"),
    [
        ("PASSED test_regression\nPASSED test_existing\n", True, True),
        ("FAILED test_regression\nPASSED test_existing\n", False, True),
        ("PASSED test_regression\nFAILED test_existing\n", False, True),
        ("PASSED test_existing\n", False, True),
    ],
)
def test_official_parser_reports_round_trip_without_docker(tmp_path, spec, test_output, resolved, executed):
    from swebench.harness.constants import END_TEST_OUTPUT, START_TEST_OUTPUT

    directory = tmp_path / "org__model" / prediction()["instance_id"]
    directory.mkdir(parents=True)
    (directory / "patch.diff").write_text(prediction()["model_patch"])
    (directory / "test_output.txt").write_text(f"{START_TEST_OUTPUT}\n{test_output}{END_TEST_OUTPUT}\n")
    assert publish_report(spec, prediction(), directory, 0)[spec.instance_id]["resolved"] is resolved
    result = import_official_results([prediction()], tmp_path)
    assert result["resolved"] == int(resolved)
    assert result["tests_executed"] == int(executed)
    assert result["attempted"] == result["evaluated"] == 1


def test_missing_test_output_stays_unknown_instead_of_becoming_an_agent_failure(tmp_path, spec):
    directory = tmp_path / "org__model" / prediction()["instance_id"]
    directory.mkdir(parents=True)
    (directory / "patch.diff").write_text(prediction()["model_patch"])
    (directory / "test_output.txt").write_text("test setup failed\n")
    with pytest.raises(RuntimeError, match="did not reach test output"):
        publish_report(spec, prediction(), directory, 0)
    assert not (directory / "report.json").exists()
    assert import_official_results([prediction()], tmp_path)["evaluated"] == 0


@pytest.mark.parametrize(("exit_code", "error"), [(124, TimeoutError), (137, TimeoutError), (127, RuntimeError)])
def test_timeouts_and_launch_errors_never_publish_grades(tmp_path, spec, exit_code, error):
    with pytest.raises(error, match="no grade published"):
        publish_report(spec, prediction(), tmp_path, exit_code)
    assert not (tmp_path / "report.json").exists()
    assert json.loads((tmp_path / "status.json").read_text())["exit_code"] == exit_code
