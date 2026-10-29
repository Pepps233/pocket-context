import json
from pathlib import Path
from shlex import split

import pytest

from pocketcontextagent.run.benchmarks.utils.swebench_verified import (
    DATASET_REVISION,
    HARNESS_REVISION,
    PUBLIC_FIELDS,
    evaluator_command,
    export_predictions,
    import_official_results,
    instance_image,
    prepare_dataset,
    public_task,
    write_public_tasks,
)


def instance(iid: str = "django__django-123") -> dict:
    return {
        "instance_id": iid,
        "problem_statement": "Public issue text",
        "repo": "django/django",
        "base_commit": "a" * 40,
        "version": "3.0",
        "patch": "SECRET GOLD",
        "test_patch": "SECRET TESTS",
        "FAIL_TO_PASS": "SECRET LABELS",
        "PASS_TO_PASS": "SECRET LABELS",
        "hints_text": "PRIVATE HINTS",
        "future_private_field": "SECRET",
        "image_name": "untrusted:image",
    }


def prediction(iid: str = "django__django-123", patch: str = "diff --git a/x b/x\n+fix\n") -> dict:
    return {"instance_id": iid, "model_name_or_path": "org/model", "model_patch": patch}


def test_public_tasks_allowlist_canonical_image_and_immutable_provenance(tmp_path):
    path = tmp_path / "tasks.jsonl"
    assert write_public_tasks([instance()], path) == [public_task(instance())]
    task = json.loads(path.read_text())
    assert set(task) == set(PUBLIC_FIELDS)
    assert task["query_id"] == task["instance_id"] == "django__django-123"
    assert task["query"] == task["problem_statement"] == "Public issue text"
    assert task["image"] == "docker.io/swebench/sweb.eval.x86_64.django_1776_django-123:latest"
    assert "SECRET" not in path.read_text() and "PRIVATE" not in path.read_text()
    assert len(DATASET_REVISION) == len(HARNESS_REVISION) == 40
    assert instance_image("Django__Django-123") == task["image"]
    with pytest.raises(ValueError, match="Duplicate"):
        write_public_tasks([instance(), instance()], path)
    with pytest.raises(ValueError, match="overwrite"):
        prepare_dataset(path, grading_path=path)
    assert json.loads(path.read_text()) == task


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("instance_id", "../bad__path"),
        ("instance_id", "repo;echo x"),
        ("base_commit", "HEAD"),
        ("version", ""),
        ("problem_statement", ""),
    ],
)
def test_invalid_public_data_rejected(field, value):
    with pytest.raises(ValueError):
        public_task(instance() | {field: value})


def test_prediction_export_preserves_failed_denominator_and_discards_prose(tmp_path):
    path = tmp_path / "predictions.jsonl"
    rows = export_predictions(
        [
            prediction() | {"submission": "Agent prose", "exit_status": "Submitted"},
            prediction("sympy__sympy-2", "") | {"exit_status": "LimitsExceeded"},
        ],
        path,
    )
    assert [json.loads(line) for line in path.read_text().splitlines()] == rows
    assert len(rows) == 2 and rows[1]["model_patch"] == ""
    assert all(set(row) == {"model_name_or_path", "instance_id", "model_patch"} for row in rows)
    assert "Agent prose" not in path.read_text()
    with pytest.raises(ValueError, match="duplicate"):
        export_predictions([prediction(), prediction()], path)
    with pytest.raises(ValueError, match="string"):
        export_predictions([prediction() | {"model_patch": None}], path)
    with pytest.raises(ValueError, match="separately"):
        export_predictions([prediction(), prediction("sympy__sympy-2") | {"model_name_or_path": "other"}], path)


def test_official_command_uses_private_pinned_grading_file_and_image_contract(tmp_path):
    command = split(
        evaluator_command(
            tmp_path / "predictions with spaces.jsonl", "unique_run", grading_dataset_path=tmp_path / "private.jsonl"
        )
    )
    assert command[:3] == ["python", "-m", "swebench.harness.run_evaluation"]
    assert (
        split(
            evaluator_command(
                tmp_path / "preds.jsonl",
                "safe",
                grading_dataset_path=tmp_path / "private",
                python_executable="/grading venv/bin/python",
            )
        )[0]
        == "/grading venv/bin/python"
    )
    assert command[command.index("--dataset_name") + 1] == str(tmp_path / "private.jsonl")
    assert command[command.index("--predictions_path") + 1] == str(tmp_path / "predictions with spaces.jsonl")
    assert command[command.index("--namespace") + 1] == "swebench"
    assert command[command.index("--instance_image_tag") + 1] == "latest"
    with pytest.raises(ValueError, match="safe run_id"):
        evaluator_command(Path("preds"), "../old", grading_dataset_path=Path("private"))


def write_report(root: Path, row: dict, *, resolved: bool = True) -> Path:
    directory = root / "org__model" / row["instance_id"]
    directory.mkdir(parents=True)
    (directory / "patch.diff").write_text(row["model_patch"])
    (directory / "report.json").write_text(
        json.dumps(
            {
                row["instance_id"]: {
                    "patch_exists": True,
                    "patch_successfully_applied": True,
                    "resolved": resolved,
                    "tests_status": {
                        "FAIL_TO_PASS": {"success": ["regression"], "failure": []},
                        "PASS_TO_PASS": {"success": ["existing"], "failure": []},
                    },
                }
            }
        )
    )
    return directory


def test_result_import_binds_actual_patch_id_and_full_attempt_denominator(tmp_path):
    rows = [prediction(), prediction("sympy__sympy-2", ""), prediction("astropy__astropy-3")]
    directory = write_report(tmp_path, rows[0])
    write_report(tmp_path, rows[2], resolved=False)
    result = import_official_results(rows, tmp_path)
    assert result["attempted"] == 3 and result["evaluated"] == 2 and result["tests_executed"] == 2
    assert result["resolved"] == 1 and result["resolved_rate"] == 1 / 3
    assert result["evaluation_coverage"] == 2 / 3
    assert result["instances"][1] == {
        "instance_id": "sympy__sympy-2",
        "evaluated": False,
        "tests_executed": False,
        "resolved": False,
    }
    (directory / "patch.diff").write_text("different patch")
    with pytest.raises(ValueError, match="Patch mismatch"):
        import_official_results(rows, tmp_path)
    (directory / "patch.diff").write_text(rows[0]["model_patch"])
    (directory / "report.json").write_text(json.dumps({"wrong__id-1": {"resolved": True}}))
    with pytest.raises(ValueError, match="ID mismatch"):
        import_official_results(rows, tmp_path)


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ({"resolved": "true"}, "Malformed official"),
        ({"patch_successfully_applied": False}, "lacks grading evidence"),
        ({"tests_status": {}}, "Malformed test coverage"),
        ({"tests_status": None}, "lacks grading evidence"),
        (
            {
                "tests_status": {
                    "FAIL_TO_PASS": {"success": ["test"], "failure": ["other"]},
                    "PASS_TO_PASS": {"success": [], "failure": []},
                }
            },
            "lacks grading evidence",
        ),
    ],
)
def test_result_import_rejects_resolved_without_real_grading_coverage(tmp_path, change, error):
    row = prediction()
    directory = write_report(tmp_path, row)
    report = json.loads((directory / "report.json").read_text())
    report[row["instance_id"]] |= change
    (directory / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match=error):
        import_official_results([row], tmp_path)
