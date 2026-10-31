"""Pinned SWE-bench Verified public inputs and separate official grading artifacts.

Evidence:
* Dataset SHA: https://huggingface.co/api/datasets/princeton-nlp/SWE-bench_Verified
* Immutable card: https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified/blob/c104f840cc67f8b6eec6f759ebc8b2693d585d4a/README.md
* Official image naming (x86_64, __ -> _1776_, swebench namespace):
  https://github.com/SWE-bench/SWE-bench/blob/726c5461e2ef52d83cf1ea2107870a8bb3328d57/swebench/harness/test_spec/test_spec.py#L106-L115
* Prediction application, patch.diff/report.json and evaluator CLI:
  https://github.com/SWE-bench/SWE-bench/blob/726c5461e2ef52d83cf1ea2107870a8bb3328d57/swebench/harness/run_evaluation.py
* Official resolved = full FAIL_TO_PASS and PASS_TO_PASS success:
  https://github.com/SWE-bench/SWE-bench/blob/726c5461e2ef52d83cf1ea2107870a8bb3328d57/swebench/harness/grading.py

Contracts target official SWE-bench v4.1.0 (commit above). Instance images use
mutable :latest tags, NOT immutable digests; record the provisioner's resolved
image digest separately when reproducibility requires it. Coding uses that image
directly in Daytona, without nested Docker. Grading uses the official test scripts
and parser with fresh Daytona grading sandboxes, never the coding sandbox.

The v4.1.0 evaluator has no dataset revision CLI flag. To grade the exact pinned
snapshot, optionally write a PRIVATE grading_dataset.jsonl, pass it as
--dataset_name, and never mount/copy it into a coding sandbox or agent input.
Only tasks.jsonl (the explicit public allowlist below) is safe for agents.
"""

import json
import re
from collections.abc import Iterable
from pathlib import Path
from shlex import join

import typer

DATASET_NAME = "princeton-nlp/SWE-bench_Verified"
DATASET_REVISION = "c104f840cc67f8b6eec6f759ebc8b2693d585d4a"
DATASET_SPLIT = "test"
HARNESS_REVISION = "726c5461e2ef52d83cf1ea2107870a8bb3328d57"
PUBLIC_FIELDS = ("query_id", "instance_id", "query", "problem_statement", "repo", "base_commit", "version", "image")


def _instance_id(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+__[A-Za-z0-9_.-]+", value):
        raise ValueError(f"Invalid SWE-bench instance ID: {value!r}")
    return value


def instance_image(instance_id: str) -> str:
    """Canonical official x86_64 image, not a row-supplied arbitrary image."""
    return f"docker.io/swebench/sweb.eval.x86_64.{_instance_id(instance_id).replace('__', '_1776_')}:latest".lower()


def public_task(instance: dict) -> dict:
    """Allowlist rather than blacklist: no gold patch, tests, hints or labels."""
    iid = _instance_id(instance["instance_id"])
    if not re.fullmatch(r"[0-9a-f]{40}", instance["base_commit"]):
        raise ValueError(f"Invalid base commit for {iid}")
    if any(not isinstance(instance[key], str) or not instance[key] for key in ("problem_statement", "repo", "version")):
        raise ValueError(f"Missing public task fields for {iid}")
    return {
        "query_id": iid,
        "instance_id": iid,
        "query": instance["problem_statement"],
        "problem_statement": instance["problem_statement"],
        "repo": instance["repo"],
        "base_commit": instance["base_commit"],
        "version": instance["version"],
        "image": instance_image(iid),
    }


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def write_public_tasks(instances: Iterable[dict], tasks_path: Path) -> list[dict]:
    tasks = [public_task(instance) for instance in instances]
    if len({task["instance_id"] for task in tasks}) != len(tasks):
        raise ValueError("Duplicate dataset instance IDs")
    _write_jsonl(tasks_path, tasks)
    return tasks


def prepare_dataset(tasks_path: Path, *, grading_path: Path | None = None) -> list[dict]:
    """Download pinned public data, with optional SEPARATE trusted grading copy.

    This function runs on the orchestrator, not in a coding sandbox. Omit
    grading_path unless provisioning a separate trusted grading environment.
    """
    if grading_path is not None and grading_path.resolve() == tasks_path.resolve():
        raise ValueError("Private grading data must not overwrite public tasks")
    from datasets import load_dataset

    instances = list(load_dataset(DATASET_NAME, revision=DATASET_REVISION, split=DATASET_SPLIT))
    tasks = write_public_tasks(instances, tasks_path)
    if grading_path is not None:
        _write_jsonl(grading_path, instances)
        grading_path.chmod(0o600)
    return tasks


def prediction(attempt: dict) -> dict:
    """Keep failed/empty attempts; never substitute submission prose for a patch."""
    iid = _instance_id(attempt["instance_id"])
    if not isinstance(attempt["model_patch"], str):
        raise ValueError("model_patch must be a string (use an empty patch for failed attempts)")
    if not isinstance(attempt["model_name_or_path"], str) or not attempt["model_name_or_path"]:
        raise ValueError("model_name_or_path must be nonempty")
    return {key: attempt[key] for key in ("model_name_or_path", "model_patch")} | {"instance_id": iid}


def export_predictions(attempts: Iterable[dict], path: Path) -> list[dict]:
    rows = [prediction(attempt) for attempt in attempts]
    if len({row["instance_id"] for row in rows}) != len(rows):
        raise ValueError("Export one prediction per instance and model run; duplicate instance IDs found")
    if len({row["model_name_or_path"] for row in rows}) > 1:
        raise ValueError("Export each model run separately")
    _write_jsonl(path, rows)
    return rows


def evaluator_command(
    predictions_path: Path,
    run_id: str,
    *,
    grading_dataset_path: Path,
    dataset_sha256: str,
    predictions_sha256: str,
    max_workers: int = 1,
    timeout: int = 1800,
    python_executable: str = "python",
) -> str:
    """Run outside coding sandboxes after installing swebench at HARNESS_REVISION.

    grading_dataset_path must be the private pinned file from prepare_dataset,
    NOT sanitized tasks.jsonl. The Daytona adapter rejects existing run directories.
    import_official_results also verifies every report's saved patch.
    """
    if (
        not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", run_id)
        or not 1 <= max_workers <= 6
        or timeout <= 0
        or not python_executable.strip()
    ):
        raise ValueError("Use a nonempty safe run_id and positive workers/timeout")
    if any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in (dataset_sha256, predictions_sha256)):
        raise ValueError("Use the exported dataset and prediction SHA-256 checksums")
    return join(
        [
            python_executable,
            "-m",
            "pocketcontextagent.run.benchmarks.swebench_daytona_evaluate",
            "--dataset_name",
            str(grading_dataset_path),
            "--predictions_path",
            str(predictions_path),
            "--run_id",
            run_id,
            "--dataset-sha256",
            dataset_sha256,
            "--predictions-sha256",
            predictions_sha256,
            "--max_workers",
            str(max_workers),
            "--timeout",
            str(timeout),
        ]
    )


def import_official_results(predictions: Iterable[dict], run_log_dir: Path) -> dict:
    """Import logs/run_evaluation/RUN_ID, binding reports to exact patches/IDs.

    Missing/failed evaluations remain in the attempted denominator as unresolved.
    Coverage is explicit; neither Submitted nor patch application implies success.
    Reports and patch.diff must come from the same trusted official harness run.
    This verifies artifact consistency, not authenticity against malicious logs.
    """
    rows = [prediction(row) for row in predictions]
    if len({row["instance_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate prediction instance IDs")
    outcomes = []
    for row in rows:
        model_dir = row["model_name_or_path"].replace("/", "__")
        if model_dir in (".", "..") or "\\" in model_dir:
            raise ValueError("Unsafe model name in report path")
        directory = run_log_dir / model_dir / row["instance_id"]
        report_path = directory / "report.json"
        outcome = {"instance_id": row["instance_id"], "evaluated": False, "tests_executed": False, "resolved": False}
        if report_path.exists():
            if (directory / "patch.diff").read_text() != row["model_patch"]:
                raise ValueError(f"Patch mismatch for {row['instance_id']}; refuse stale grading")
            report = json.loads(report_path.read_text())
            if set(report) != {row["instance_id"]}:
                raise ValueError(f"Report instance ID mismatch for {row['instance_id']}")
            result = report[row["instance_id"]]
            if any(
                type(result.get(key)) is not bool for key in ("resolved", "patch_exists", "patch_successfully_applied")
            ):
                raise ValueError(f"Malformed official report for {row['instance_id']}")
            tests = result.get("tests_status")
            tested = result["patch_successfully_applied"] and isinstance(tests, dict)
            if tested:
                for group in ("FAIL_TO_PASS", "PASS_TO_PASS"):
                    if not isinstance(tests.get(group), dict) or any(
                        not isinstance(tests[group].get(status), list) for status in ("success", "failure")
                    ):
                        raise ValueError(f"Malformed test coverage for {row['instance_id']}")
            if result["resolved"] and (
                not tested
                or not result["patch_exists"]
                or not row["model_patch"].strip()
                or not tests["FAIL_TO_PASS"]["success"]
                or tests["FAIL_TO_PASS"]["failure"]
                or tests["PASS_TO_PASS"]["failure"]
            ):
                raise ValueError(f"Resolved report lacks grading evidence for {row['instance_id']}")
            outcome |= {"evaluated": True, "tests_executed": tested, "resolved": result["resolved"]}
        outcomes.append(outcome)
    attempted = len(rows)
    return {
        "attempted": attempted,
        "evaluated": sum(outcome["evaluated"] for outcome in outcomes),
        "tests_executed": sum(outcome["tests_executed"] for outcome in outcomes),
        "resolved": sum(outcome["resolved"] for outcome in outcomes),
        "evaluation_coverage": sum(outcome["evaluated"] for outcome in outcomes) / attempted if attempted else 0.0,
        "resolved_rate": sum(outcome["resolved"] for outcome in outcomes) / attempted if attempted else 0.0,
        "instances": outcomes,
    }


app = typer.Typer(add_completion=False)


@app.command()
def prepare(tasks_path: Path = Path("tasks.jsonl"), grading_path: Path | None = None) -> None:
    """Prepare public tasks; optional grading_path is private and must stay outside sandboxes."""
    prepare_dataset(tasks_path, grading_path=grading_path)


if __name__ == "__main__":
    app()
