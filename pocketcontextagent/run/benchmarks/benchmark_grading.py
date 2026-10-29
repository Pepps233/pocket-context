"""Export predictions and bind trusted official SWE-bench reports to matrix attempts."""

import hashlib
import json
import re
from pathlib import Path

import typer
import yaml

from pocketcontextagent.comparisons.registry import METHODS
from pocketcontextagent.comparisons.report import compare_runs, comparison_markdown
from pocketcontextagent.run.benchmarks.benchmark_matrix import export_predictions
from pocketcontextagent.run.benchmarks.utils import swebench_verified as swe


def evaluation_binding(runs: Path, settings: dict, run_prefix: str) -> dict:
    return {
        "run_prefix": run_prefix,
        **{
            key: settings[key]
            for key in ("harness_revision", "dataset_revision", "queries_sha256", "grading_dataset_sha256")
        },
        "predictions_sha256": {
            method: hashlib.sha256((runs / method / "predictions.jsonl").read_bytes()).hexdigest() for method in METHODS
        },
        "run_fingerprints": {
            method: hashlib.sha256(
                json.dumps(yaml.safe_load((runs / method / "run.yaml").read_text()), sort_keys=True).encode()
            ).hexdigest()
            for method in METHODS
        },
    }


def export_evaluation(
    runs: Path, data_dir: Path, run_prefix: str, max_workers: int = 1, harness_python: str = "python"
) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", run_prefix):
        raise ValueError("Use a safe, nonempty run prefix beginning with a letter or digit")
    grading = data_dir / "grading_dataset.jsonl"
    if not grading.is_file():
        raise ValueError("Prepare the private pinned grading dataset first; never upload it to coding sandboxes")
    settings = export_predictions(runs)["settings"]
    if (
        hashlib.sha256(grading.read_bytes()).hexdigest() != settings["grading_dataset_sha256"]
        or hashlib.sha256((data_dir / "queries.jsonl").read_bytes()).hexdigest() != settings["queries_sha256"]
    ):
        raise ValueError("Grading data does not match the run's frozen public/private snapshot")
    commands = {
        method: swe.evaluator_command(
            runs / method / "predictions.jsonl",
            f"{run_prefix}-{method}",
            grading_dataset_path=grading,
            max_workers=max_workers,
            python_executable=harness_python,
        )
        for method in METHODS
    }
    (runs / "evaluation_commands.yaml").write_text(
        yaml.safe_dump(evaluation_binding(runs, settings, run_prefix) | {"commands": commands}, sort_keys=False),
        encoding="utf-8",
    )
    for method, command in commands.items():
        typer.echo(f"{method}: {command}")
    return commands


def import_evaluation(runs: Path, logs: Path, run_prefix: str, output: Path) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", run_prefix):
        raise ValueError("Use a safe, nonempty run prefix")
    if any(output.resolve().is_relative_to((runs / method).resolve()) for method in METHODS):
        raise ValueError("Store grades outside individual method result directories")
    settings = export_predictions(runs)["settings"]
    handoff = yaml.safe_load((runs / "evaluation_commands.yaml").read_text())
    if any(handoff.get(key) != value for key, value in evaluation_binding(runs, settings, run_prefix).items()):
        raise ValueError("Official reports must use the matching exported run prefix, predictions, dataset and harness")
    coverage = {}
    for method in METHODS:
        results = {}
        for path in sorted((runs / method).glob("*.json")):
            result = json.loads(path.read_text())
            results[str(result["query_id"])] = result
        predictions = [json.loads(line) for line in (runs / method / "predictions.jsonl").read_text().splitlines()]
        official = swe.import_official_results(predictions, logs / f"{run_prefix}-{method}")
        coverage[method] = {key: value for key, value in official.items() if key not in ("instances", "resolved_rate")}
        coverage[method].update(observed_resolved_rate=official["resolved_rate"], known_outcomes=0)
        destination = output / method
        destination.mkdir(parents=True, exist_ok=True)
        for outcome in official["instances"]:
            result = results[outcome["instance_id"]]
            empty = (
                result.get("patch_capture_status") == "captured"
                and not result["model_patch"].strip()
                and result.get("execution_error") is None
            )
            known = outcome["evaluated"] or empty
            coverage[method]["known_outcomes"] += int(known)
            judgment = {
                "query_id": result["query_id"],
                "response": result["model_patch"],
                "is_completed": result["status"] == "completed",
                "model_info": {"judge": "official-swebench-tests", "harness_revision": settings["harness_revision"]},
                "evaluation_run_id": f"{run_prefix}-{method}",
                "judge_result": {"correct": outcome["resolved"] if known else None, "parse_error": False},
                "grading_basis": "official_harness"
                if outcome["evaluated"]
                else "confirmed_empty_patch"
                if empty
                else "missing_official_report",
                "evaluation": outcome,
                "patch_sha256": hashlib.sha256(result["model_patch"].encode()).hexdigest(),
            }
            (destination / f"{result['query_id']}_eval.json").write_text(
                json.dumps(judgment, indent=2), encoding="utf-8"
            )
        count = official["attempted"]
        coverage[method].update(
            known_outcome_coverage=coverage[method]["known_outcomes"] / count if count else 0.0,
            test_execution_coverage=official["tests_executed"] / count if count else 0.0,
        )
    (output / "coverage.yaml").write_text(yaml.safe_dump(coverage, sort_keys=False), encoding="utf-8")
    report = compare_runs(runs, output)
    (runs / "comparison-graded.yaml").write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")
    typer.echo(comparison_markdown(report))
    return report
