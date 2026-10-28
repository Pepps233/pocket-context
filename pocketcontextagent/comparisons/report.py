"""Matched-run comparisons with optional upstream BrowseComp-Plus judgments."""

import copy
import hashlib
import json
from pathlib import Path

import yaml

from pocketcontextagent.comparisons.registry import METHOD_AGENT_FIELDS, METHODS, configure_method, method_provenance
from pocketcontextagent.run.benchmarks.utils.metrics import summarize_results


def shared_settings(manifest: dict) -> dict:
    config = copy.deepcopy(manifest["config"])
    config["agent"].pop("agent_class", None)
    for key in METHOD_AGENT_FIELDS:
        config["agent"].pop(key, None)
    config["model"].pop("context_tools", None)
    config["environment"].pop("context_tools", None)
    return {
        "config": config,
        **{
            key: manifest[key]
            for key in (
                "split",
                "workers",
                "retain_recorded_attempts",
                "selected_query_ids",
                "dataset_revision",
                "index_revision",
                "queries_sha256",
                "index_files",
                "implementation_sha256",
            )
        },
    }


def compare_runs(runs: Path, grades: Path | None = None) -> dict:
    if grades is not None and not grades.is_dir():
        raise ValueError(f"Judge directory does not exist: {grades}")
    rows = {}
    reference = None
    judge_settings = None
    for method in METHODS:
        directory = runs / method
        manifest = yaml.safe_load((directory / "run.yaml").read_text())
        if manifest.get("method") != method:
            raise ValueError(f"{directory}: expected a run created with --method {method}")
        expected = configure_method(manifest["config"], method)
        if expected != manifest["config"]:
            raise ValueError(
                f"{method}: method-specific prompts or native tool configuration differ from the versioned port"
            )
        if manifest.get("method_provenance") != method_provenance(manifest["config"], method):
            raise ValueError(f"{method}: missing or changed method provenance")
        settings = shared_settings(manifest)
        if reference is not None and reference != settings:
            raise ValueError(
                f"{method}: model, tools, budgets, workers, data, implementation, or selected tasks differ"
            )
        reference = settings
        fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        results = [json.loads(path.read_text()) for path in sorted(directory.glob("*.json"))]
        ids = [str(result["query_id"]) for result in results]
        if len(ids) != len(set(ids)) or set(ids) != set(manifest["selected_query_ids"]):
            raise ValueError(f"{method}: results must cover exactly the selected tasks, including failures")
        judgments = {}
        if grades is not None:
            if not (grades / method).is_dir():
                raise ValueError(f"Expected a judge subdirectory for {method}: {grades / method}")
            for path in sorted((grades / method).rglob("*_eval.json")):
                judgment = json.loads(path.read_text())
                iid = str(judgment["query_id"])
                if iid in judgments or iid not in ids:
                    raise ValueError(f"{method}: duplicate or unexpected judgment for {iid}")
                judgments[iid] = judgment
        for result in results:
            if result.get("metadata", {}).get("fingerprint") != fingerprint:
                raise ValueError(f"{method}: result {result['query_id']} does not match its run manifest")
            result["correct"] = False if result["status"] != "completed" else None
            if judgment := judgments.get(str(result["query_id"])):
                response = result["result"][-1]["output"] if result["result"] else ""
                if judgment.get("response") != response or judgment.get("is_completed") != (
                    result["status"] == "completed"
                ):
                    raise ValueError(f"{method}: stale judgment for {result['query_id']}")
                correct = judgment.get("judge_result", {}).get("correct")
                if type(correct) is bool and not judgment["judge_result"].get("parse_error"):
                    identity = judgment.get("model_info")
                    if not identity:
                        raise ValueError(f"{method}: judgment must identify its judge model and settings")
                    if judge_settings is not None and identity != judge_settings:
                        raise ValueError("Judge model or settings differ between judgments")
                    judge_settings = identity
                    result["correct"] = correct if result["status"] == "completed" else False
        rows[method] = summarize_results(results)
    return {
        "settings": reference,
        "judge": judge_settings,
        "methods": rows,
        "method_provenance": {
            name: yaml.safe_load((runs / name / "run.yaml").read_text())["method_provenance"] for name in METHODS
        },
    }


def comparison_markdown(report: dict) -> str:
    def number(value, *, percent=False):
        if value is None:
            return "N/A"
        return f"{value * 100:.2f}%" if percent else f"{value:,.2f}"

    lines = [
        "| Method | Success rate | Total tokens/task | Cost USD/task | Latency s/task | Peak input tokens/task |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method, row in report["methods"].items():
        values = [
            number(row["success_rate"], percent=True),
            number(row["avg_total_tokens_per_task"]),
            number(row["avg_cost_usd_per_task"]),
            number(row["avg_latency_seconds_per_task"]),
            number(row["avg_peak_input_tokens"]),
        ]
        lines.append(f"| {method} | " + " | ".join(values) + " |")
    tasks = next(iter(report["methods"].values()))["tasks"]
    lines.append(f"\nTasks per method: {tasks}; split: {report['settings']['split']}.")
    lines.append("N/A means incomplete measurement or grading; completed submissions are not correctness judgments.")
    return "\n".join(lines)
