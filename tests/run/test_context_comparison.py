import hashlib
import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from pocketcontextagent.run.benchmarks.browsecomp_plus import app
from pocketcontextagent.run.benchmarks.utils.comparison import compare_runs, comparison_markdown
from pocketcontextagent.run.benchmarks.utils.methods import METHODS, configure_method
from pocketcontextagent.run.benchmarks.utils.metrics import episode_metrics


def write_runs(root: Path) -> None:
    config = {
        "agent": {"context_token_limit": 128000, "step_limit": 100, "cost_limit": 3.0},
        "model": {"model_class": "litellm", "model_name": "example/model"},
        "environment": {"k": 5},
    }
    for method in METHODS:
        directory = root / method
        directory.mkdir(parents=True)
        manifest = {
            "method": method,
            "config": configure_method(config, method),
            "split": "all",
            "workers": 1,
            "retain_recorded_attempts": True,
            "selected_query_ids": ["a", "b"],
            "dataset_revision": "dataset",
            "index_revision": "index",
            "queries_sha256": "questions",
            "index_files": [["segments_1", 10, 1]],
            "implementation_sha256": "source",
        }
        (directory / "run.yaml").write_text(yaml.safe_dump(manifest))
        fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        for iid in ("a", "b"):
            result = {
                "query_id": iid,
                "status": "completed" if iid == "a" else "LimitsExceeded",
                "metadata": {"fingerprint": fingerprint},
                "model_stats": {"instance_cost": 0.01, "api_calls": 1},
                "result": [{"type": "output_text", "output": "Exact Answer: example" if iid == "a" else ""}],
                "metrics": episode_metrics(
                    [{"object": "response", "id": iid, "usage": {"input_tokens": 100, "output_tokens": 20}}],
                    latency_seconds=2,
                ),
            }
            (directory / f"{iid}.json").write_text(json.dumps(result))


def write_grades(root: Path) -> None:
    for method in METHODS:
        directory = root / method
        directory.mkdir(parents=True)
        (directory / "a_eval.json").write_text(
            json.dumps(
                {
                    "query_id": "a",
                    "response": "Exact Answer: example",
                    "is_completed": True,
                    "judge_result": {"correct": True, "parse_error": False},
                    "model_info": {"judge_model": "example/judge", "max_output_tokens": 4096},
                }
            )
        )


def test_matched_report_requires_grading_and_counts_failed_tasks(tmp_path: Path):
    runs, grades = tmp_path / "runs", tmp_path / "grades"
    write_runs(runs)
    assert all(row["success_rate"] is None for row in compare_runs(runs)["methods"].values())
    write_grades(grades)
    report = compare_runs(runs, grades)
    for row in report["methods"].values():
        assert row["tasks"] == 2 and row["completed"] == 1
        assert row["success_rate"] == 0.5
        assert row["avg_total_tokens_per_task"] == 120
        assert row["avg_cost_usd_per_task"] == 0.01
        assert row["avg_latency_seconds_per_task"] == 2
        assert row["avg_peak_input_tokens"] == 100
    assert "50.00%" in comparison_markdown(report)
    output = tmp_path / "comparison.yaml"
    result = CliRunner().invoke(app, ["compare", "--runs", str(runs), "--grades", str(grades), "--output", str(output)])
    assert result.exit_code == 0, result.output
    assert yaml.safe_load(output.read_text()) == report
    assert not (runs / "pocket_context" / "comparison.yaml").exists()


@pytest.mark.parametrize(
    ("mutation",),  # noqa: PT006
    [(value,) for value in ("budget", "workers", "task", "fingerprint", "method", "extra-result", "missing-result")],
)
def test_report_rejects_unmatched_or_partial_runs(tmp_path: Path, mutation: str):
    write_runs(tmp_path)
    directory = tmp_path / "acm"
    manifest = yaml.safe_load((directory / "run.yaml").read_text())
    if mutation == "budget":
        manifest["config"]["agent"]["cost_limit"] = 10
    elif mutation == "workers":
        manifest["workers"] = 2
    elif mutation == "task":
        manifest["selected_query_ids"] = ["different"]
    elif mutation == "method":
        manifest["method"] = "react"
    elif mutation == "fingerprint":
        result = json.loads((directory / "a.json").read_text())
        result["metadata"]["fingerprint"] = "stale"
        (directory / "a.json").write_text(json.dumps(result))
    elif mutation == "extra-result":
        (directory / "extra.json").write_text(json.dumps({"query_id": "extra"}))
    else:
        (directory / "b.json").unlink()
    (directory / "run.yaml").write_text(yaml.safe_dump(manifest))
    with pytest.raises(ValueError):
        compare_runs(tmp_path)


@pytest.mark.parametrize(
    ("mutation",),  # noqa: PT006
    [(value,) for value in ("response", "judge", "duplicate", "invalid-grade")],
)
def test_report_validates_upstream_judgments(tmp_path: Path, mutation: str):
    runs, grades = tmp_path / "runs", tmp_path / "grades"
    write_runs(runs)
    write_grades(grades)
    path = grades / "acm" / "a_eval.json"
    judgment = json.loads(path.read_text())
    if mutation == "response":
        judgment["response"] = "stale answer"
    elif mutation == "judge":
        judgment["model_info"]["judge_model"] = "different"
    elif mutation == "duplicate":
        (path.parent / "duplicate_eval.json").write_text(json.dumps(judgment))
    else:
        judgment["judge_result"] = {"correct": "yes", "parse_error": False}
    path.write_text(json.dumps(judgment))
    if mutation == "invalid-grade":
        assert compare_runs(runs, grades)["methods"]["acm"]["success_rate"] is None
    else:
        with pytest.raises(ValueError):
            compare_runs(runs, grades)


def test_method_selection_is_explicit_and_does_not_mutate_shared_config():
    config = {"agent": {"cost_limit": 1}, "model": {"model_name": "example/model"}, "environment": {"k": 5}}
    assert configure_method(config, "react")["model"]["context_tools"] == "none"
    assert configure_method(config, "acm")["agent"]["agent_class"] == "acm"
    assert config == {"agent": {"cost_limit": 1}, "model": {"model_name": "example/model"}, "environment": {"k": 5}}
    with pytest.raises(ValueError, match="Unknown method"):
        configure_method(config, "unknown")
