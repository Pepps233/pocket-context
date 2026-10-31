import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from benchmarks.benchmark_allocation import allocation
from benchmarks.browsecomp_plus import app
from pocketcontextagent.config import builtin_config_dir


def test_allocation_keeps_full_reserved_corpus_and_samples_125_coding_tasks(tmp_path: Path):
    corpus = tmp_path / "corpus"
    coding = tmp_path / "coding"
    corpus.mkdir()
    coding.mkdir()
    reserved = json.loads((builtin_config_dir / "benchmarks/browsecomp_plus_eval150.json").read_text())["query_ids"]
    (corpus / "queries.jsonl").write_text(
        "".join(
            json.dumps({"query_id": str(iid), "query": "Research this.", "answer": "PRIVATE_GOLD"}) + "\n"
            for iid in [*reserved, "development-only"]
        )
    )
    (coding / "queries.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "instance_id": f"fixture__repo-{i}",
                    "base_commit": "0" * 40,
                    "problem_statement": "Fix this issue.",
                    "repo": "fixture/repo",
                    "version": "1",
                    "patch": "PRIVATE_GOLD",
                }
            )
            + "\n"
            for i in range(500)
        )
    )
    output = tmp_path / "allocation.yaml"
    plan = allocation(corpus, coding, output)
    assert plan["unique_tasks"] == 275 and plan["attempts"] == 1100
    assert len(plan["methods"]) == 4 and plan["workers_per_method"] == 30
    assert set(plan["benchmarks"]["browsecomp-plus"]["selected_query_ids"]) == set(map(str, reserved))
    assert plan["benchmarks"]["browsecomp-plus"]["attempts"] == 600
    selected = plan["benchmarks"]["swebench-verified"]["selected_query_ids"]
    assert len(selected) == len(set(selected)) == 125
    assert set(selected) <= {f"fixture__repo-{i}" for i in range(500)}
    assert plan["benchmarks"]["swebench-verified"]["attempts"] == 500
    assert yaml.safe_load(output.read_text()) == plan
    assert "PRIVATE_GOLD" not in output.read_text()
    assert allocation(corpus, coding) == plan


def test_removed_benchmark_and_coding_pool_misuse_are_rejected_before_api_calls(tmp_path: Path):
    result = CliRunner().invoke(app, ["daytona", "matrix", "--benchmark", "deepsearchqa", "--dry-run"])
    assert isinstance(result.exception, ValueError) and "Select browsecomp-plus or swebench-verified" in str(
        result.exception
    )
    result = CliRunner().invoke(
        app,
        [
            "daytona",
            "matrix",
            "--benchmark",
            "swebench-verified",
            "--pool",
            str(tmp_path / "not-a-coding-pool.json"),
            "--dry-run",
        ],
    )
    assert isinstance(result.exception, ValueError) and "omit --pool" in str(result.exception)
