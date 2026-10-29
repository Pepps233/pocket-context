"""Freeze the requested 150 corpus / 125 coding task allocation without API calls."""

import hashlib
from pathlib import Path

import typer
import yaml

from pocketcontextagent.comparisons.registry import METHODS
from pocketcontextagent.config import get_config_from_spec
from pocketcontextagent.run.benchmarks.benchmark_matrix import load_instances
from pocketcontextagent.run.benchmarks.browsecomp_plus_daytona import select_matrix_instances


def allocation(browsecomp_data: Path, swebench_data: Path, output: Path | None = None) -> dict:
    config = get_config_from_spec("context_evaluation")
    corpus = config["benchmarks"]["browsecomp-plus"]
    coding = config["benchmarks"]["swebench-verified"]
    selected = {
        "browsecomp-plus": select_matrix_instances(
            browsecomp_data / "queries.jsonl", corpus["split"], [], corpus["limit"], config["seed"]
        ),
        "swebench-verified": load_instances(
            swebench_data, query_ids=[], limit=coding["limit"], seed=config["seed"], split=coding["split"]
        ),
    }
    report = {
        "seed": config["seed"],
        "workers_per_method": config["workers_per_method"],
        "methods": list(METHODS),
        "benchmarks": {
            name: config["benchmarks"][name]
            | {"selected_query_ids": [row["query_id"] for row in rows], "attempts": len(rows) * len(METHODS)}
            for name, rows in selected.items()
        },
        "source_queries_sha256": {
            name: hashlib.sha256((path / "queries.jsonl").read_bytes()).hexdigest()
            for name, path in [("browsecomp-plus", browsecomp_data), ("swebench-verified", swebench_data)]
        },
        "unique_tasks": sum(map(len, selected.values())),
        "attempts": sum(len(rows) * len(METHODS) for rows in selected.values()),
    }
    rendered = yaml.safe_dump(report, sort_keys=False)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    typer.echo(rendered)
    return report
