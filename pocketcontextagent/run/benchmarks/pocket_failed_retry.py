"""Prepare a separate Pocket Context experiment from officially unresolved tasks.

Preparation is offline. Launch is explicit and never automatically repeated.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import typer
import yaml

from pocketcontextagent.run.benchmarks.benchmark_controller import submission_plan
from pocketcontextagent.run.benchmarks.browsecomp_plus import implementation_hash

app = typer.Typer(no_args_is_help=True)
DEFAULT_SOURCE = Path("runs/context-evaluation/swe-64k-method-controller-full-run")
DEFAULT_OUTPUT = Path("runs/context-evaluation/pocket-64k-failed-retry-v1")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def failed_tasks(index: dict, grading: dict, results: list[dict]) -> list[str]:
    cohort = index["selected_query_ids"]
    methods = [method for method in grading["methods"] if method["method"] == "pocket_context"]
    if len(cohort) != 125 or len(set(cohort)) != 125 or len(methods) != 1:
        raise ValueError("Expected one Pocket Context report for the frozen 125-task cohort")
    reports = methods[0]["summary"]["instances"]
    for rows in (reports, results):
        if len(rows) != 125 or {row["instance_id"] for row in rows} != set(cohort):
            raise ValueError("Missing or duplicate task results; reconcile grading first")
    saved = {row["instance_id"]: row for row in results}
    for row in reports:
        if type(row.get("resolved")) is not bool:
            raise ValueError("Grading outcome is not conclusive")
        if not row.get("evaluated") and saved[row["instance_id"]]["model_patch"].strip():
            raise ValueError("A nonempty patch lacks official grading; reconcile grading first")
    unresolved = {row["instance_id"] for row in reports if not row["resolved"]}
    selected = [iid for iid in cohort if iid in unresolved]
    if not 1 <= len(selected) <= 30:
        raise ValueError("Concurrent retry requires 1 to 30 unresolved tasks")
    return selected


def command(*args: str) -> list[str]:
    return [sys.executable, "-m", "pocketcontextagent.run.utilities.mini_extra", "benchmark-controller", *args]


def submit_command(root: Path, ids: list[str]) -> list[str]:
    return command(
        "submit",
        "--manifest",
        str(root / "controller.json"),
        "--benchmark",
        "swebench-verified",
        "--method",
        "pocket_context",
        "--data-dir",
        str(root / "input-data"),
        "--limit",
        "0",
        "--seed",
        "42",
        "--workers-per-method",
        str(len(ids)),
        "-c",
        str(root / "config.yaml"),
        "--context-window",
        "64000",
        *[arg for iid in ids for arg in ("--query-id", iid)],
    )


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


@app.command()
def prepare(source: Path = DEFAULT_SOURCE, output: Path = DEFAULT_OUTPUT) -> None:
    """Freeze failed IDs, source evidence, configuration and commands without API calls."""
    source, output = source.resolve(), output.resolve()
    index = json.loads((source / "run-index.json").read_text())
    grading = json.loads((source / "grading-status.json").read_text())
    results_path = source / "grading-inputs-v1/pocket_context/results.json"
    results = json.loads(results_path.read_text())
    ids = failed_tasks(index, grading, results)
    if index["context_token_limit"] != 64000 or digest(source / "shared-config.yaml") != index["config_sha256"]:
        raise ValueError("Source config differs from the frozen 64K experiment")
    config = yaml.safe_load((source / "shared-config.yaml").read_text())
    config["sandbox"].update(snapshot_build_workers=1, startup_timeout_seconds=300)
    config["model"]["model_kwargs"]["num_retries"] = 2
    output.mkdir(parents=True, exist_ok=False)
    (output / "input-data").mkdir()
    for name in ("queries.jsonl", "manifest.json", "grading_dataset.jsonl"):
        original = source / "input-data" / name
        if digest(original) != index["dataset_files"][name]:
            raise ValueError(f"Frozen input changed: {name}")
        (output / "input-data" / name).write_bytes(original.read_bytes())
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    _, selection = submission_plan(
        output / "input-data",
        [str(output / "config.yaml")],
        0,
        42,
        len(ids),
        ids,
        method="pocket_context",
        context_window=64000,
    )
    if set(selection["selected_query_ids"]) != set(ids) or selection["planned_attempts"] != len(ids):
        raise ValueError("Submission plan differs from the failed cohort")
    evidence = [source / "run-index.json", source / "grading-status.json", results_path, source / "shared-config.yaml"]
    prepared = [output / "config.yaml", *sorted((output / "input-data").iterdir())]
    write_json(
        output / "plan.json",
        {
            "experiment": "New diagnostic retry of prior failures; do not merge into original pass@1",
            "source_directory": str(source),
            "selected_query_ids": ids,
            "previous_task_jobs": {iid: index["methods"]["pocket_context"]["task_job"][iid] for iid in ids},
            "implementation_sha256": implementation_hash(),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "selection": selection,
            "sha256": {str(path): digest(path) for path in evidence + prepared},
            "controller": {"cpu": 4, "memory": 8, "disk": 10},
            "capacity": {
                "worker_slots": len(ids),
                "worker_cpu": len(ids) * config["sandbox"]["cpu"],
                "worker_memory_gib": len(ids) * config["sandbox"]["memory"],
            },
            "startup": "All tasks get concurrent slots; snapshot imports are serialized, not a simultaneous start barrier.",
            "transport": "Two LiteLLM retries plus the controller's bounded outer retry; no automatic task or launch replay.",
            "limitations": "Daytona snapshot or shell 502s can still abort the run; reconcile journals before recovery.",
            "submit_command": submit_command(output, ids),
        },
    )
    typer.echo(f"Prepared {len(ids)} failed tasks with {len(ids)} worker slots: {output / 'plan.json'}")
    typer.echo("No sandboxes, controllers, model calls or grading launched.")


@app.command()
def launch(directory: Path = DEFAULT_OUTPUT) -> None:
    """Create an isolated controller/volume and submit once; ambiguous outcomes require inspection."""
    root = directory.resolve()
    plan = json.loads((root / "plan.json").read_text())
    if plan["implementation_sha256"] != implementation_hash():
        raise ValueError("Code changed since preparation; prepare a fresh run directory")
    for name, expected in plan["sha256"].items():
        if digest(Path(name)) != expected:
            raise ValueError(f"Frozen evidence or input changed: {name}")
    if (root / "controller.json").exists():
        raise ValueError("Controller already recorded; inspect it before any continuation")
    # Exclusive intent prevents replay even if the first API response is lost.
    with (root / "launch-intent.json").open("x") as handle:
        json.dump(plan, handle, indent=2)
    volume = "pca-failed-" + hashlib.sha256(str(root).encode()).hexdigest()[:16]
    create = command(
        "create",
        "--manifest",
        str(root / "controller.json"),
        "--project",
        str(Path.cwd()),
        "--cpu",
        "4",
        "--memory",
        "8",
        "--disk",
        "10",
        "--volume-name",
        volume,
    )
    for stage, argv in (("create", create), ("submit", submit_command(root, plan["selected_query_ids"]))):
        with (root / f"{stage}.log").open("x") as log:
            result = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"{stage} failed or is ambiguous. Inspect {root / (stage + '.log')}; do not relaunch.")
    document = json.loads((root / "controller.json").read_text())
    if len(document["jobs"]) != 1:
        raise ValueError("Unexpected controller job ownership; inspect before continuing")
    job = document["jobs"][0]
    if set(job["selection"]["selected_query_ids"]) != set(plan["selected_query_ids"]):
        raise ValueError("Accepted job differs from the prepared cohort")
    write_json(root / "job.json", job)
    typer.echo(f"Accepted {len(plan['selected_query_ids'])} tasks as {job['id']}. See controller.json for status.")


if __name__ == "__main__":
    app()
