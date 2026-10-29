"""SWE-bench connector for the shared four-method Daytona scheduler.

Agents run on the orchestrator; only shell tools run in official instance images.
This keeps model and control-plane credentials outside agent-controlled checkouts.
"""

import hashlib
import json
import random
import sys
import tarfile
import threading
import time
from pathlib import Path
from uuid import uuid4

import typer
import yaml

from pocketcontextagent import __version__
from pocketcontextagent.agents import get_agent
from pocketcontextagent.comparisons.registry import METHODS, configure_method, method_provenance
from pocketcontextagent.comparisons.report import compare_runs, comparison_markdown
from pocketcontextagent.config import get_config_from_spec
from pocketcontextagent.environments.swebench_verified import SWEBenchVerifiedEnvironment
from pocketcontextagent.models import get_model
from pocketcontextagent.run.benchmarks.browsecomp_plus import implementation_hash
from pocketcontextagent.run.benchmarks.browsecomp_plus_daytona import (
    client,
    matrix_provider_key,
    provider_environment,
    publish_matrix_task,
    run_matrix_workers,
)
from pocketcontextagent.run.benchmarks.utils import swebench_verified as swe
from pocketcontextagent.run.benchmarks.utils.metrics import episode_metrics
from pocketcontextagent.utils.serialize import recursive_merge


def prepare(data_dir: Path) -> None:
    swe.prepare_dataset(data_dir / "queries.jsonl", grading_path=data_dir / "grading_dataset.jsonl")
    (data_dir / "manifest.json").write_text(
        json.dumps(
            {
                "dataset": swe.DATASET_NAME,
                "dataset_revision": swe.DATASET_REVISION,
                "split": "test",
                "harness_revision": swe.HARNESS_REVISION,
                "queries_sha256": hashlib.sha256((data_dir / "queries.jsonl").read_bytes()).hexdigest(),
                "grading_dataset_sha256": hashlib.sha256((data_dir / "grading_dataset.jsonl").read_bytes()).hexdigest(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def load_instances(
    data_dir: Path, *, query_ids: list[str], limit: int, seed: int | None, split: str = "test"
) -> list[dict]:
    if split not in ("test", "all"):
        raise ValueError("SWE-bench Verified uses the test split")
    rows = [
        swe.public_task(json.loads(line))
        for line in (data_dir / "queries.jsonl").read_text().splitlines()
        if line.strip()
    ]
    ids = [row["query_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate task IDs")
    if set(query_ids) - set(ids):
        raise ValueError("Requested task IDs are absent from the prepared dataset")
    rows = [row for row in rows if not query_ids or row["query_id"] in query_ids]
    if not rows or limit < 0 or limit > len(rows):
        raise ValueError(f"Select 1 to {len(rows)} tasks; --limit 0 selects all")
    return random.Random(seed).sample(rows, limit or len(rows)) if seed is not None else rows[: limit or None]


def benchmark_config(config_spec: list[str]) -> dict:
    config = recursive_merge(
        get_config_from_spec("swebench_verified_context"),
        *[get_config_from_spec(spec) for spec in config_spec or ["context_benchmark_deepseek"]],
    )
    if config.get("benchmark") != "swebench-verified":
        raise ValueError("The coding connector requires benchmark=swebench-verified")
    allowed = {"cwd", "timeout", "research_tools", "context_tools"}
    if set(config.get("environment", {})) - allowed:
        raise ValueError("Use a coding provider profile, not a BrowseComp-Plus retrieval profile")
    matrix_provider_key(config)
    if "api_key" in config["model"].get("model_kwargs", {}):
        raise ValueError("Keep provider keys in the environment, not saved run configurations")
    return config


def run_manifest(config: dict, method: str, instances: list[dict], workers: int, source: Path, split: str) -> dict:
    metadata = (
        json.loads((source.parent / "manifest.json").read_text()) if (source.parent / "manifest.json").exists() else {}
    )
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    grading = source.parent / "grading_dataset.jsonl"
    grading_hash = hashlib.sha256(grading.read_bytes()).hexdigest() if grading.exists() else None
    if metadata and (
        metadata.get("queries_sha256") != source_hash
        or metadata.get("grading_dataset_sha256") != grading_hash
        or metadata.get("dataset_revision") != swe.DATASET_REVISION
    ):
        raise ValueError("Prepared public/private task snapshots changed; prepare the pinned dataset again")
    return {
        "benchmark": "swebench-verified",
        "config": configure_method(config, method),
        "method": method,
        "method_description": METHODS[method].description,
        "method_provenance": method_provenance(configure_method(config, method), method),
        "split": split,
        "workers": workers,
        "retain_recorded_attempts": True,
        "selected_query_ids": [row["query_id"] for row in instances],
        "version": __version__,
        "implementation_sha256": implementation_hash(),
        "dataset_revision": metadata.get("dataset_revision", "local:" + source_hash),
        "queries_sha256": source_hash,
        "grading_dataset_sha256": grading_hash,
        "index_revision": "official-swebench-instance-images",
        "index": "",
        "index_files": [],
        "task_images": {row["query_id"]: row["image"] for row in instances},
        "runtime": "local-agent/daytona-shell",
        "harness_revision": swe.HARNESS_REVISION,
    }


def run_instance(instance: dict, output: Path, config: dict, env: SWEBenchVerifiedEnvironment, manifest: dict) -> dict:
    """Capture the actual patch on every exit; submission is not official test success."""
    iid = instance["query_id"]
    trajectory = output / "trajectories" / iid
    output.mkdir(parents=True, exist_ok=True)
    (output / "run.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    agent = get_agent(
        get_model(config=config["model"]),
        env,
        config["agent"]
        | {
            "output_path": trajectory / "trajectory.json",
            "context_db_path": trajectory / "context.sqlite",
        },
    )
    started = time.monotonic()
    ended = False
    try:
        agent.run(instance["query"])
        ended = True
    finally:
        elapsed = time.monotonic() - started
        info = agent.save(trajectory / "trajectory.json")["info"]
        error = type(sys.exc_info()[1]).__name__ if not ended and sys.exc_info()[1] is not None else None
        patch = env.capture_patch()
        metrics = episode_metrics(
            agent.messages
            + [call["response"] for call in agent.compactions]
            + [call["response"] for call in getattr(agent, "memory_queries", [])],
            model_stats=info["model_stats"],
            latency_seconds=elapsed,
        )
        if error and metrics["model_calls"] < info["model_stats"]["api_calls"]:
            info["model_stats"]["cost_available"] = False
            metrics["cost_usd"] = None
        if error:
            agent.save(
                trajectory / "trajectory.json", {"info": {"execution_error": error, "model_stats": info["model_stats"]}}
            )
        result = {
            "query_id": iid,
            "instance_id": iid,
            "status": "completed"
            if info["exit_status"] == "Submitted"
            else info["exit_status"] or error or "interrupted",
            "execution_error": error,
            "agent_exit_status": info["exit_status"],
            "agent_submission": info["submission"],
            "model_name_or_path": agent.model.config.model_name,
            "model_patch": patch,
            "patch_capture_status": "captured",
            "result": [{"type": "output_text", "output": patch}],
            "tool_call_counts": metrics["tool_call_counts"],
            "model_stats": info["model_stats"],
            "metrics": metrics,
            "metadata": {
                "model": agent.model.config.model_name,
                "method": manifest["method"],
                "fingerprint": fingerprint,
            },
        }
        temporary = output / f"{iid}.json.tmp"
        temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(output / f"{iid}.json")
        with tarfile.open(output / "trajectories.tar.gz", "w:gz") as archive:
            archive.add(output / "trajectories", arcname="trajectories")
        typer.echo(f"{manifest['method']} {iid}: {result['status']}; patch={len(patch)} characters")
    return result


def bounded_command(command: str, timeout: int) -> str:
    if timeout <= 0:
        raise ValueError("Command timeout must be positive")
    return f"timeout --signal=TERM --kill-after=5s {timeout}s {command}"


def sandbox_params(instance: dict, method: str):
    from daytona import CreateSandboxFromImageParams, Resources

    return CreateSandboxFromImageParams(
        image=instance["image"],
        name=f"pca-swe-{method.replace('_', '-')}-{uuid4().hex[:12]}",
        os_user="root",
        public=False,
        network_block_all=True,
        env_vars={"BASH_ENV": "/root/.bashrc", "PAGER": "cat", "GIT_TERMINAL_PROMPT": "0"},
        labels={"project": "pocket-context", "benchmark": "swebench-verified", "method": method},
        resources=Resources(cpu=2, memory=8, disk=10),
        auto_stop_interval=30,
        auto_delete_interval=0,
    )


def run_matrix(
    *,
    data_dir: Path,
    split: str,
    query_ids: list[str],
    limit: int,
    seed: int | None,
    config_spec: list[str],
    output: Path,
    dry_run: bool,
    workers_per_method: int,
) -> None:
    instances = load_instances(data_dir, query_ids=query_ids, limit=limit, seed=seed, split=split)
    config = benchmark_config(config_spec)
    active = min(workers_per_method, len(instances))
    if not 1 <= workers_per_method <= 30:
        raise ValueError("Select 1 to 30 workers per method")
    selection = {
        "benchmark": "swebench-verified",
        "split": split,
        "seed": seed,
        "workers_per_method": workers_per_method,
        "active_workers_per_method": active,
        "selected_query_ids": [row["query_id"] for row in instances],
        "attempts": len(instances) * len(METHODS),
        "maximum_active_sandboxes": active * len(METHODS),
    }
    manifests = {
        method: run_manifest(config, method, instances, active, data_dir / "queries.jsonl", split) for method in METHODS
    }
    typer.echo(yaml.safe_dump(selection, sort_keys=False))
    if dry_run:
        return
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a fresh output directory; failed attempts are never retried automatically")
    api = client()
    provider_environment(matrix_provider_key(config))
    for manifest in manifests.values():
        model = get_model(config=manifest["config"]["model"])
        if hasattr(model.config, "action_regex") or not callable(getattr(model, "get_tools", None)):
            raise ValueError("The coding matrix requires a native-tool model")
    output.mkdir(parents=True, exist_ok=True)
    (output / "selection.yaml").write_text(yaml.safe_dump(selection, sort_keys=False), encoding="utf-8")
    locks = {method: threading.Lock() for method in METHODS}
    journal_lock = threading.Lock()
    sandboxes = []
    groups = {method: [{"id": f"{method}-{i}"} for i in range(active)] for method in METHODS}

    def execute(method: str, worker: dict, instance: dict) -> None:
        task = output / method / "tasks" / instance["query_id"]
        sandbox = api.create(sandbox_params(instance, method), timeout=600)
        prepared = False
        try:
            task.mkdir(parents=True)
            (task / "worker.yaml").write_text(
                yaml.safe_dump({"sandbox_id": sandbox.id, "image": instance["image"], "slot": worker["id"]})
            )
            with journal_lock:
                sandboxes.append({"id": sandbox.id, "method": method, "query_id": instance["query_id"]})
                temporary = output / "sandbox-cleanup.json.tmp"
                temporary.write_text(json.dumps({"sandboxes": sandboxes}, indent=2), encoding="utf-8")
                temporary.replace(output / "sandbox-cleanup.json")
            typer.echo(f"{method} {instance['query_id']}: sandbox {sandbox.id}")

            def command_backend(command: str, *, cwd: str, timeout: int) -> dict:
                result = sandbox.process.exec(bounded_command(command, timeout), cwd=cwd, timeout=timeout + 15)
                return {"output": result.result, "returncode": result.exit_code}

            env = SWEBenchVerifiedEnvironment(
                command_backend, **manifests[method]["config"]["environment"], base_commit=instance["base_commit"]
            )
            env.prepare(fresh_isolated=True)
            sandbox.set_auto_delete_interval(-1)
            prepared = True
            try:
                run_instance(instance, task, manifests[method]["config"], env, manifests[method])
            finally:
                if (task / f"{instance['query_id']}.json").exists():
                    with locks[method]:
                        publish_matrix_task(task, output / method, instance["query_id"], manifests[method])
        finally:
            if not prepared or (
                (task / f"{instance['query_id']}.json").exists() and (task / "trajectories.tar.gz").exists()
            ):
                api.delete(sandbox)
            else:
                typer.echo(
                    f"Retaining sandbox {sandbox.id} for recovery; delete it afterward using {output / 'sandbox-cleanup.json'}"
                )
                current = api.get(sandbox.id)
                if getattr(current.state, "value", current.state) in ("started", "starting"):
                    api.stop(current)

    run_matrix_workers(groups, instances, execute)
    report = export_predictions(output)
    (output / "comparison.yaml").write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")
    typer.echo(comparison_markdown(report))
    if any(row["completed"] < row["tasks"] for row in report["methods"].values()):
        raise typer.Exit(1)


def export_predictions(runs: Path) -> dict:
    report = compare_runs(runs)
    for method in METHODS:
        rows = [json.loads(path.read_text()) for path in sorted((runs / method).glob("*.json"))]
        manifest = yaml.safe_load((runs / method / "run.yaml").read_text())
        if manifest["config"].get("benchmark") != "swebench-verified":
            raise ValueError("Export requires a SWE-bench Verified matrix")
        if manifest.get("harness_revision") != swe.HARNESS_REVISION:
            raise ValueError("Use the connector matching the run's recorded harness revision")
        if {row["query_id"] for row in rows} != set(manifest["selected_query_ids"]):
            raise ValueError("Export requires all selected attempts, including failures")
        if any(row.get("patch_capture_status") != "captured" for row in rows):
            raise ValueError("Recover incomplete patch captures before exporting predictions")
        if any(
            row.get("instance_id") != row["query_id"]
            or row.get("result") != [{"type": "output_text", "output": row["model_patch"]}]
            for row in rows
        ):
            raise ValueError("Prediction IDs and patches must match the recorded attempt response")
        swe.export_predictions(rows, runs / method / "predictions.jsonl")
    return report
