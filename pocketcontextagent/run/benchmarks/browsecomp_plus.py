"""Prepare and run BrowseComp-Plus with Pocket Context Agent."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path

import typer
import yaml
from dotenv import load_dotenv

from pocketcontextagent import __version__
from pocketcontextagent.agents import get_agent
from pocketcontextagent.config import get_config_from_spec
from pocketcontextagent.environments.browsecomp_plus import BrowseCompPlusEnvironment, extract_answer
from pocketcontextagent.models import get_model, get_model_name
from pocketcontextagent.run.benchmarks.browsecomp_plus_daytona import app as daytona_app
from pocketcontextagent.run.benchmarks.utils.browsecomp_plus import (
    DATASET_REVISION,
    INDEX_REVISION,
    BM25Retriever,
    load_instances,
    prepare_data,
    reserve_evaluation,
)
from pocketcontextagent.run.benchmarks.utils.metrics import episode_metrics, summarize_results
from pocketcontextagent.utils.serialize import recursive_merge

DEFAULT_DATA_DIR = Path("data/browsecomp-plus")
DEFAULT_CONFIG = "benchmarks/browsecomp_plus.yaml"
app = typer.Typer(add_completion=False, no_args_is_help=True)
app.add_typer(daytona_app, name="daytona")


@app.callback()
def benchmark() -> None:
    """Prepare data and evaluate Pocket Context Agent on BrowseComp-Plus."""
    load_dotenv(Path(".env"))


@app.command()
def prepare(
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR, help="Local dataset and index directory."),
    download_index: bool = typer.Option(True, "--index/--no-index", help="Download the official BM25 index (2.2 GB)."),
) -> None:
    """Download pinned queries, decrypt locally, and fetch the official BM25 index."""
    prepare_data(data_dir, download_index=download_index)
    typer.echo(f"Prepared BrowseComp-Plus in {data_dir}")


@app.command()
def reserve(data_dir: Path = typer.Option(DEFAULT_DATA_DIR)) -> None:
    """Reserve the authors' exact 150 evaluation tasks and separate 680 development tasks."""
    typer.echo(yaml.safe_dump(reserve_evaluation(data_dir)))


def run_instance(instance: dict, output: Path, config: dict, retriever: BM25Retriever, fingerprint: str) -> bool:
    query_id = str(instance["query_id"])
    trajectory_dir = output / "trajectories" / query_id
    env = BrowseCompPlusEnvironment(retriever, **config.get("environment", {}))
    agent = get_agent(
        get_model(config=config["model"]),
        env,
        config["agent"]
        | {"output_path": trajectory_dir / "trajectory.json", "context_db_path": trajectory_dir / "context.sqlite"},
        default_type="default",
    )
    try:
        agent.run(instance["query"])
    finally:
        info = agent.save(trajectory_dir / "trajectory.json")["info"]
        result = {
            "query_id": query_id,
            "status": "completed" if info["exit_status"] == "Submitted" else info["exit_status"] or "interrupted",
            "tool_call_counts": env.tool_call_counts,
            "retrieved_docids": sorted(env.retrieved_docids),
            "result": [{"type": "output_text", "output": info["submission"]}],
            **extract_answer(info["submission"]),
            "metadata": {"model": agent.model.config.model_name, "retriever": "BM25", "fingerprint": fingerprint},
            "model_stats": info["model_stats"],
            "metrics": episode_metrics(agent.messages),
        }
        temporary = output / f"{query_id}.json.tmp"
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(output / f"{query_id}.json")
        typer.echo(f"{query_id}: {result['status']} (${agent.cost:.4f})")
    return result["status"] == "completed"


@app.command()
def run(
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR, help="Directory created by prepare."),
    queries: Path | None = typer.Option(None, help="Decrypted JSONL; defaults to DATA_DIR/queries.jsonl."),
    index: Path | None = typer.Option(None, help="Lucene index; defaults to DATA_DIR/indexes/bm25."),
    output: Path = typer.Option(Path("runs/browsecomp-plus/default"), "-o", "--output"),
    model: str | None = typer.Option(None, "-m", "--model", help="Provider/model name; can also be set in config."),
    model_class: str | None = typer.Option(None, help="Existing model interface, e.g. litellm or openrouter."),
    config_spec: list[str] = typer.Option([], "-c", "--config", help="YAML or key=value overrides, merged in order."),
    query_id: list[str] = typer.Option([], help="Run only these query IDs (repeatable)."),
    offset: int = typer.Option(0, min=0, help="Skip this many selected queries."),
    limit: int = typer.Option(0, min=0, help="Maximum selected queries; 0 runs all."),
    workers: int = typer.Option(1, "-w", "--workers", min=1),
    split: str = typer.Option("development680", help="development680 (default), eval150, or all."),
    redo_existing: bool = typer.Option(False, help="Rerun completed queries too."),
) -> None:
    """Run corpus-only research and export files accepted by the official evaluator."""
    queries = queries or data_dir / "queries.jsonl"
    index = index or data_dir / "indexes" / "bm25"
    config = recursive_merge(
        get_config_from_spec(DEFAULT_CONFIG), *[get_config_from_spec(spec) for spec in config_spec]
    )
    config["model"]["model_name"] = get_model_name(model, config["model"])
    if model_class:
        config["model"]["model_class"] = model_class
    instances = load_instances(queries, split=split, query_ids=query_id, offset=offset, limit=limit)
    manifest = {
        "config": config,
        "split": split,
        "version": __version__,
        "dataset_revision": DATASET_REVISION,
        "index_revision": INDEX_REVISION,
        "queries_sha256": hashlib.sha256(queries.read_bytes()).hexdigest(),
        "index": str(index.resolve()),
        "index_files": [
            [path.name, path.stat().st_size, path.stat().st_mtime_ns]
            for path in sorted(index.iterdir())
            if path.is_file()
        ],
    }
    fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "run.yaml"
    if not manifest_path.exists() and list(output.glob("*.json")):
        raise ValueError("Output contains results without a run manifest. Use a new --output directory.")
    if manifest_path.exists() and yaml.safe_load(manifest_path.read_text()) != manifest:
        raise ValueError("Run settings or data changed. Use a new --output directory to avoid mixing results.")
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    if not redo_existing:
        instances = [
            instance
            for instance in instances
            if not (path := output / f"{instance['query_id']}.json").exists()
            or json.loads(path.read_text())["status"] != "completed"
        ]
    typer.echo(f"Running {len(instances)} queries; results: {output}")
    if not instances:
        return
    with closing(BM25Retriever(index)) as retriever, ThreadPoolExecutor(max_workers=workers) as executor:
        completed = list(
            executor.map(lambda instance: run_instance(instance, output, config, retriever, fingerprint), instances)
        )
    results = [json.loads(path.read_text()) for path in sorted(output.glob("*.json"))]
    (output / "summary.yaml").write_text(yaml.safe_dump(summarize_results(results)), encoding="utf-8")
    if not all(completed):
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
