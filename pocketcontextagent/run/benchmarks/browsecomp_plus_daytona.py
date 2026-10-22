"""A fixed pool of Daytona workers, with one BrowseComp-Plus task per sandbox."""

import hashlib
import json
import os
import shlex
import shutil
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from uuid import uuid4

import typer
import yaml
from dotenv import load_dotenv

from pocketcontextagent.config import get_config_from_spec
from pocketcontextagent.run.benchmarks.utils.browsecomp_plus import INDEX_REVISION, load_instances
from pocketcontextagent.run.benchmarks.utils.metrics import summarize_results
from pocketcontextagent.utils.serialize import recursive_merge

app = typer.Typer(add_completion=False, no_args_is_help=True)
DEFAULT_POOL = Path("data/browsecomp-plus/daytona-pool.json")
REMOTE_INDEX = "/opt/browsecomp-plus/indexes/bm25"
TOKENIZER = "Qwen/Qwen3.5-9B"


def client() -> Any:
    load_dotenv(Path(".env"))
    if not os.getenv("DAYTONA_API_KEY"):
        raise ValueError("Set DAYTONA_API_KEY in the ignored .env file or your shell")
    from daytona import Daytona

    return Daytona()


def runtime_image(project: Path) -> Any:
    from daytona import Image

    download = (
        "from huggingface_hub import snapshot_download; "
        "snapshot_download('Tevatron/browsecomp-plus-indexes', repo_type='dataset', "
        f"revision={INDEX_REVISION!r}, allow_patterns=['bm25/*'], local_dir='/opt/browsecomp-plus/indexes'); "
        f"from transformers import AutoTokenizer; AutoTokenizer.from_pretrained({TOKENIZER!r})"
    )
    return (
        Image.base("python:3.11-slim-trixie")
        .run_commands(
            "apt-get update && apt-get install -y --no-install-recommends openjdk-21-jre-headless "
            "&& rm -rf /var/lib/apt/lists/*"
        )
        .pip_install("torch", index_url="https://download.pytorch.org/whl/cpu", extra_options="--no-cache-dir")
        .pip_install_from_pyproject(
            str(project / "pyproject.toml"), ["browsecomp-plus"], extra_options="--no-cache-dir"
        )
        .env({"HF_HOME": "/opt/huggingface"})
        .run_commands(shlex.join(["python", "-c", download]))
        .run_commands("useradd --create-home --shell /bin/bash daytona && mkdir /workspace && chown daytona /workspace")
        .env(
            {
                "PATH": "/home/daytona/.local/bin:/usr/local/bin:/usr/bin:/bin",
                "HF_HOME": "/opt/huggingface",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "MSWEA_GLOBAL_CONFIG_DIR": "/workspace/.config",
                "MSWEA_SILENT_STARTUP": "1",
                "MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT": "2",
                "LITELLM_LOCAL_MODEL_COST_MAP": "True",
                "JAVA_HOME": "/usr/lib/jvm/java-21-openjdk-amd64",
            }
        )
        .workdir("/workspace")
        .dockerfile_commands(["USER daytona"])
    )


@app.command("snapshot")
def create_snapshot(
    name: str = typer.Option("pocket-browsecomp-plus"),
    project: Path = typer.Option(Path(".")),
) -> None:
    """Build a shared CPU runtime with Java, dependencies, tokenizer, and the official BM25 index."""
    from daytona import CreateSnapshotParams, Resources

    snapshot = client().snapshot.create(
        CreateSnapshotParams(name=name, image=runtime_image(project), resources=Resources(cpu=1, memory=4, disk=10)),
        on_logs=lambda chunk: typer.echo(chunk, nl=False),
        timeout=1800,
    )
    typer.echo(f"Snapshot ready: {snapshot.name}")


def build_wheel(project: Path, destination: Path) -> Path:
    from build import ProjectBuilder

    source = destination / "source"
    source.mkdir()
    shutil.copytree(
        project / "pocketcontextagent",
        source / "pocketcontextagent",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    for name in ("pyproject.toml", "README.md", "LICENSE.md"):
        shutil.copyfile(project / name, source / name)
    return destination / ProjectBuilder(str(source)).build("wheel", str(destination))


def install_worker(sandbox: Any, wheel: Path) -> None:
    remote = f"/tmp/{wheel.name}"
    sandbox.fs.upload_file(wheel.read_bytes(), remote)
    command = shlex.join(["python", "-m", "pip", "install", "--no-deps", "--user", remote])
    result = sandbox.process.exec(command, timeout=120)
    if result.exit_code:
        raise RuntimeError(f"Worker package installation failed in sandbox {sandbox.id}: {result.result}")
    sandbox.fs.delete_file(remote)


def worker_command(arguments: list[str]) -> str:
    launcher = (
        "import os,sys; from pathlib import Path; "
        "os.environ.setdefault('JAVA_HOME', str(Path('/usr/bin/java').resolve().parents[1])); "
        "os.execvp(sys.argv[1], sys.argv[1:])"
    )
    return shlex.join(["python", "-c", launcher, *arguments])


@app.command("check")
def check_pool(pool: Path = typer.Option(DEFAULT_POOL)) -> None:
    """Verify Java, tokenizer, package, and corpus retrieval without making model calls."""
    api = client()
    workers = json.loads(pool.read_text())["sandboxes"]
    code = (
        "from pathlib import Path; "
        "from pocketcontextagent.run.benchmarks.utils.browsecomp_plus import BM25Retriever; "
        "from pocketcontextagent.environments.browsecomp_plus import BrowseCompPlusEnvironment; "
        "from pocketcontextagent.config import get_config_from_spec; "
        f"r=BM25Retriever(Path({REMOTE_INDEX!r})); "
        "env=BrowseCompPlusEnvironment(r, **get_config_from_spec('browsecomp_plus_openrouter')['environment']); "
        "assert r.search('jazz festival',1); r.close()"
    )

    def check(worker: dict) -> None:
        response = api.get(worker["id"]).process.exec(worker_command(["python", "-c", code]), timeout=120)
        if response.exit_code:
            raise RuntimeError(f"Preflight failed for {worker['id']}: {response.result}")
        typer.echo(f"Worker ready: {worker['id']}")

    with ThreadPoolExecutor(max_workers=len(workers)) as executor:
        list(executor.map(check, workers))


@app.command("create")
def create_pool(
    snapshot: str = typer.Option("pocket-browsecomp-plus"),
    workers: int = typer.Option(10, min=1, max=10),
    pool: Path = typer.Option(DEFAULT_POOL),
    project: Path = typer.Option(Path(".")),
) -> None:
    """Start up to ten isolated workers; rollback all new sandboxes if setup fails."""
    from daytona import CreateSandboxFromSnapshotParams

    api = client()
    if not os.getenv("OPENROUTER_API_KEY"):
        raise ValueError("Set OPENROUTER_API_KEY in the ignored .env file or your shell")
    if pool.exists():
        raise ValueError(f"Pool manifest already exists: {pool}. Reuse it or delete its workers first.")
    identifier = uuid4().hex[:12]
    lock = threading.Lock()
    with tempfile.TemporaryDirectory(prefix="pocket-daytona-build-") as temporary:
        wheel = build_wheel(project, Path(temporary))
        with ExitStack() as rollback, ThreadPoolExecutor(max_workers=workers) as executor:

            def create_worker(index: int) -> dict:
                sandbox = api.create(
                    CreateSandboxFromSnapshotParams(
                        snapshot=snapshot,
                        name=f"pocket-bcp-{identifier}-{index}",
                        os_user="daytona",
                        env_vars={"OPENROUTER_API_KEY": os.environ["OPENROUTER_API_KEY"]},
                        labels={"project": "pocket-context", "benchmark": "browsecomp-plus", "pool": identifier},
                        public=False,
                        auto_stop_interval=30,
                        auto_delete_interval=0,
                    ),
                    timeout=180,
                )
                with lock:
                    rollback.callback(api.delete, sandbox)
                install_worker(sandbox, wheel)
                typer.echo(f"Worker {index + 1}/{workers} ready: {sandbox.id}")
                return {"id": sandbox.id, "name": sandbox.name}

            sandboxes = list(executor.map(create_worker, range(workers)))
            manifest = {
                "pool_id": identifier,
                "snapshot": snapshot,
                "package_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
                "sandboxes": sandboxes,
            }
            pool.parent.mkdir(parents=True, exist_ok=True)
            pool.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            rollback.pop_all()
    typer.echo(f"Pool ready: {pool}")


@app.command("status")
def pool_status(pool: Path = typer.Option(DEFAULT_POOL)) -> None:
    """Read worker states without changing them."""
    api = client()
    for worker in json.loads(pool.read_text())["sandboxes"]:
        sandbox = api.get(worker["id"])
        typer.echo(f"{sandbox.id}: {sandbox.state}")


def run_remote_task(sandbox: Any, instance: dict, config: dict, output: Path, *, keep_artifacts: bool) -> dict:
    from daytona.common.process import SessionExecuteRequest

    iid = str(instance["query_id"])
    session = f"bcp-{uuid4().hex}"
    remote = f"/workspace/{session}"
    with ExitStack() as cleanup:
        sandbox.fs.create_folder(remote, "700")
        cleanup.callback(sandbox.fs.delete_file, remote, recursive=True)
        sandbox.fs.upload_file(
            (json.dumps({"query_id": iid, "query": instance["query"]}) + "\n").encode(), f"{remote}/query.jsonl"
        )
        sandbox.fs.upload_file(yaml.safe_dump(config).encode(), f"{remote}/config.yaml")
        sandbox.process.create_session(session)
        cleanup.callback(sandbox.process.delete_session, session)
        command = worker_command(
            [
                "mini-extra",
                "browsecomp-plus",
                "run",
                "--queries",
                f"{remote}/query.jsonl",
                "--index",
                REMOTE_INDEX,
                "--output",
                f"{remote}/output",
                "--split",
                "all",
                "-c",
                f"{remote}/config.yaml",
                "--workers",
                "1",
            ]
        )
        execution = sandbox.process.execute_session_command(
            session, SessionExecuteRequest(command=command, run_async=True)
        )
        deadline = time.monotonic() + (config["agent"].get("wall_time_limit_seconds") or 1800) + 180
        while sandbox.process.get_session_command(session, execution.cmd_id).exit_code is None:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Remote task {iid} exceeded its wall-time allowance in sandbox {sandbox.id}")
            time.sleep(5)
        if not any(file.name == "output" for file in sandbox.fs.list_files(remote)) or not any(
            file.name == f"{iid}.json" for file in sandbox.fs.list_files(f"{remote}/output")
        ):
            detail = sandbox.process.get_session_command_logs(session, execution.cmd_id).output[-4000:]
            for name in ("OPENROUTER_API_KEY", "DAYTONA_API_KEY"):
                if os.getenv(name):
                    detail = detail.replace(os.environ[name], "[REDACTED]")
            raise RuntimeError(f"Remote task {iid} exited before producing a result: {detail}")
        result = json.loads(sandbox.fs.download_file(f"{remote}/output/{iid}.json"))
        result["metadata"]["sandbox_id"] = sandbox.id
        if keep_artifacts:
            archives = output / "archives"
            archives.mkdir(exist_ok=True)
            archive = f"{remote}/trajectory.tar.gz"
            response = sandbox.process.exec(
                shlex.join(["tar", "-czf", archive, "-C", f"{remote}/output", "trajectories"]), timeout=120
            )
            if response.exit_code:
                raise RuntimeError(f"Failed to collect the trajectory for task {iid}")
            sandbox.fs.download_file(archive, str(archives / f"{iid}.tar.gz"))
        (output / f"{iid}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        typer.echo(f"{iid}: {result['status']} on {sandbox.id}; peak={result['metrics']['peak_input_tokens']} tokens")
        return result


def run_pool_tasks(
    api: Any,
    workers: list[dict],
    instances: list[dict],
    config: dict,
    output: Path,
    *,
    keep_artifacts: bool,
    success_limit: int = 0,
) -> list[dict]:
    results = []
    cursor = 0
    with ThreadPoolExecutor(max_workers=len(workers)) as executor:
        while cursor < len(instances):
            remaining = success_limit - sum(result["status"] == "completed" for result in results)
            if success_limit and remaining <= 0:
                break
            batch = instances[cursor : cursor + min(len(workers), remaining if success_limit else len(workers))]

            def run_worker(item: tuple[dict, dict]) -> dict:
                worker, instance = item
                return run_remote_task(api.get(worker["id"]), instance, config, output, keep_artifacts=keep_artifacts)

            results.extend(executor.map(run_worker, zip(workers, batch)))
            cursor += len(batch)
            typer.echo(
                f"Progress: {sum(result['status'] == 'completed' for result in results)} completed / {len(results)} attempted"
            )
    return results


@app.command("run")
def run_pool(
    pool: Path = typer.Option(DEFAULT_POOL),
    data_dir: Path = typer.Option(Path("data/browsecomp-plus")),
    split: str = typer.Option("development680", help="Use eval150 explicitly for the held-out benchmark."),
    limit: int = typer.Option(10, min=0, help="Number of tasks; 0 selects the whole split."),
    success_limit: int = typer.Option(
        0, min=0, help="Stop after exactly this many completed tasks; 0 runs all selected tasks."
    ),
    offset: int = typer.Option(0, min=0),
    workers: int = typer.Option(10, min=1, max=10),
    config_spec: list[str] = typer.Option([], "-c", "--config"),
    output: Path = typer.Option(Path("runs/browsecomp-plus/daytona")),
    cleanup: bool = typer.Option(
        True, "--cleanup/--keep-artifacts", help="Delete local smoke outputs after reporting metrics."
    ),
) -> None:
    """Run at most ten tasks concurrently, one per sandbox; always remove remote task workspaces."""
    api = client()
    manifest = json.loads(pool.read_text())
    if len(manifest["sandboxes"]) < workers:
        raise ValueError(f"Requested {workers} workers, but the pool contains only {len(manifest['sandboxes'])}")
    instances = load_instances(data_dir / "queries.jsonl", split=split, offset=offset, limit=limit)
    if success_limit > len(instances):
        raise ValueError("success-limit cannot exceed the number of selected tasks")
    config = recursive_merge(
        get_config_from_spec("benchmarks/browsecomp_plus.yaml"),
        get_config_from_spec("benchmarks/browsecomp_plus_openrouter.yaml"),
        *[get_config_from_spec(spec) for spec in config_spec],
    )
    settings = {
        "pool": manifest,
        "split": split,
        "config": config,
        "queries_sha256": hashlib.sha256((data_dir / "queries.jsonl").read_bytes()).hexdigest(),
    }
    with ExitStack() as resources:
        if cleanup:
            output = Path(resources.enter_context(tempfile.TemporaryDirectory(prefix="pocket-daytona-smoke-")))
        output.mkdir(parents=True, exist_ok=True)
        settings_path = output / "run.yaml"
        if not settings_path.exists() and list(output.glob("*.json")):
            raise ValueError("Output contains results without a manifest. Use a new --output directory.")
        if settings_path.exists() and yaml.safe_load(settings_path.read_text()) != settings:
            raise ValueError("Run settings changed. Use a new --output directory.")
        settings_path.write_text(yaml.safe_dump(settings), encoding="utf-8")
        pending = [
            instance
            for instance in instances
            if not (path := output / f"{instance['query_id']}.json").exists()
            or json.loads(path.read_text())["status"] != "completed"
        ]
        existing = [
            json.loads((output / f"{instance['query_id']}.json").read_text())
            for instance in instances
            if instance not in pending
        ]
        remaining_successes = max(0, success_limit - len(existing)) if success_limit else 0
        concurrent = min(workers, len(pending), remaining_successes if success_limit else len(pending))
        typer.echo(f"Selected {len(pending)} pending tasks; up to {concurrent} parallel sandboxes ({split})")
        if success_limit:
            typer.echo(f"Stopping after {remaining_successes} additional submissions")
        if not success_limit or remaining_successes:
            run_pool_tasks(
                api,
                manifest["sandboxes"][:workers],
                pending,
                config,
                output,
                keep_artifacts=not cleanup,
                success_limit=remaining_successes,
            )
        results = [
            json.loads((output / f"{instance['query_id']}.json").read_text())
            for instance in instances
            if (output / f"{instance['query_id']}.json").exists()
        ]
        summary = summarize_results(results)
        if success_limit:
            summary["success_target"] = success_limit
            summary["successful_tasks"] = summarize_results(
                [result for result in results if result["status"] == "completed"]
            )
        if not cleanup:
            (output / "summary.yaml").write_text(yaml.safe_dump(summary), encoding="utf-8")
        typer.echo(yaml.safe_dump(summary, sort_keys=False))
    if cleanup:
        typer.echo("Local smoke artifacts deleted; remote task workspaces removed")
    if summary["completed"] < (success_limit or summary["tasks"]):
        raise typer.Exit(1)


@app.command("delete")
def delete_pool(pool: Path = typer.Option(DEFAULT_POOL)) -> None:
    """Delete only the sandboxes recorded in this pool's local manifest."""
    api = client()
    workers = json.loads(pool.read_text())["sandboxes"]
    with ThreadPoolExecutor(max_workers=len(workers)) as executor:
        list(executor.map(lambda worker: api.delete(api.get(worker["id"])), workers))
    pool.unlink()
    typer.echo("Pool deleted")
