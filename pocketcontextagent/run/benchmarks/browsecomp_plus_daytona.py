"""A fixed pool of Daytona workers, with one BrowseComp-Plus task per sandbox."""

import hashlib
import json
import os
import random
import re
import shlex
import shutil
import tempfile
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from uuid import uuid4

import typer
import yaml
from dotenv import load_dotenv

from pocketcontextagent.comparisons.registry import METHODS, configure_method, method_provenance
from pocketcontextagent.config import get_config_from_spec
from pocketcontextagent.run.benchmarks.utils.browsecomp_plus import DATASET_REVISION, INDEX_REVISION, load_instances
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

    with ThreadPoolExecutor(max_workers=min(len(workers), 10)) as executor:
        list(executor.map(check, workers))


def provider_environment(key: str) -> dict[str, str]:
    if not re.fullmatch(r"[A-Z][A-Z0-9_]*_API_KEY", key) or key == "DAYTONA_API_KEY":
        raise ValueError("Select a model provider's *_API_KEY, not the Daytona control-plane key")
    if not os.getenv(key):
        raise ValueError(f"Set {key} in the ignored .env file or your shell")
    return {key: os.environ[key]}


@app.command("create")
def create_pool(
    snapshot: str = typer.Option("pocket-browsecomp-plus"),
    workers: int = typer.Option(10, min=1, max=120),
    pool: Path = typer.Option(DEFAULT_POOL),
    project: Path = typer.Option(Path(".")),
    provider_key: str = typer.Option("OPENROUTER_API_KEY", help="Name of the provider credential to forward."),
) -> None:
    """Start up to 120 isolated workers; rollback all new sandboxes if setup fails."""
    from daytona import CreateSandboxFromSnapshotParams

    api = client()
    environment = provider_environment(provider_key)
    if pool.exists():
        raise ValueError(f"Pool manifest already exists: {pool}. Reuse it or delete its workers first.")
    identifier = uuid4().hex[:12]
    lock = threading.Lock()
    with tempfile.TemporaryDirectory(prefix="pocket-daytona-build-") as temporary:
        wheel = build_wheel(project, Path(temporary))
        with ExitStack() as rollback, ThreadPoolExecutor(max_workers=min(workers, 10)) as executor:

            def create_worker(index: int) -> dict:
                sandbox = api.create(
                    CreateSandboxFromSnapshotParams(
                        snapshot=snapshot,
                        name=f"pocket-bcp-{identifier}-{index}",
                        os_user="daytona",
                        env_vars=environment,
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
                "provider_key": provider_key,
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


def select_matrix_instances(
    queries: Path, split: str, query_ids: list[str], limit: int, seed: int | None
) -> list[dict]:
    instances = load_instances(queries, split=split, query_ids=query_ids)
    if set(query_ids) - {str(instance["query_id"]) for instance in instances}:
        raise ValueError("Some requested query IDs are outside the selected split")
    if limit < 0 or limit > len(instances):
        raise ValueError(f"Select between 0 and {len(instances)} questions; 0 selects all")
    if seed is not None:
        instances = random.Random(seed).sample(instances, limit or len(instances))
    elif limit:
        instances = instances[:limit]
    return [{"query_id": str(instance["query_id"]), "query": instance["query"]} for instance in instances]


def matrix_query_bytes(instances: list[dict]) -> bytes:
    return "".join(json.dumps(instance, ensure_ascii=False) + "\n" for instance in instances).encode()


def matrix_config(config_spec: list[str]) -> dict:
    return recursive_merge(
        get_config_from_spec("browsecomp_plus"),
        *[get_config_from_spec(spec) for spec in (config_spec or ["browsecomp_plus_deepseek"])],
    )


def matrix_provider_key(config: dict) -> str:
    if not config["model"].get("model_name"):
        raise ValueError("Include a provider profile, e.g. -c browsecomp_plus_deepseek, before other overrides")
    provider = (
        "openrouter"
        if config["model"].get("model_class") == "openrouter"
        else config["model"]["model_name"].split("/")[0]
    )
    keys = {"deepseek": "DEEPSEEK_API_KEY", "meta": "META_API_KEY", "openrouter": "OPENROUTER_API_KEY"}
    if provider not in keys:
        raise ValueError("Matrix supports the DeepSeek, Meta, and OpenRouter provider profiles")
    return keys[provider]


def matrix_arguments(
    remote: str, method: str, split: str, *, query_id: str | None = None, workers: int = 1
) -> list[str]:
    return [
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
        split,
        "--method",
        method,
        "-c",
        f"{remote}/config.yaml",
        "--workers",
        str(workers),
    ] + (["--execute-query-id", query_id] if query_id is not None else [])


def run_remote_method(
    sandbox: Any,
    instances: list[dict],
    config: dict,
    output: Path,
    method: str,
    split: str,
    *,
    query_id: str | None = None,
    workers: int = 1,
) -> None:
    from daytona.common.process import SessionExecuteRequest

    from pocketcontextagent.run.benchmarks.browsecomp_plus import implementation_hash

    check = sandbox.process.exec(
        worker_command(
            [
                "python",
                "-c",
                "from pocketcontextagent.run.benchmarks.browsecomp_plus import implementation_hash; "
                f"assert implementation_hash() == {implementation_hash()!r}, 'Stale worker package; create a fresh pool'",
            ]
        ),
        timeout=120,
    )
    if check.exit_code:
        raise RuntimeError(f"{method}: worker package preflight failed: {check.result}")
    expected = [instance for instance in instances if query_id is None or instance["query_id"] == query_id]
    if not expected:
        raise ValueError(f"Task {query_id} is outside the selected cohort")
    session = f"matrix-{uuid4().hex}"
    remote = f"/workspace/{session}"
    output.mkdir(parents=True)
    (output / "worker.yaml").write_text(
        yaml.safe_dump({"sandbox_id": sandbox.id, "query_ids": [instance["query_id"] for instance in expected]}),
        encoding="utf-8",
    )
    sandbox.fs.create_folder(remote, "700")
    typer.echo(f"{method}: recovery workspace on {sandbox.id}: {remote}")
    with ExitStack() as cleanup:
        sandbox.fs.upload_file(matrix_query_bytes(instances), f"{remote}/query.jsonl")
        sandbox.fs.upload_file(yaml.safe_dump(config).encode(), f"{remote}/config.yaml")
        sandbox.process.create_session(session)
        cleanup.callback(sandbox.process.delete_session, session)
        execution = sandbox.process.execute_session_command(
            session,
            SessionExecuteRequest(
                command=worker_command(matrix_arguments(remote, method, split, query_id=query_id, workers=workers)),
                run_async=True,
            ),
        )
        typer.echo(f"{method}: launched {len(expected)} questions on {sandbox.id}")
        deadline = time.monotonic() + len(expected) * ((config["agent"].get("wall_time_limit_seconds") or 1800) + 180)
        timed_out = False
        while sandbox.process.get_session_command(session, execution.cmd_id).exit_code is None:
            if time.monotonic() >= deadline:
                timed_out = True
                break
            time.sleep(max(5, workers))
        detail = sandbox.process.get_session_command_logs(session, execution.cmd_id).output[-4000:]
    # Stop the session before collecting; retain its workspace if collection fails.
    files = []
    if any(file.name == "output" for file in sandbox.fs.list_files(remote)):
        files = [file.name for file in sandbox.fs.list_files(f"{remote}/output")]
    for name in ["run.yaml", "summary.yaml", *[f"{instance['query_id']}.json" for instance in expected]]:
        if name in files:
            temporary = output / f"{name}.tmp"
            sandbox.fs.download_file(f"{remote}/output/{name}", str(temporary))
            temporary.replace(output / name)
    if "trajectories" in files:
        archive = f"{remote}/trajectories.tar.gz"
        response = sandbox.process.exec(
            shlex.join(["tar", "-czf", archive, "-C", f"{remote}/output", "trajectories"]), timeout=120
        )
        if response.exit_code:
            raise RuntimeError(f"{method}: failed to archive trajectories")
        sandbox.fs.download_file(archive, str(output / "trajectories.tar.gz.tmp"))
        (output / "trajectories.tar.gz.tmp").replace(output / "trajectories.tar.gz")
    missing = [instance["query_id"] for instance in expected if f"{instance['query_id']}.json" not in files]
    unexpected = {name for name in files if name.endswith(".json")} - {
        f"{instance['query_id']}.json" for instance in expected
    }
    if timed_out:
        raise TimeoutError(
            f"{method}: wall-time allowance exceeded; partial artifacts collected, workspace retained at {remote}"
        )
    if missing or unexpected or "run.yaml" not in files:
        for key, value in os.environ.items():
            if key.endswith("_API_KEY") and value:
                detail = detail.replace(value, "[REDACTED]")
        raise RuntimeError(
            f"{method}: missing artifacts for {missing}, unexpected results {sorted(unexpected)}; "
            f"workspace retained at {remote}; {detail}"
        )
    sandbox.fs.delete_file(remote, recursive=True)
    typer.echo(f"{method}: artifacts collected in {output}")


def matrix_worker_groups(manifest: dict, requested: int, tasks: int) -> dict[str, list[dict]]:
    if not 1 <= requested <= 30 or tasks < 1:
        raise ValueError("Select 1 to 30 workers per method and at least one task")
    active = min(requested, tasks)
    required = len(METHODS) * active
    workers = manifest["sandboxes"][:required]
    if len(workers) != required or len({worker["id"] for worker in workers}) != required:
        raise ValueError(f"Matrix requires {required} distinct sandboxes; create a pool with --workers {required}")
    return {method: workers[index * active : (index + 1) * active] for index, method in enumerate(METHODS)}


def run_matrix_workers(
    groups: dict[str, list[dict]],
    instances: list[dict],
    execute: Callable[[str, dict, dict], None],
) -> None:
    queues = {method: iter(instances) for method in groups}
    locks = {method: threading.Lock() for method in groups}

    def run_worker(item: tuple[str, dict]) -> None:
        method, worker = item
        while True:
            with locks[method]:
                instance = next(queues[method], None)
            if instance is None:
                return
            execute(method, worker, instance)

    with ThreadPoolExecutor(max_workers=sum(map(len, groups.values()))) as executor:
        list(executor.map(run_worker, [(method, worker) for method, workers in groups.items() for worker in workers]))


def publish_matrix_task(task: Path, output: Path, query_id: str, expected: dict) -> None:
    """Publish an unchanged task result; callers serialize publication per method."""
    if {path.name for path in task.glob("*.json")} != {f"{query_id}.json"}:
        raise ValueError(f"{task}: expected exactly one result for the assigned query")
    manifest = yaml.safe_load((task / "run.yaml").read_text())
    result = json.loads((task / f"{query_id}.json").read_text())
    fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    if (
        manifest["method"] != output.name
        or query_id not in manifest["selected_query_ids"]
        or any(manifest.get(key) != value for key, value in expected.items())
        or not manifest["retain_recorded_attempts"]
        or str(result["query_id"]) != query_id
        or result.get("metadata", {}).get("fingerprint") != fingerprint
    ):
        raise ValueError(f"{task}: task result does not match its distributed cohort manifest")
    manifest_path = output / "run.yaml"
    if manifest_path.exists() and yaml.safe_load(manifest_path.read_text()) != manifest:
        raise ValueError(f"{task}: worker settings, implementation, or data differ from the method's other tasks")
    output.mkdir(parents=True, exist_ok=True)
    for source, destination in [
        (task / "run.yaml", manifest_path),
        (task / f"{query_id}.json", output / f"{query_id}.json"),
    ]:
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        shutil.copyfile(source, temporary)
        temporary.replace(destination)
    results = [json.loads(path.read_text()) for path in sorted(output.glob("*.json"))]
    (output / "summary.yaml").write_text(yaml.safe_dump(summarize_results(results)), encoding="utf-8")


@app.command("matrix")
def run_matrix(
    pool: Path = typer.Option(DEFAULT_POOL),
    data_dir: Path = typer.Option(Path("data/browsecomp-plus")),
    split: str | None = typer.Option(None, help="Defaults to development680 for corpus tasks or test for coding."),
    query_id: list[str] = typer.Option([], help="Select specific IDs (repeatable)."),
    limit: int = typer.Option(1, min=0, help="Questions per method; 0 selects the entire split."),
    seed: int | None = typer.Option(None, help="Sample once without replacement using this random seed."),
    config_spec: list[str] = typer.Option([], "-c", "--config", help="Shared overrides; defaults to direct DeepSeek."),
    output: Path = typer.Option(Path("runs/browsecomp-plus/matrix")),
    dry_run: bool = typer.Option(False, help="Validate selection/config only; no API calls or writes."),
    workers_per_method: int = typer.Option(
        1, min=1, max=30, help="Dedicated sandboxes per method; one task per sandbox."
    ),
    benchmark: str = typer.Option("browsecomp-plus", help="browsecomp-plus or swebench-verified."),
) -> None:
    """Run matched questions with up to 30 parallel sandboxes per method."""
    from pocketcontextagent.comparisons.report import compare_runs, comparison_markdown
    from pocketcontextagent.run.benchmarks.browsecomp_plus import implementation_hash

    if benchmark == "swebench-verified":
        from pocketcontextagent.run.benchmarks.benchmark_matrix import run_matrix as run_coding_matrix

        if pool != DEFAULT_POOL:
            raise ValueError("Coding tasks create fresh official instance-image sandboxes; omit --pool")
        run_coding_matrix(
            data_dir=Path("data/swebench-verified") if data_dir == Path("data/browsecomp-plus") else data_dir,
            split=split or "test",
            query_ids=query_id,
            limit=limit,
            seed=seed,
            config_spec=config_spec,
            output=Path("runs/swebench-verified/matrix") if output == Path("runs/browsecomp-plus/matrix") else output,
            dry_run=dry_run,
            workers_per_method=workers_per_method,
        )
        return
    if benchmark != "browsecomp-plus":
        raise ValueError("Select browsecomp-plus or swebench-verified")
    split = split or "development680"
    instances = select_matrix_instances(data_dir / "queries.jsonl", split, query_id, limit, seed)
    config = matrix_config(config_spec)
    for method in METHODS:
        configure_method(config, method)
    provider_key = matrix_provider_key(config)
    selection = {
        "split": split,
        "seed": seed,
        "model": config["model"]["model_name"],
        "workers_per_method": workers_per_method,
        "active_workers_per_method": min(workers_per_method, len(instances)),
        "selected_query_ids": [instance["query_id"] for instance in instances],
    }
    typer.echo(yaml.safe_dump(selection, sort_keys=False))
    if dry_run:
        return
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a fresh --output directory; matrix never replaces or retries recorded attempts")
    manifest = json.loads(pool.read_text())
    groups = matrix_worker_groups(manifest, workers_per_method, len(instances))
    active = selection["active_workers_per_method"]
    if manifest.get("provider_key", "OPENROUTER_API_KEY") != provider_key:
        raise ValueError("Pool credentials do not match the model; create a fresh pool with --provider-key NAME")
    api = client()
    selection["sandboxes"] = {method: [worker["id"] for worker in workers] for method, workers in groups.items()}
    output.mkdir(parents=True, exist_ok=True)
    (output / "selection.yaml").write_text(yaml.safe_dump(selection, sort_keys=False), encoding="utf-8")

    if active == 1:

        def run_method(method: str) -> None:
            run_remote_method(api.get(groups[method][0]["id"]), instances, config, output / method, method, split)

        with ThreadPoolExecutor(max_workers=len(METHODS)) as executor:
            list(executor.map(run_method, METHODS))
    else:
        locks = {method: threading.Lock() for method in METHODS}
        expected = {
            "split": split,
            "workers": active,
            "selected_query_ids": selection["selected_query_ids"],
            "retain_recorded_attempts": True,
            "queries_sha256": hashlib.sha256(matrix_query_bytes(instances)).hexdigest(),
            "implementation_sha256": implementation_hash(),
            "dataset_revision": DATASET_REVISION,
            "index_revision": INDEX_REVISION,
        }
        manifests = {
            method: expected
            | {
                "method": method,
                "config": configure_method(config, method),
                "method_provenance": method_provenance(configure_method(config, method), method),
            }
            for method in METHODS
        }

        def execute(method: str, worker: dict, instance: dict) -> None:
            query_id = instance["query_id"]
            task = output / method / "tasks" / query_id
            run_remote_method(
                api.get(worker["id"]), instances, config, task, method, split, query_id=query_id, workers=active
            )
            with locks[method]:
                publish_matrix_task(task, output / method, query_id, manifests[method])
            typer.echo(f"{method}: recorded {query_id} from sandbox {worker['id']}")

        run_matrix_workers(groups, instances, execute)
    report = compare_runs(output)
    (output / "comparison.yaml").write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")
    typer.echo(comparison_markdown(report))
    if any(row["completed"] < row["tasks"] for row in report["methods"].values()):
        raise typer.Exit(1)


@app.command("delete")
def delete_pool(pool: Path = typer.Option(DEFAULT_POOL)) -> None:
    """Delete only the sandboxes recorded in this pool's local manifest."""
    api = client()
    workers = {worker["id"] for worker in json.loads(pool.read_text())["sandboxes"]}
    sandboxes = [sandbox for sandbox in api.list() if sandbox.id in workers]
    with ThreadPoolExecutor(max_workers=10) as executor:
        list(executor.map(api.delete, sandboxes))
    pool.unlink()
    typer.echo("Pool deleted")


@app.command("prepare")
def prepare_benchmark(
    benchmark: str = typer.Option("swebench-verified"),
    data_dir: Path | None = typer.Option(None),
) -> None:
    """Prepare pinned public tasks and separate private grading data; no model calls."""
    if benchmark == "swebench-verified":
        from pocketcontextagent.run.benchmarks.benchmark_matrix import prepare

        prepare(data_dir or Path("data/swebench-verified"))
    elif benchmark == "browsecomp-plus":
        from pocketcontextagent.run.benchmarks.utils.browsecomp_plus import prepare_data

        prepare_data(data_dir or Path("data/browsecomp-plus"))
    else:
        raise ValueError("Select browsecomp-plus or swebench-verified")


@app.command("allocation")
def show_allocation(
    browsecomp_data: Path = typer.Option(Path("data/browsecomp-plus")),
    swebench_data: Path = typer.Option(Path("data/swebench-verified")),
    output: Path | None = typer.Option(None),
) -> None:
    """Freeze all 150 corpus and 125 sampled coding tasks; never starts sandboxes."""
    from pocketcontextagent.run.benchmarks.benchmark_allocation import allocation

    allocation(browsecomp_data, swebench_data, output)


@app.command("compare")
def compare_matrix(
    runs: Path = typer.Option(...), grades: Path | None = typer.Option(None), output: Path | None = typer.Option(None)
) -> None:
    """Compare matched attempts; coding correctness requires official test reports."""
    from pocketcontextagent.run.benchmarks.browsecomp_plus import compare

    compare(runs=runs, grades=grades, output=output)


@app.command("export-swebench")
def export_swebench(
    runs: Path = typer.Option(...),
    data_dir: Path = typer.Option(Path("data/swebench-verified")),
    run_prefix: str = typer.Option(...),
    max_workers: int = typer.Option(1, min=1),
    harness_python: str = typer.Option("python", help="Python executable in the trusted grading environment."),
) -> None:
    """Export predictions and print trusted Docker-host grading commands; never executes them."""
    from pocketcontextagent.run.benchmarks.benchmark_grading import export_evaluation

    export_evaluation(runs, data_dir, run_prefix, max_workers, harness_python)


@app.command("import-swebench")
def import_swebench(
    runs: Path = typer.Option(...),
    logs: Path = typer.Option(Path("logs/run_evaluation")),
    run_prefix: str = typer.Option(...),
    output: Path = typer.Option(...),
) -> None:
    """Validate and import official reports bound to the exact predicted patches."""
    from pocketcontextagent.run.benchmarks.benchmark_grading import import_evaluation

    import_evaluation(runs, logs, run_prefix, output)
