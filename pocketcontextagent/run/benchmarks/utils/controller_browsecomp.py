"""Trusted-controller inference with credential-free, pooled BM25 corpus workers."""

import hashlib
import json
import math
import os
import sys
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Protocol
from uuid import uuid4

import typer
import yaml

from pocketcontextagent import __version__
from pocketcontextagent.comparisons.registry import CONTEXT_WINDOW_HELP, METHODS, configure_method, method_provenance
from pocketcontextagent.comparisons.report import compare_runs, comparison_markdown
from pocketcontextagent.models import get_model
from pocketcontextagent.run.benchmarks.benchmark_matrix import archive_trajectories
from pocketcontextagent.run.benchmarks.browsecomp_plus import implementation_hash, run_instance
from pocketcontextagent.run.benchmarks.browsecomp_plus_daytona import (
    REMOTE_INDEX,
    TOKENIZER,
    client,
    install_worker,
    matrix_config,
    matrix_provider_key,
    matrix_query_bytes,
    provider_environment,
    publish_matrix_task,
    run_matrix_workers,
    select_matrix_instances,
    worker_command,
)
from pocketcontextagent.run.benchmarks.utils.browsecomp_plus import DATASET_REVISION, INDEX_REVISION
from pocketcontextagent.run.benchmarks.utils.controller_artifacts import ArtifactStore, require_free_space
from pocketcontextagent.run.benchmarks.utils.controller_runtime import check_runtime
from pocketcontextagent.run.benchmarks.utils.controller_setup import parse_remote_json
from pocketcontextagent.run.benchmarks.utils.controller_volumes import verify_volume_mount
from pocketcontextagent.run.benchmarks.utils.swebench_daytona import (
    ResourceJournal,
    RuntimeConfig,
    delete_resources,
    sandbox_metadata,
)

DEFAULT_SNAPSHOT = "pocket-browsecomp-plus-20261022"
PROTOCOL = "sdk-process-exec/bm25-json-v1"
PUBLIC_ENV = {
    "HF_HOME": "/opt/huggingface",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "JAVA_HOME": "/usr/lib/jvm/java-21-openjdk-amd64",
    "MSWEA_SILENT_STARTUP": "1",
    "MSWEA_GLOBAL_CONFIG_DIR": "/tmp/pca-corpus-config",
    "LITELLM_LOCAL_MODEL_COST_MAP": "True",
}
RETRIEVAL_CODE = """
import json, sys
from pathlib import Path
from pocketcontextagent.run.benchmarks.utils.browsecomp_plus import BM25Retriever
args = json.loads(sys.argv[1])
assert set(args) == {'operation', 'arguments', 'index'}
assert args['operation'] in ('search', 'get_document')
r = BM25Retriever(Path(args['index']))
try:
    if args['operation'] == 'search':
        result = r.search(*args['arguments'])
    else:
        result = r.get_document(*args['arguments'])
    print('PCA_CORPUS_RESULT:' + json.dumps(result, ensure_ascii=False, allow_nan=False))
finally:
    r.close()
"""


class CommandBackend(Protocol):
    def __call__(self, command: str, *, timeout: int) -> dict: ...


class RemoteRetriever:
    def __init__(self, backend: CommandBackend, index: Path = Path(REMOTE_INDEX)):
        self.backend, self.index = backend, index

    def _command(self, operation: str, arguments: list) -> object:
        if operation not in ("search", "get_document"):
            raise ValueError("Unsupported retrieval operation")
        response = self.backend(
            worker_command(
                [
                    "python",
                    "-c",
                    RETRIEVAL_CODE,
                    json.dumps({"operation": operation, "arguments": arguments, "index": str(self.index)}),
                ]
            ),
            timeout=120,
        )
        if (
            not isinstance(response, dict)
            or type(response.get("returncode")) is not int
            or not isinstance(response.get("output"), str)
        ):
            raise ValueError("Invalid retrieval command response")
        if response["returncode"]:
            raise RuntimeError(f"Corpus retrieval failed (exit {response['returncode']})")
        result = parse_remote_json(response["output"], "PCA_CORPUS_RESULT:")

        def document(row: object) -> bool:
            return isinstance(row, dict) and isinstance(row.get("docid"), str) and isinstance(row.get("text"), str)

        if operation == "search":
            if (
                not isinstance(result, list)
                or len(result) > arguments[1]
                or any(
                    not document(row) or type(row.get("score")) not in (int, float) or not math.isfinite(row["score"])
                    for row in result
                )
            ):
                raise ValueError("Invalid search response")
        elif result is not None and (not document(result) or result["docid"] != arguments[0]):
            raise ValueError("Invalid document response")
        return result

    def search(self, query: str, k: int) -> list[dict]:
        if not isinstance(query, str) or type(k) is not int or k < 1:
            raise ValueError("Search requires text and a positive k")
        return self._command("search", [query, k])

    def get_document(self, docid: str) -> dict | None:
        if not isinstance(docid, str):
            raise ValueError("Document ID must be text")
        return self._command("get_document", [docid])

    def close(self) -> None:
        pass


def plan(
    data_dir: Path,
    config_spec: list[str],
    limit: int,
    seed: int | None,
    workers: int,
    query_ids: list[str],
    split: str = "eval150",
    method: str | None = None,
    context_window: int | None = None,
) -> tuple[dict, dict]:
    if not 1 <= workers <= 30:
        raise ValueError("Select 1 to 30 workers per method")
    if method is not None and method not in METHODS:
        raise ValueError(f"Unknown method: {method}; select one of {', '.join(METHODS)}")
    methods = [method] if method is not None else list(METHODS)
    instances = select_matrix_instances(data_dir / "queries.jsonl", split, query_ids, limit, seed)
    config = matrix_config(config_spec, context_window)
    matrix_provider_key(config)
    if "api_key" in config["model"].get("model_kwargs", {}):
        raise ValueError("Keep provider keys outside saved configurations")
    for method in methods:
        configure_method(config, method)
    active = min(workers, len(instances))
    ttl = max(
        240, math.ceil(len(instances) / active) * math.ceil(config["agent"]["wall_time_limit_seconds"] / 60) + 120
    )
    configured_ttl = config.get("corpus_ttl_minutes", ttl)
    if type(configured_ttl) is not int or configured_ttl < ttl:
        raise ValueError("Corpus pool TTL must cover all task waves and collection grace")
    return config, {
        "benchmark": "browsecomp-plus",
        "split": split,
        "seed": seed,
        "limit": limit,
        "query_ids": query_ids,
        "implementation_sha256": implementation_hash(),
        "model": config["model"]["model_name"],
        "context_token_limit": config["agent"]["context_token_limit"],
        "workers_per_method": workers,
        "active_workers_per_method": active,
        "methods": methods,
        "maximum_active_sandboxes": active * len(methods),
        "selected_query_ids": [row["query_id"] for row in instances],
        "instances": instances,
        "queries_sha256": hashlib.sha256(matrix_query_bytes(instances)).hexdigest(),
        "attempts": len(instances) * len(methods),
        "planned_attempts": len(instances) * len(methods),
        "sandbox_ttl_minutes": configured_ttl,
        "retrieval_protocol": PROTOCOL,
    }


def resolve_snapshot(api, name: str) -> dict:
    snapshot = api.snapshot.get(name)
    if getattr(snapshot.state, "value", snapshot.state) != "active" or (snapshot.cpu, snapshot.mem, snapshot.disk) != (
        1,
        4,
        10,
    ):
        raise ValueError("Corpus snapshot must be active with cpu=1, memory=4, disk=10")
    return {"name": snapshot.name, "id": snapshot.id}


def run_manifest(config: dict, selection: dict, method: str, snapshot: dict) -> dict:
    configured = configure_method(config, method)
    return {
        "benchmark": "browsecomp-plus",
        "config": configured,
        "method": method,
        "method_description": METHODS[method].description,
        "method_provenance": method_provenance(configured, method),
        "split": selection["split"],
        "workers": selection["active_workers_per_method"],
        "selected_query_ids": selection["selected_query_ids"],
        "retain_recorded_attempts": True,
        "queries_sha256": selection["queries_sha256"],
        "implementation_sha256": implementation_hash(),
        "dataset_revision": DATASET_REVISION,
        "index_revision": INDEX_REVISION,
        "index": REMOTE_INDEX,
        "index_files": [],
        "index_sha256": config.get("corpus_index_sha256"),
        "version": __version__,
        "snapshot": snapshot,
        "runtime": "trusted-controller/daytona-corpus",
        "retrieval_protocol": PROTOCOL,
        "controller_id": os.getenv("MSWEA_CONTROLLER_ID"),
        "artifact_volume_id": os.getenv("MSWEA_ARTIFACT_VOLUME_ID"),
    }


def worker_params(name: str, snapshot: str, method: str, runtime: RuntimeConfig):
    from daytona import CreateSandboxFromSnapshotParams

    return CreateSandboxFromSnapshotParams(
        name=name,
        snapshot=snapshot,
        os_user="daytona",
        public=False,
        network_block_all=True,
        ephemeral=False,
        auto_stop_interval=0,
        auto_pause_interval=0,
        auto_delete_interval=-1,
        ttl_minutes=runtime.ttl_minutes,
        env_vars=PUBLIC_ENV,
        labels={"project": "pocket-context", "benchmark": "browsecomp-plus", "role": "corpus-worker", "method": method},
    )


def check_worker(
    sandbox, expected_hash: str, tokenizer_revision: str | None = None, index_hashes: dict[str, str] | None = None
) -> RemoteRetriever:
    code = (
        "import os,json; from pathlib import Path; "
        "from pocketcontextagent.run.benchmarks.browsecomp_plus import implementation_hash; "
        "assert not any(k.endswith(('_API_KEY', '_TOKEN')) for k in os.environ), 'Frozen worker credentials'; "
        f"assert implementation_hash() == {expected_hash!r}, 'Worker implementation differs'; "
        f"assert Path({REMOTE_INDEX!r}).is_dir(); "
        "from transformers import AutoTokenizer; "
        f"assert AutoTokenizer.from_pretrained({TOKENIZER!r}, revision={tokenizer_revision!r}, local_files_only=True).encode('ready'); "
        "print('PCA_CORPUS_CHECK:' + json.dumps({'ready': True}))"
    )
    if index_hashes:
        if any(Path(name).name != name for name in index_hashes):
            raise ValueError("Index hashes must name direct index files")
        verify_code = (
            "import hashlib,json; from pathlib import Path; expected=json.loads(__import__('sys').argv[1]); root=Path("
            + repr(REMOTE_INDEX)
            + "); actual={p.name:hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in root.iterdir() if p.is_file()}; assert actual==expected, 'Corpus index differs from its frozen hashes'"
        )
        response = sandbox.process.exec(
            worker_command(["python", "-c", verify_code, json.dumps(index_hashes)]), timeout=180
        )
        if response.exit_code:
            raise ValueError("Corpus index checksum preflight failed")
    response = sandbox.process.exec(worker_command(["python", "-c", code]), timeout=120)
    if response.exit_code or parse_remote_json(response.result, "PCA_CORPUS_CHECK:") != {"ready": True}:
        raise RuntimeError("Corpus worker package/index/tokenizer preflight failed")

    def backend(command: str, *, timeout: int) -> dict:
        response = sandbox.process.exec(command, timeout=timeout)
        return {"output": response.result, "returncode": response.exit_code}

    retriever = RemoteRetriever(backend)
    hits = retriever.search("jazz festival", 1)
    if not hits or retriever.get_document(hits[0]["docid"]) != {"docid": hits[0]["docid"], "text": hits[0]["text"]}:
        raise RuntimeError("Corpus search/document preflight failed")
    return retriever


def run_matrix(
    *,
    data_dir: Path,
    output: Path,
    config_spec: list[str],
    limit: int,
    seed: int | None,
    workers_per_method: int,
    method: str | None = None,
    split: str = "eval150",
    query_ids: list[str] | None = None,
    snapshot: str = DEFAULT_SNAPSHOT,
    wheel: Path | None = None,
    dry_run: bool = False,
    preflight: bool = False,
    context_window: int | None = None,
) -> None:
    config, selection = plan(
        data_dir, config_spec, limit, seed, workers_per_method, query_ids or [], split, method, context_window
    )
    methods = selection["methods"]
    selection.update(snapshot={"name": snapshot}, preflight_only=preflight)
    if preflight:
        selection["attempts"] = 0
    if dry_run:
        typer.echo(yaml.safe_dump(selection, sort_keys=False))
        return
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a fresh output directory; recorded attempts are never retried")
    if wheel is None or not wheel.is_file():
        raise ValueError("Provide the staged public --wheel")
    archive_job = os.getenv("MSWEA_ARTIFACT_JOB_ID", "")
    archive_root = os.getenv("MSWEA_ARTIFACT_STORE_ROOT")
    if archive_root:
        verify_volume_mount(Path(archive_root).parent)
    store = ArtifactStore(Path(archive_root)) if archive_root else None
    min_free = int(os.getenv("MSWEA_ARTIFACT_MIN_FREE_BYTES", str(1024**3)))
    if os.getenv("MSWEA_CONTROLLER_ID") and store is None:
        raise ValueError("Managed corpus runs require a mounted artifact volume")

    def admission() -> None:
        require_free_space(output.parent, min_free)
        if store is not None:
            if not archive_job:
                raise ValueError("Volume archival requires a job identifier")
            verify_volume_mount(store.root.parent)

    admission()
    if tokenizer := config.get("environment", {}).get("tokenizer_name"):
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained(
            tokenizer, revision=config["environment"].get("tokenizer_revision"), local_files_only=True
        )
    if not preflight:
        provider_environment(matrix_provider_key(config))
        if os.getenv("MSWEA_CONTROLLER_ID"):
            check_runtime()
        for method in methods:
            model = get_model(config=configure_method(config, method)["model"])
            if hasattr(model.config, "action_regex") or not callable(getattr(model, "get_tools", None)):
                raise ValueError("Corpus matrix requires a native-tool model")
    api = client()
    selection["snapshot"] = resolve_snapshot(api, snapshot)
    if config.get("corpus_snapshot") is not None and config["corpus_snapshot"] != selection["snapshot"]:
        raise ValueError("Corpus snapshot identity differs from its frozen preparation")
    runtime = RuntimeConfig(cpu=1, memory=4, disk=10, ttl_minutes=selection["sandbox_ttl_minutes"])
    selection["sandbox"] = asdict(runtime)
    output.mkdir(parents=True, exist_ok=True)
    (output / "selection.yaml").write_text(yaml.safe_dump(selection, sort_keys=False))
    manifests = {method: run_manifest(config, selection, method, selection["snapshot"]) for method in methods}
    if not preflight:
        for method, manifest in manifests.items():
            (output / method).mkdir()
            (output / method / "run.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    journal = ResourceJournal(output / "sandbox-cleanup.json")
    groups = {method: [] for method in methods}
    locks = {method: threading.Lock() for method in methods}
    archive_lock = threading.Lock()
    aborted = threading.Event()

    def execute(method: str, worker: dict, instance: dict) -> None:
        if aborted.is_set():
            raise RuntimeError("Corpus admission stopped after an uncaptured attempt")
        iid, name = instance["query_id"], worker["name"]
        task = output / method / "tasks" / iid
        started = captured = False
        try:
            admission()
            task.mkdir(parents=True)
            manifest = manifests[method]
            (task / "run.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
            (task / "worker.yaml").write_text(yaml.safe_dump({"id": worker["id"], "name": name}))
            fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
            journal.record("sandboxes", name, query_id=iid, phase="running_agent", retain=True)
            started = True
            try:
                run_instance(instance, task, manifest["config"], worker["retriever"], fingerprint)
            finally:
                if (task / "trajectories").is_dir():
                    archive_trajectories(task)
                if (task / f"{iid}.json").is_file():
                    with locks[method]:
                        publish_matrix_task(task, output / method, iid, manifest)
                if (task / f"{iid}.json").is_file() and (task / "trajectories.tar.gz").is_file():
                    if store is not None:
                        with archive_lock:
                            verify_volume_mount(store.root.parent)
                            store.archive_task(task, archive_job, method, iid, prune=True)
                    captured = True
                    journal.record("sandboxes", name, phase="captured", retain=False)
        finally:
            error = sys.exc_info()[1]
            if error is not None:
                aborted.set()
                if started and not captured:
                    api.stop(api.get(worker["id"]))
                journal.failure("sandboxes", name, error)

    try:
        for method in methods:
            for slot in range(selection["active_workers_per_method"]):
                admission()
                name = f"pca-bcp-{method.replace('_', '-')}-{uuid4().hex[:12]}"
                journal.record("sandboxes", name, phase="creating", method=method, slot=slot, snapshot=snapshot)
                sandbox = api.create(
                    worker_params(name, selection["snapshot"]["id"], method, runtime),
                    timeout=runtime.startup_timeout_seconds,
                )
                journal.record(
                    "sandboxes",
                    name,
                    id=sandbox.id,
                    phase="checking",
                    lifecycle=sandbox_metadata(api.get(sandbox.id).model_dump(), runtime),
                )
                install_worker(sandbox, wheel)
                retriever = check_worker(
                    sandbox,
                    manifests[method]["implementation_sha256"],
                    config["environment"].get("tokenizer_revision"),
                    config.get("corpus_index_sha256") if preflight else None,
                )
                groups[method].append({"name": name, "id": sandbox.id, "retriever": retriever})
                journal.record("sandboxes", name, phase="ready", model_calls=0)
        if not preflight:
            run_matrix_workers(groups, selection["instances"], execute)
    finally:
        delete_resources(api, journal.document(), runtime.cleanup_timeout_seconds, preserve_retained=True)
        for row in journal.document()["sandboxes"]:
            if not row.get("retain"):
                journal.record("sandboxes", row["name"], phase="destroyed", cleanup_confirmed=True)
        confirmed = not any(row.get("retain") for row in journal.document()["sandboxes"])
        (output / "cleanup.yaml").write_text(yaml.safe_dump({"confirmed": confirmed}))
    if preflight:
        (output / "preflight.yaml").write_text(
            yaml.safe_dump(
                {"model_calls": 0, "cleanup_confirmed": confirmed, "workers": journal.document()["sandboxes"]}
            )
        )
        return
    report = compare_runs(output, methods=methods)
    (output / "comparison.yaml").write_text(yaml.safe_dump(report, sort_keys=False))
    typer.echo(comparison_markdown(report))
    if any(row["completed"] < row["tasks"] for row in report["methods"].values()):
        raise typer.Exit(1)


app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


@app.command()
def main(
    data_dir: Path = typer.Option(...),
    output: Path = typer.Option(...),
    config_spec: list[str] = typer.Option([], "-c", "--config"),
    context_window: int | None = typer.Option(None, min=1, help=CONTEXT_WINDOW_HELP),
    limit: int = 0,
    seed: int = 42,
    workers_per_method: int | None = typer.Option(
        None, min=1, max=30, help="Defaults to 6 with --method, otherwise 10."
    ),
    method: str | None = typer.Option(None, help="Run only this method; omitted runs all four methods."),
    split: str = "eval150",
    snapshot: str = DEFAULT_SNAPSHOT,
    wheel: Path | None = None,
    dry_run: bool = False,
    preflight: bool = False,
    query_id: list[str] = typer.Option([], "--query-id"),
) -> None:
    run_matrix(
        data_dir=data_dir,
        output=output,
        config_spec=config_spec,
        context_window=context_window,
        limit=limit,
        seed=seed,
        workers_per_method=workers_per_method if workers_per_method is not None else (6 if method is not None else 10),
        method=method,
        split=split,
        query_ids=query_id,
        snapshot=snapshot,
        wheel=wheel,
        dry_run=dry_run,
        preflight=preflight,
    )


if __name__ == "__main__":
    app()
