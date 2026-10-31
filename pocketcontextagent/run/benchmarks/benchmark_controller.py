"""Durable Daytona controllers for isolated coding and corpus workers."""

import hashlib
import json
import os
import shlex
import subprocess
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

import typer
import yaml

from pocketcontextagent.comparisons.registry import CONTEXT_WINDOW_HELP, METHODS
from pocketcontextagent.run.benchmarks.browsecomp_plus import implementation_hash
from pocketcontextagent.run.benchmarks.browsecomp_plus_daytona import (
    build_wheel,
    client,
    matrix_provider_key,
    provider_environment,
)
from pocketcontextagent.run.benchmarks.utils.controller_runtime import RUNTIME_REQUIREMENTS
from pocketcontextagent.run.benchmarks.utils.controller_setup import (
    ControllerConfig,
    extract_artifacts,
    parse_remote_json,
    provider_endpoint,
    sandbox_params,
    write_private_json,
)
from pocketcontextagent.run.benchmarks.utils.controller_volumes import (
    archive_environment,
    resolve_volume,
    validate_volume_name,
)
from pocketcontextagent.run.benchmarks.utils.swebench_daytona import RuntimeConfig, delete_resources, sandbox_metadata

app = typer.Typer(add_completion=False, no_args_is_help=True, pretty_exceptions_show_locals=False)
DEFAULT_MANIFEST = Path("runs/context-evaluation/controller.json")
ROOT = "/workspace/pocket-controller"
PYTHON = "/usr/local/bin/python"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _exec(sandbox, command: list[str], *, timeout: int = 60) -> str:
    response = sandbox.process.exec(shlex.join(command), cwd=ROOT, timeout=timeout)
    if response.exit_code:
        text = response.result
        for name, value in os.environ.items():
            if name.endswith("_API_KEY") and value:
                text = text.replace(value, "[REDACTED]")
        raise RuntimeError(f"Controller command failed ({response.exit_code}): {text[-3000:]}")
    return response.result.strip()


def _remote(sandbox, operation: str, **arguments) -> dict:
    marker = "PCA_REMOTE_" + uuid4().hex + ":"
    code = (
        "import json,sys; from pocketcontextagent.run.benchmarks.utils.controller_remote import action; print("
        + repr(marker)
        + "+json.dumps(action(*json.loads(sys.argv[1]))))"
    )
    result = parse_remote_json(
        _exec(
            sandbox,
            [PYTHON, "-c", code, json.dumps([operation, ROOT, arguments])],
            timeout=600 if operation in {"pack", "cleanup", "archive", "import-controller"} else 60,
        ),
        marker,
    )
    if not isinstance(result, dict):
        raise ValueError("Controller response must be an object")
    return result


def _upload(sandbox, data: bytes, target: str) -> None:
    temporary = target + "." + uuid4().hex + ".tmp"
    sandbox.fs.upload_file(data, temporary)
    sandbox.fs.set_file_permissions(temporary, mode="600")
    _exec(sandbox, ["mv", temporary, target])


def _credentials(key: str) -> dict:
    values = {
        "DAYTONA_API_KEY": os.environ["DAYTONA_API_KEY"],
        **provider_environment(key),
        **{name: os.environ[name] for name in ("DAYTONA_API_URL", "DAYTONA_TARGET") if os.getenv(name)},
    }
    if any(any(character in value for character in "\0\r\n") for value in values.values()):
        raise ValueError("Credentials and routing settings must be single-line values")
    return values


def _load(manifest: Path, *, start: bool = False):
    document = json.loads(manifest.read_text())
    api = client()
    sandbox = api.get(document["sandboxes"][0].get("id") or document["sandboxes"][0]["name"])
    if sandbox.labels.get("role") != "trusted-controller" or (
        document.get("organization_id") and document["organization_id"] != sandbox.organization_id
    ):
        raise ValueError("Controller identity/organization differs from the saved manifest")
    if start and getattr(sandbox.state, "value", sandbox.state) in {"stopped", "archived", "paused"}:
        api.start(sandbox, timeout=180)
        sandbox = api.get(sandbox.id)
        _remote(sandbox, "enable")
    if start and getattr(sandbox.state, "value", sandbox.state) == "started":
        sandbox_metadata(sandbox.model_dump(), RuntimeConfig(**document["settings"]))
        if volume := document.get("artifact_volume"):
            if not any(
                item.volume_id == volume["id"] and item.mount_path == volume["mount_path"]
                for item in sandbox.volumes or []
            ):
                raise ValueError("Controller volume differs from its saved identity")
            _remote(sandbox, "volume-info")
    return api, sandbox, document


def _runtime_image(project: Path):
    from daytona import Image

    return (
        Image.base("python:3.11-slim-bookworm")
        .run_commands(
            "apt-get update && apt-get install -y --no-install-recommends git curl openssh-client tmux procps && rm -rf /var/lib/apt/lists/*"
        )
        .pip_install_from_pyproject(str(project / "pyproject.toml"), ["daytona"], extra_options="--no-cache-dir")
        .pip_install(*RUNTIME_REQUIREMENTS, extra_options="--no-cache-dir")
        .workdir(ROOT)
    )


@app.command("create")
def create(
    manifest: Path = typer.Option(DEFAULT_MANIFEST),
    project: Path = typer.Option(Path(".")),
    cpu: int = typer.Option(2, min=1),
    memory: int = typer.Option(8, min=1),
    disk: int = typer.Option(10, min=1),
    ttl_minutes: int = typer.Option(
        0, min=0, help="0 requests no absolute destruction deadline; actual server policy is verified."
    ),
    provider_key: str = typer.Option("DEEPSEEK_API_KEY"),
    volume_name: str | None = typer.Option(
        None, help="Create/reuse a persistent controller-only artifact volume; never deleted with the controller."
    ),
) -> None:
    """Create a persistent trusted controller; no benchmark inference is launched."""
    from daytona import CreateSnapshotParams, Resources

    if manifest.exists():
        raise ValueError("Controller manifest already exists; reuse it or explicitly delete its resources")
    config = ControllerConfig(cpu=cpu, memory=memory, disk=disk, ttl_minutes=ttl_minutes)
    if volume_name is not None:
        validate_volume_name(volume_name)
    api = client()
    credentials = _credentials(provider_key)
    identifier = uuid4().hex[:12]
    document = {
        "version": 1,
        "role": "trusted-controller",
        "state": "creating",
        "root": ROOT,
        "settings": asdict(config),
        "implementation_sha256": implementation_hash(),
        "sandboxes": [{"name": f"pca-trusted-controller-{identifier}"}],
        "snapshots": [{"name": f"pca-controller-runtime-{identifier}"}],
        "jobs": [],
        "volume_requested_name": volume_name,
        "artifact_volume": None,
    }
    write_private_json(manifest, document)
    complete = False
    try:
        if volume_name is not None:
            document["artifact_volume"] = resolve_volume(api, volume_name, create=True)
            write_private_json(manifest, document)
        with tempfile.TemporaryDirectory(prefix="pca-controller-build-") as temporary:
            wheel = build_wheel(project, Path(temporary))
            document["wheel_sha256"] = digest(wheel)
            write_private_json(manifest, document)
            snapshot = api.snapshot.create(
                CreateSnapshotParams(
                    name=document["snapshots"][0]["name"],
                    image=_runtime_image(project),
                    resources=Resources(cpu=cpu, memory=memory, disk=disk),
                ),
                timeout=1800,
            )
            sandbox = api.create(
                sandbox_params(
                    snapshot.name, document["sandboxes"][0]["name"], config, volume=document["artifact_volume"]
                ),
                timeout=180,
            )
            document["sandboxes"][0]["id"] = sandbox.id
            document["organization_id"] = sandbox.organization_id
            document["lifecycle"] = sandbox_metadata(
                api.get(sandbox.id).model_dump(),
                RuntimeConfig(cpu=cpu, memory=memory, disk=disk, ttl_minutes=ttl_minutes),
            )
            write_private_json(manifest, document)
            sandbox.fs.create_folder(ROOT, "700")
            sandbox.fs.set_file_permissions(ROOT, mode="700")
            sandbox.fs.create_folder(ROOT + "/jobs", "700")
            sandbox.fs.create_folder(ROOT + "/credentials", "700")
            remote_wheel = ROOT + "/" + wheel.name
            _upload(sandbox, wheel.read_bytes(), remote_wheel)
            _exec(
                sandbox,
                [PYTHON, "-m", "pip", "install", "--no-deps", "--disable-pip-version-check", remote_wheel],
                timeout=180,
            )
            sandbox.fs.delete_file(remote_wheel)
            _upload(sandbox, json.dumps(credentials).encode(), ROOT + "/credentials.json")
            if document["artifact_volume"] is not None:
                _upload(sandbox, json.dumps(document["artifact_volume"]).encode(), ROOT + "/volume.json")
                _remote(sandbox, "volume-info")
            document["runtime_readiness"] = _remote(sandbox, "adapters")
            document["dependencies"] = json.loads(
                _exec(
                    sandbox,
                    [
                        PYTHON,
                        "-c",
                        "import json,importlib.metadata; print(json.dumps({d.metadata['Name']:d.version for d in importlib.metadata.distributions()}))",
                    ],
                )
            )
        document["state"] = "ready"
        _upload(sandbox, json.dumps(document).encode(), ROOT + "/controller.json")
        write_private_json(manifest, document)
        complete = True
    finally:
        if not complete:
            delete_resources(api, document)
            document["state"] = "failed_cleanup_confirmed"
            write_private_json(manifest, document)
    typer.echo(
        f"Controller ready: {document['sandboxes'][0]['id']}\nManifest: {manifest}\nNo inference launched. Running controller compute is billable until explicitly stopped."
    )


@app.command("check")
def check(
    manifest: Path = typer.Option(DEFAULT_MANIFEST),
    config_spec: list[str] = typer.Option([], "-c", "--config"),
) -> None:
    """Check provider connectivity FROM THE CLOUD without requesting inference."""
    from pocketcontextagent.run.benchmarks.benchmark_matrix import benchmark_config

    config = benchmark_config(config_spec)
    provider_endpoint(config["model"])
    _, sandbox, _ = _load(manifest, start=True)
    key = matrix_provider_key(config)
    _upload(sandbox, json.dumps(_credentials(key)).encode(), ROOT + "/credentials.json")
    typer.echo(json.dumps(_remote(sandbox, "check", model=config["model"], provider_key=key), indent=2))


@app.command("upgrade")
def upgrade(
    manifest: Path = typer.Option(DEFAULT_MANIFEST),
    project: Path = typer.Option(Path(".")),
) -> None:
    """Repair/update an idle controller's frozen code and pinned inference runtime; never rerun jobs."""
    _, sandbox, document = _load(manifest, start=True)
    _remote(sandbox, "maintenance")
    document["maintenance"] = True
    write_private_json(manifest, document)
    complete = False
    try:
        with tempfile.TemporaryDirectory(prefix="pca-controller-upgrade-") as temporary:
            wheel = build_wheel(project, Path(temporary))
            remote_wheel = ROOT + "/" + wheel.name
            _upload(sandbox, wheel.read_bytes(), remote_wheel)
            _exec(
                sandbox,
                [
                    PYTHON,
                    "-m",
                    "pip",
                    "install",
                    "--disable-pip-version-check",
                    "--no-cache-dir",
                    *RUNTIME_REQUIREMENTS,
                ],
                timeout=300,
            )
            _exec(
                sandbox,
                [
                    PYTHON,
                    "-m",
                    "pip",
                    "install",
                    "--disable-pip-version-check",
                    "--force-reinstall",
                    "--no-deps",
                    remote_wheel,
                ],
                timeout=180,
            )
            sandbox.fs.delete_file(remote_wheel)
            readiness = _remote(sandbox, "adapters")
            dependencies = json.loads(
                _exec(
                    sandbox,
                    [
                        PYTHON,
                        "-c",
                        "import json,importlib.metadata; print(json.dumps({d.metadata['Name']:d.version for d in importlib.metadata.distributions()}))",
                    ],
                )
            )
            document.setdefault("upgrades", []).append(
                {
                    "previous_implementation_sha256": document["implementation_sha256"],
                    "previous_dependencies": document["dependencies"],
                }
            )
            document.update(
                implementation_sha256=implementation_hash(),
                wheel_sha256=digest(wheel),
                runtime_readiness=readiness,
                dependencies=dependencies,
                maintenance=False,
                runtime_snapshot_updated=False,
            )
        _upload(sandbox, json.dumps(document).encode(), ROOT + "/controller.json")
        write_private_json(manifest, document)
        _remote(sandbox, "enable")
        complete = True
    finally:
        if not complete:
            document["maintenance"] = True
            write_private_json(manifest, document)
    typer.echo(
        "Controller runtime and frozen code upgraded; adapter imports verified. Existing jobs/results unchanged; no jobs rerun."
    )


def submission_plan(
    data_dir: Path,
    config_spec: list[str],
    limit: int,
    seed: int,
    workers: int,
    query_ids: list[str],
    *,
    benchmark: str = "swebench-verified",
    split: str | None = None,
    method: str | None = None,
    context_window: int | None = None,
) -> tuple[dict, dict]:
    if benchmark == "browsecomp-plus":
        from pocketcontextagent.run.benchmarks.utils.controller_browsecomp import plan

        return plan(
            data_dir,
            config_spec,
            limit,
            seed,
            workers,
            query_ids,
            split=split or "eval150",
            method=method,
            context_window=context_window,
        )
    if method is not None and method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    methods = [method] if method is not None else list(METHODS)
    if benchmark != "swebench-verified" or split not in (None, "test"):
        raise ValueError("Select browsecomp-plus/eval150 or swebench-verified/test")
    from pocketcontextagent.run.benchmarks.benchmark_matrix import (
        benchmark_config,
        load_instances,
        pin_instances,
        run_manifest,
    )

    config = benchmark_config(config_spec, context_window)
    if not 1 <= workers <= 30:
        raise ValueError("Select 1 to 30 workers per method")
    instances = pin_instances(
        load_instances(data_dir, query_ids=query_ids, limit=limit, seed=seed), config.get("task_images")
    )
    run_manifest(config, "react", instances, min(workers, len(instances)), data_dir / "queries.jsonl", "test")
    provider_endpoint(config["model"])
    return config, {
        "benchmark": "swebench-verified",
        "methods": methods,
        "queries_sha256": hashlib.sha256((data_dir / "queries.jsonl").read_bytes()).hexdigest(),
        "selected_query_ids": [row["query_id"] for row in instances],
        "limit": limit,
        "seed": seed,
        "query_ids": query_ids,
        "workers_per_method": workers,
        "maximum_active_coding_sandboxes": min(workers, len(instances)) * len(methods),
        "planned_attempts": len(instances) * len(methods),
        "model": config["model"]["model_name"],
        "context_token_limit": config["agent"]["context_token_limit"],
        "implementation_sha256": implementation_hash(),
    }


def validate_remote_plan(plan: dict, remote: dict) -> None:
    keys = [
        "selected_query_ids",
        "context_token_limit",
        "methods",
        "workers_per_method",
        "attempts",
        "preflight_only",
        "implementation_sha256",
        "queries_sha256",
    ]
    if plan["benchmark"] == "browsecomp-plus":
        keys += [
            "methods",
            "workers_per_method",
            "active_workers_per_method",
            "maximum_active_sandboxes",
            "planned_attempts",
            "attempts",
            "implementation_sha256",
            "queries_sha256",
            "split",
            "model",
            "sandbox_ttl_minutes",
            "preflight_only",
        ]
    if mismatches := [key for key in keys if remote.get(key) != plan[key]]:
        raise ValueError(f"Remote plan differs from the frozen local plan: {', '.join(mismatches)}")


@app.command("submit")
def submit(
    manifest: Path = typer.Option(DEFAULT_MANIFEST),
    data_dir: Path | None = typer.Option(None, help="Prepared data; defaults to data/BENCHMARK."),
    benchmark: str = typer.Option("swebench-verified", help="swebench-verified or browsecomp-plus."),
    split: str | None = typer.Option(None, help="Defaults to test for SWE or eval150 for BrowseComp."),
    corpus_snapshot: str = typer.Option(
        "pocket-browsecomp-plus-20261022", help="Credential-free corpus runtime snapshot for BrowseComp only."
    ),
    limit: int = typer.Option(1, min=0),
    seed: int = typer.Option(42),
    workers_per_method: int | None = typer.Option(
        None, min=1, max=30, help="Defaults to 6 with --method, otherwise 1."
    ),
    method: str | None = typer.Option(None, help="Isolate one method on this controller."),
    query_ids: list[str] = typer.Option([], "--query-id"),
    config_spec: list[str] = typer.Option([], "-c", "--config"),
    context_window: int | None = typer.Option(None, min=1, help=CONTEXT_WINDOW_HELP),
    min_free_bytes: int = typer.Option(
        1024**3, min=1, help="Minimum free controller-disk bytes before new volume-backed work."
    ),
    preflight: bool = typer.Option(False, help="Infrastructure-only check of selected methods; no model inference."),
    dry_run: bool = typer.Option(False, help="Print a local plan; no API calls, uploads or inference."),
) -> None:
    """Accept one detached matched benchmark job; execution stays on the cloud controller."""
    data_dir = data_dir or Path("data") / benchmark
    workers_per_method = workers_per_method if workers_per_method is not None else (6 if method is not None else 1)
    config, plan = submission_plan(
        data_dir,
        config_spec,
        limit,
        seed,
        workers_per_method,
        query_ids,
        benchmark=benchmark,
        split=split,
        method=method,
        context_window=context_window,
    )
    if benchmark == "browsecomp-plus":
        if config.get("corpus_snapshot") and config["corpus_snapshot"]["name"] != corpus_snapshot:
            raise ValueError("Corpus snapshot name differs from its frozen preparation")
        plan["corpus_snapshot"] = config.get("corpus_snapshot") or {"name": corpus_snapshot}
    plan.update(
        preflight_only=preflight,
        attempts=0 if preflight else plan["planned_attempts"],
        volume_min_free_bytes=min_free_bytes,
    )
    if dry_run:
        typer.echo(yaml.safe_dump(plan, sort_keys=False))
        return
    _, sandbox, document = _load(manifest, start=True)
    if document["implementation_sha256"] != plan["implementation_sha256"]:
        raise ValueError("Controller code differs from this checkout; create a new frozen controller before submitting")
    view = _remote(sandbox, "view")
    if view.get("disabled"):
        raise ValueError("Controller is in maintenance mode; finish its upgrade before submitting")
    if any(row.get("runner_alive") or row.get("child_alive") for row in view["jobs"].values()):
        raise ValueError("Controller already has an active job")
    key = matrix_provider_key(config)
    _upload(sandbox, json.dumps(_credentials(key)).encode(), ROOT + "/credentials.json")
    readiness = _remote(sandbox, "check", model=config["model"], provider_key=key)
    identifier = uuid4().hex[:12]
    remote = f"{ROOT}/jobs/{identifier}"
    record = {"id": identifier, "remote": remote, "state": "staging", "selection": plan, "provider_check": readiness}
    document["jobs"].append(record)
    write_private_json(manifest, document)
    sandbox.fs.create_folder(remote, "700")
    sandbox.fs.create_folder(remote + "/input", "700")
    sandbox.fs.create_folder(remote + "/input/data", "700")
    job_credentials = f"{ROOT}/credentials/{identifier}.json"
    _upload(sandbox, json.dumps(_credentials(key)).encode(), job_credentials)
    if benchmark == "browsecomp-plus":
        from pocketcontextagent.run.benchmarks.browsecomp_plus_daytona import (
            matrix_query_bytes,
            select_matrix_instances,
        )

        _upload(
            sandbox,
            matrix_query_bytes(select_matrix_instances(data_dir / "queries.jsonl", "all", [], 0, None)),
            remote + "/input/data/queries.jsonl",
        )
    else:
        for filename in ("queries.jsonl", "manifest.json", "grading_dataset.jsonl"):
            path = data_dir / filename
            if path.exists():
                _upload(sandbox, path.read_bytes(), remote + "/input/data/" + filename)
    _upload(sandbox, yaml.safe_dump(config).encode(), remote + "/input/config.yaml")
    arguments = [
        PYTHON,
        "-m",
        "pocketcontextagent.run.utilities.mini_extra",
        "benchmark-matrix",
        "matrix",
        "--benchmark",
        "swebench-verified",
        "--data-dir",
        remote + "/input/data",
        "--limit",
        str(limit),
        "--seed",
        str(seed),
        "--workers-per-method",
        str(workers_per_method),
        "-c",
        remote + "/input/config.yaml",
        "--output",
        remote + "/output",
    ]
    if benchmark == "browsecomp-plus":
        with tempfile.TemporaryDirectory(prefix="pca-corpus-worker-package-") as temporary:
            wheel = build_wheel(Path("."), Path(temporary))
            remote_wheel = remote + "/input/" + wheel.name
            _upload(sandbox, wheel.read_bytes(), remote_wheel)
            record["corpus_worker_wheel_sha256"] = digest(wheel)
        arguments = [
            PYTHON,
            "-m",
            "pocketcontextagent.run.benchmarks.utils.controller_browsecomp",
            "--data-dir",
            remote + "/input/data",
            "--limit",
            str(limit),
            "--seed",
            str(seed),
            "--workers-per-method",
            str(workers_per_method),
            "--split",
            split or "eval150",
            "--snapshot",
            corpus_snapshot,
            "--wheel",
            remote_wheel,
            "-c",
            remote + "/input/config.yaml",
            "--output",
            remote + "/output",
        ]
    if method is not None:
        arguments.extend(["--method", method])
    for identifier_value in query_ids:
        arguments.extend(["--query-id", identifier_value])
    if preflight:
        arguments.append("--preflight")
    remote_plan = yaml.safe_load(_exec(sandbox, [*arguments, "--dry-run"]))
    validate_remote_plan(plan, remote_plan)
    _upload(sandbox, json.dumps(plan).encode(), remote + "/plan.json")
    _upload(
        sandbox,
        json.dumps(
            {
                "argv": arguments,
                "cwd": remote,
                "secrets_path": job_credentials,
                "env": {
                    "MSWEA_CONTROLLER_ID": sandbox.id,
                    "MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT": "2",
                    "PYTHONUNBUFFERED": "1",
                    "LITELLM_LOCAL_MODEL_COST_MAP": "True",
                    **({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"} if benchmark == "browsecomp-plus" else {}),
                    **(
                        archive_environment(document["artifact_volume"], identifier, min_free_bytes)
                        | {"MSWEA_ARTIFACT_VOLUME_ID": document["artifact_volume"]["id"]}
                        if document.get("artifact_volume")
                        else {}
                    ),
                },
            }
        ).encode(),
        remote + "/request.json",
    )
    record.update(_remote(sandbox, "launch", job=identifier), state="submitted")
    write_private_json(manifest, document)
    _upload(sandbox, json.dumps(document).encode(), ROOT + "/controller.json")
    if record["acceptance"]["state"] in {"staged", "starting"} or record["acceptance"].get("error_type"):
        raise ValueError(
            f"Job {identifier} acceptance is pending or failed; inspect status, do not resubmit automatically"
        )
    typer.echo(
        f"Job submitted: {identifier}\nController: {sandbox.id}\nRemote directory: {remote}\nSafe to disconnect: execution and cleanup are cloud-side, not laptop polling."
    )


@app.command("status")
def status(manifest: Path = typer.Option(DEFAULT_MANIFEST), job: str | None = typer.Option(None)) -> None:
    """Read controller state and durable job status; does not start a stopped controller."""
    _, sandbox, document = _load(manifest)
    result = {
        "controller_id": sandbox.id,
        "controller_state": str(sandbox.state),
        "state": str(sandbox.state),
        "auto_destroy_at": str(sandbox.auto_destroy_at),
    }
    if getattr(sandbox.state, "value", sandbox.state) == "started":
        result.update(_remote(sandbox, "status", job=job) if job else _remote(sandbox, "view"))
    else:
        result["saved_jobs"] = document["jobs"]
    typer.echo(json.dumps(result, indent=2))


@app.command("logs")
def logs(
    manifest: Path = typer.Option(DEFAULT_MANIFEST),
    job: str = typer.Option(...),
    lines: int = typer.Option(100, min=1, max=10000),
) -> None:
    """Read the redacted cloud log; use SSH and tail -f for continuous following."""
    _, sandbox, _ = _load(manifest)
    typer.echo(_remote(sandbox, "logs", job=job, lines=lines)["log"], nl=False)


@app.command("ssh")
def ssh(
    manifest: Path = typer.Option(DEFAULT_MANIFEST),
    command: str | None = typer.Option(None),
    expires_minutes: int = typer.Option(60, min=1),
) -> None:
    """Attach with a fresh native SSH credential; no API keys/tokens are printed or saved."""
    _, sandbox, _ = _load(manifest, start=True)
    access = sandbox.create_ssh_access(expires_in_minutes=expires_minutes)
    arguments = shlex.split(access.ssh_command)
    if not arguments or arguments[0] != "ssh":
        raise ValueError("Unexpected SSH command from Daytona")
    try:
        result = subprocess.run([*arguments, *([command] if command else [])], check=False)
    finally:
        sandbox.revoke_ssh_access(access.token)
    if result.returncode:
        raise typer.Exit(result.returncode)


@app.command("fetch")
def fetch(
    manifest: Path = typer.Option(DEFAULT_MANIFEST),
    job: str = typer.Option(...),
    output: Path = typer.Option(...),
    max_bytes: int = typer.Option(2_000_000_000, min=1),
) -> None:
    """Download verified outputs/log/status only, never controller credentials or private grading inputs."""
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a fresh output directory")
    _, sandbox, _ = _load(manifest, start=True)
    packed = _remote(sandbox, "pack", job=job)
    if packed["bytes"] > max_bytes:
        raise ValueError("Archive exceeds download budget; increase --max-bytes or collect individual files over SSH")
    with tempfile.TemporaryDirectory(prefix="pca-controller-fetch-") as temporary:
        archive = Path(temporary) / "artifacts.tar.gz"
        sandbox.fs.download_file(packed["path"], str(archive))
        if digest(archive) != packed["sha256"]:
            raise ValueError("Controller artifact checksum mismatch")
        extract_artifacts(archive, output)
    sandbox.fs.delete_file(packed["path"])
    typer.echo(f"Collected unchanged run artifacts: {output}")


@app.command("cancel")
def cancel(manifest: Path = typer.Option(DEFAULT_MANIFEST), job: str = typer.Option(...)) -> None:
    """Stop an owned matrix process group, retaining uncollected workers for explicit recovery."""
    _, sandbox, _ = _load(manifest)
    typer.echo(json.dumps(_remote(sandbox, "cancel", job=job), indent=2))
    typer.echo(
        "Inspect worker journals before cleanup; cancellation does not fabricate patches or silently discard worker state."
    )


@app.command("cleanup")
def cleanup(
    manifest: Path = typer.Option(DEFAULT_MANIFEST),
    job: str = typer.Option(...),
    discard_uncollected: bool = typer.Option(False),
) -> None:
    """Reconcile only a stopped job's owned workers; reject uncollected state unless explicitly discarded."""
    _, sandbox, _ = _load(manifest, start=True)
    typer.echo(json.dumps(_remote(sandbox, "cleanup", job=job, discard=discard_uncollected), indent=2))


@app.command("volume-info")
def volume_info(manifest: Path = typer.Option(DEFAULT_MANIFEST)) -> None:
    """Verify the actual controller-only FUSE mount without reading private data."""
    _, sandbox, _ = _load(manifest, start=True)
    typer.echo(json.dumps(_remote(sandbox, "volume-info"), indent=2))


@app.command("archive")
def archive_job(manifest: Path = typer.Option(DEFAULT_MANIFEST), job: str = typer.Option(...)) -> None:
    """Archive an inactive legacy/recovered job; never rerun its agent tasks."""
    _, sandbox, _ = _load(manifest, start=True)
    typer.echo(json.dumps(_remote(sandbox, "archive", job=job), indent=2))


@app.command("import-results")
def import_results(
    source_controller: str = typer.Option(...),
    job: str = typer.Option(...),
    manifest: Path = typer.Option(DEFAULT_MANIFEST),
    max_bytes: int = typer.Option(2_000_000_000, min=1),
) -> None:
    """Copy completed public results cloud-to-cloud into the volume; no laptop data relay or inference."""
    api, sandbox, document = _load(manifest, start=True)
    if not document.get("artifact_volume"):
        raise ValueError("Create a volume-backed target controller first")
    source = api.get(source_controller)
    if source.labels.get("role") != "trusted-controller" or source.organization_id != sandbox.organization_id:
        raise ValueError("Source must be a trusted controller in the same organization")
    if getattr(source.state, "value", source.state) in {"stopped", "archived", "paused"}:
        api.start(source, timeout=180)
    typer.echo(
        json.dumps(
            _remote(sandbox, "import-controller", source_controller=source.id, job=job, max_bytes=max_bytes), indent=2
        )
    )


@app.command("volume-delete")
def volume_delete(name: str = typer.Option(...), confirm_destroy_data: bool = typer.Option(False)) -> None:
    """Explicit, irreversible deletion of an unmounted volume; never part of controller cleanup."""
    if not confirm_destroy_data:
        raise ValueError("Volume data is independent: explicitly pass --confirm-destroy-data after backup")
    validate_volume_name(name)
    api = client()
    volume = api.volume.get(name, create=False)
    api.volume.delete(volume)
    deadline = time.monotonic() + 180
    while any(row.id == volume.id for row in api.volume.list()):
        if time.monotonic() >= deadline:
            raise TimeoutError("Volume deletion was not confirmed")
        time.sleep(1)
    typer.echo("Volume deletion confirmed; archived data cannot be recovered")


@app.command("start")
def start(manifest: Path = typer.Option(DEFAULT_MANIFEST)) -> None:
    """Start the existing filesystem; interrupted jobs are never automatically rerun."""
    _, sandbox, _ = _load(manifest, start=True)
    _remote(sandbox, "enable")
    typer.echo(f"Controller started: {sandbox.id}; no jobs rerun")


@app.command("stop")
def stop(manifest: Path = typer.Option(DEFAULT_MANIFEST)) -> None:
    """Stop idle controller compute, preserving disk; refuses an active job."""
    api, sandbox, _ = _load(manifest)
    if getattr(sandbox.state, "value", sandbox.state) in {"stopped", "archived"}:
        typer.echo("Controller already stopped; disk retained")
        return
    _remote(sandbox, "maintenance")
    api.stop(sandbox, timeout=180)
    typer.echo("Controller stopped; disk retained. No job can continue across a controller stop.")


@app.command("delete")
def delete(manifest: Path = typer.Option(DEFAULT_MANIFEST), discard_uncollected: bool = typer.Option(False)) -> None:
    """Confirm deletion of controller/owned snapshot after reconciling worker journals; active jobs block deletion."""
    document = json.loads(manifest.read_text())
    if document["state"] == "ready":
        api, sandbox, _ = _load(manifest, start=True)
        _remote(sandbox, "maintenance")
        for job, state in _remote(sandbox, "view")["jobs"].items():
            if state.get("archived_only"):
                continue
            journal = _remote(sandbox, "journal", job=job)
            write_private_json(manifest.parent / "controller-recovery" / job / "sandbox-cleanup.json", journal)
            if document.get("artifact_volume"):
                _remote(sandbox, "archive", job=job, allow_maintenance=True)
            _remote(sandbox, "cleanup", job=job, discard=discard_uncollected)
    else:
        api = client()
    delete_resources(api, document)
    document["state"] = "deleted_cleanup_confirmed"
    write_private_json(manifest, document)
    typer.echo(
        "Controller and owned runtime snapshot deletion confirmed; public recovery manifest retained. Any artifact volume is preserved."
    )
