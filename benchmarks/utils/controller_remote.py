"""Operations executed on the trusted controller, never in coding sandboxes."""

import fcntl
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import requests

from benchmarks.utils.controller_artifacts import ArtifactStore, require_free_space
from benchmarks.utils.controller_job import launch, read_status
from benchmarks.utils.controller_runtime import check_runtime
from benchmarks.utils.controller_setup import (
    extract_artifacts,
    provider_endpoint,
    write_private_json,
)
from benchmarks.utils.controller_volumes import verify_volume_mount, volume_mount


def job_path(root: Path, identifier: str) -> Path:
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", identifier):
        raise ValueError("Invalid job identifier")
    return root / "jobs" / identifier


def _store(root: Path) -> ArtifactStore | None:
    path = root / "volume.json"
    if not path.is_file():
        return None
    record = json.loads(path.read_text())
    volume_mount(record)
    verify_volume_mount(Path(record["mount_path"]))
    return ArtifactStore(Path(record["store_path"]))


def _status(job: Path) -> dict:
    if (job / "status.json").is_file():
        return read_status(job)
    store = _store(job.parent.parent)
    return (store.job_status(job.name) if store else None) or {"state": "staged"}


def _active(status: dict) -> bool:
    return status.get("child_alive", False) or status.get("runner_alive", False)


@contextmanager
def _archive_lock(root: Path, *, allow_maintenance: bool = False):
    with (root / "jobs" / "active.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "jobs" / "disabled").exists() and not allow_maintenance:
            raise ValueError("Controller is in maintenance; archive admission is disabled")
        yield


def _credentials(root: Path) -> dict:
    values = json.loads((root / "credentials.json").read_text())
    if not isinstance(values, dict) or any(
        not isinstance(value, str) or not value or any(character in value for character in "\0\r\n")
        for value in values.values()
    ):
        raise ValueError("Invalid controller credential file")
    return values


def _redact(text: str, root: Path) -> str:
    for key, value in _credentials(root).items():
        if key.endswith("_API_KEY"):
            text = text.replace(value, "[REDACTED]")
    return text


def _journal(root: Path, job: Path) -> dict:
    path = job / "output" / "sandbox-cleanup.json"
    if path.is_file():
        return json.loads(path.read_text())
    store = _store(root)
    data = store.read_job_file(job.name, "output/sandbox-cleanup.json") if store else None
    return json.loads(data) if data else {}


def _cleanup(root: Path, job: Path, discard: bool) -> dict:
    from benchmarks.browsecomp_plus_daytona import client
    from benchmarks.utils.swebench_daytona import delete_resources

    if _active(_status(job)):
        raise ValueError("Cancel the job before reconciling its workers")
    document = _journal(root, job)
    if not document:
        return {"confirmed": True, "journal": False}
    store = _store(root)
    if store is not None and not discard:
        store.verify_job(job.name, local=job)
    for row in document["sandboxes"]:
        if discard or row.get("cleanup_confirmed"):
            continue
        task = job / "output" / row["method"] / "tasks" / row["query_id"]
        available = (task / f"{row['query_id']}.json").is_file() and (task / "trajectories.tar.gz").is_file()
        if not available and store:
            available = store.task_receipt(job.name, row["method"], row["query_id"]) is not None
        if not available:
            raise ValueError("Uncollected worker state: recover it first, or explicitly discard uncollected work")
    os.environ.update({key: value for key, value in _credentials(root).items() if key.startswith("DAYTONA_")})
    delete_resources(client(), document)
    result = {"confirmed": True, "journal": True, "discard_uncollected": discard}
    if job.is_dir():
        write_private_json(job / "controller-cleanup.json", result)
    return result


def archive_directory(
    store: ArtifactStore, job: Path, identifier: str, status: dict, *, imported: bool = False
) -> dict:
    """Import inactive legacy results without changing benchmark records or inputs."""
    if status.get("state") == "archive_failed":
        status = status | {
            "state": status["workload_state"],
            "archive_state": "complete",
            "recovered_archive_failure": True,
        }
    if _active(status) or status.get("state") not in {"completed", "failed", "cancelled", "interrupted"}:
        raise ValueError("Only inactive terminal jobs may be archived")
    existing = store.job_status(identifier)
    if existing is not None:
        store.verify_job(identifier, local=job)
        return existing
    for task in sorted((job / "output").glob("*/tasks/*")):
        if (task / f"{task.name}.json").is_file() and (task / "trajectories.tar.gz").is_file():
            if imported:
                (task / "volume-artifacts.json").unlink(missing_ok=True)
            store.archive_task(task, identifier, task.parent.parent.name, task.name, prune=True)
    store.finalize_job(job, identifier, status)
    return store.job_status(identifier)


def _import_controller(root: Path, store: ArtifactStore, args: dict) -> dict:
    from benchmarks.browsecomp_plus_daytona import client

    identifier = args["job"]
    job_path(root, identifier)
    if store.job_status(identifier) is not None:
        store.verify_job(identifier)
        return store.job_status(identifier)
    os.environ.update({key: value for key, value in _credentials(root).items() if key.startswith("DAYTONA_")})
    source = client().get(args["source_controller"])
    own = json.loads((root / "controller.json").read_text())
    if source.labels.get("role") != "trusted-controller" or source.organization_id != own["organization_id"]:
        raise ValueError("Import source must be a trusted controller in the same organization")
    if source.id == own["sandboxes"][0]["id"]:
        raise ValueError("Use archive for a job already on this controller")
    code = "import json,sys; from pocketcontextagent.run.benchmarks.utils.controller_remote import action; print(json.dumps(action(*json.loads(sys.argv[1]))))"
    response = source.process.exec(
        shlex.join(["/usr/local/bin/python", "-c", code, json.dumps(["pack", str(root), {"job": identifier}])]),
        cwd=str(root),
        timeout=240,
    )
    if response.exit_code:
        raise RuntimeError("Source controller could not produce an inactive public artifact archive")
    packed = json.loads(response.result)
    if packed["bytes"] > args.get("max_bytes", 2_000_000_000):
        raise ValueError("Legacy import exceeds its bounded staging budget")
    require_free_space(root, packed["bytes"] + 1024**3)
    with tempfile.TemporaryDirectory(prefix="volume-import-", dir=root) as temporary:
        archive = Path(temporary) / "source.tar.gz"
        source.fs.download_file(packed["path"], str(archive))
        digest = hashlib.sha256()
        with archive.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != packed["sha256"]:
            raise ValueError("Source-controller archive checksum mismatch")
        with tarfile.open(archive) as bundle:
            require_free_space(root, sum(member.size for member in bundle.getmembers()) + 1024**3)
        imported = Path(temporary) / "job"
        extract_artifacts(archive, imported)
        result = archive_directory(
            store, imported, identifier, json.loads((imported / "status.json").read_text()), imported=True
        )
    source.fs.delete_file(packed["path"])
    return result


def action(name: str, root_path: str, arguments: dict | None = None) -> dict:
    root, args = Path(root_path), arguments or {}
    if not root.is_absolute():
        raise ValueError("Controller root must be absolute")
    jobs = root / "jobs"
    if name == "view":
        store = _store(root)
        archived = store.list_jobs() if store else {}
        return {
            "disabled": (jobs / "disabled").exists(),
            "jobs": archived | {path.name: _status(path) for path in sorted(jobs.iterdir()) if path.is_dir()},
        }
    if name == "volume-info":
        store = _store(root)
        return {
            "enabled": store is not None,
            "volume": json.loads((root / "volume.json").read_text()) if store else None,
        }
    if name == "adapters":
        return check_runtime()
    if name == "check":
        key = args["provider_key"]
        if key not in {"DEEPSEEK_API_KEY", "META_API_KEY", "OPENROUTER_API_KEY"}:
            raise ValueError("Select a model provider credential")
        endpoint = provider_endpoint(args["model"])
        readiness = check_runtime()
        if (root / "volume.json").exists():
            _store(root)
        response = requests.get(
            endpoint,
            headers={"Authorization": "Bearer " + _credentials(root)[key]},
            timeout=(10, 20),
            allow_redirects=False,
        )
        if response.status_code != 200:
            raise RuntimeError(f"Provider metadata check failed: HTTP {response.status_code}; no inference requested")
        response.json()
        return {
            "endpoint": endpoint,
            "http_status": 200,
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "inference_calls": 0,
            "adapter_readiness": readiness,
        }
    if name == "maintenance":
        with (jobs / "active.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if any(_active(_status(path)) for path in jobs.iterdir() if path.is_dir()):
                raise ValueError("Controller still owns an active runner or orphaned child")
            (jobs / "disabled").touch(mode=0o600)
        return {"disabled": True}
    if name == "enable":
        (jobs / "disabled").unlink(missing_ok=True)
        return {"disabled": False}
    if name == "import-controller":
        store = _store(root)
        if store is None:
            raise ValueError("Configure a mounted artifact volume before importing")
        with _archive_lock(root):
            return _import_controller(root, store, args)
    job = job_path(root, args["job"])
    if name == "launch":
        pid = launch(job / "request.json")
        deadline = time.monotonic() + 30
        while _status(job)["state"] in {"staged", "starting"} and time.monotonic() < deadline:
            time.sleep(0.1)
        return {"runner_pid": pid, "acceptance": _status(job)}
    if name == "status":
        return _status(job)
    if name == "logs":
        path = job / "run.log"
        if path.exists():
            text = subprocess.run(
                ["tail", "-n", str(args.get("lines", 100)), str(path)], capture_output=True, text=True, check=True
            ).stdout
        else:
            store = _store(root)
            data = store.read_job_file(job.name, "run.log") if store else None
            text = "\n".join((data or b"").decode().splitlines()[-args.get("lines", 100) :])
        return {"log": _redact(text, root), "status": _status(job)}
    if name == "cancel":
        status = _status(job)
        if status.get("child_alive"):
            if status["child_pgid"] != status["child_pid"]:
                raise ValueError("Refusing to signal an unowned process group")
            subprocess.run(["/bin/kill", "-TERM", "--", f"-{status['child_pgid']}"], capture_output=True, check=False)
            deadline = time.monotonic() + 20
            while _active(_status(job)) and time.monotonic() < deadline:
                time.sleep(0.1)
        return _status(job)
    if name == "journal":
        return _journal(root, job)
    if name == "cleanup":
        return _cleanup(root, job, args.get("discard", False))
    if name == "archive":
        store = _store(root)
        if store is None:
            raise ValueError("No artifact volume configured")
        with _archive_lock(root, allow_maintenance=args.get("allow_maintenance", False)):
            status = _status(job)
            result = archive_directory(store, job, job.name, status)
            if status.get("state") == "archive_failed":
                write_private_json(
                    job / "status.json",
                    {
                        key: value
                        for key, value in result.items()
                        if key not in {"archived_only", "runner_alive", "child_alive"}
                    },
                )
            return result
    if name == "pack":
        if _active(_status(job)):
            raise ValueError("Wait for completion or cancel before collecting a consistent archive")
        store = _store(root)
        if store is not None:
            if store.job_status(job.name) is None:
                with _archive_lock(root):
                    archive_directory(store, job, job.name, _status(job))
            return store.export_job(job.name, store.root / "exports" / f"{job.name}-{uuid4().hex}.tar.gz")
        output = job / "output"
        size = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
        if shutil.disk_usage(job).free < size + 10_000_000:
            raise ValueError("Insufficient controller disk for a safe archive; collect individual files over SSH")
        temporary = job / "artifacts.tar.gz.tmp"
        with tarfile.open(temporary, "w:gz", dereference=False) as archive:
            if output.exists():
                archive.add(output, arcname="output")
            for filename in ("status.json", "plan.json", "run.log", "controller-cleanup.json"):
                if (job / filename).is_file():
                    archive.add(job / filename, arcname=filename)
        target = job / "artifacts.tar.gz"
        temporary.replace(target)
        digest = hashlib.sha256()
        with target.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return {"path": str(target), "bytes": target.stat().st_size, "sha256": digest.hexdigest()}
    raise ValueError(f"Unknown controller operation: {name}")
