"""Operations executed on the trusted controller, never in coding sandboxes."""

import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from pocketcontextagent.run.benchmarks.utils.controller_job import launch, read_status
from pocketcontextagent.run.benchmarks.utils.controller_setup import provider_endpoint, write_private_json


def job_path(root: Path, identifier: str) -> Path:
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", identifier):
        raise ValueError("Invalid job identifier")
    return root / "jobs" / identifier


def _status(job: Path) -> dict:
    return read_status(job) if (job / "status.json").is_file() else {"state": "staged"}


def _active(status: dict) -> bool:
    return status.get("child_alive", False) or status.get("runner_alive", False)


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


def _cleanup(root: Path, job: Path, discard: bool) -> dict:
    from pocketcontextagent.run.benchmarks.browsecomp_plus_daytona import client
    from pocketcontextagent.run.benchmarks.utils.swebench_daytona import delete_resources

    if _active(_status(job)):
        raise ValueError("Cancel the job before reconciling its workers")
    path = job / "output" / "sandbox-cleanup.json"
    if not path.exists():
        return {"confirmed": True, "journal": False}
    document = json.loads(path.read_text())
    for row in document["sandboxes"]:
        task = job / "output" / row["method"] / "tasks" / row["query_id"]
        if (
            not discard
            and not row.get("cleanup_confirmed")
            and (not (task / f"{row['query_id']}.json").is_file() or not (task / "trajectories.tar.gz").is_file())
        ):
            raise ValueError("Uncollected worker state: recover it first, or explicitly discard uncollected work")
    os.environ.update({key: value for key, value in _credentials(root).items() if key.startswith("DAYTONA_")})
    delete_resources(client(), document)
    result = {"confirmed": True, "journal": True, "discard_uncollected": discard}
    write_private_json(job / "controller-cleanup.json", result)
    return result


def action(name: str, root_path: str, arguments: dict | None = None) -> dict:
    root, args = Path(root_path), arguments or {}
    if not root.is_absolute():
        raise ValueError("Controller root must be absolute")
    jobs = root / "jobs"
    if name == "view":
        return {"jobs": {path.name: _status(path) for path in sorted(jobs.iterdir()) if path.is_dir()}}
    if name == "check":
        key = args["provider_key"]
        if key not in {"DEEPSEEK_API_KEY", "META_API_KEY", "OPENROUTER_API_KEY"}:
            raise ValueError("Select a model provider credential")
        endpoint = provider_endpoint(args["model"])
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
    job = job_path(root, args["job"])
    if name == "launch":
        pid = launch(job / "request.json")
        deadline = time.monotonic() + 30
        while _status(job)["state"] in {"staged", "starting"} and time.monotonic() < deadline:
            time.sleep(0.1)
        status = _status(job)
        return {"runner_pid": pid, "acceptance": status}
    if name == "status":
        return _status(job)
    if name == "logs":
        path = job / "run.log"
        text = (
            subprocess.run(
                ["tail", "-n", str(args.get("lines", 100)), str(path)], capture_output=True, text=True, check=True
            ).stdout
            if path.exists()
            else ""
        )
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
        path = job / "output" / "sandbox-cleanup.json"
        return json.loads(path.read_text()) if path.exists() else {}
    if name == "cleanup":
        return _cleanup(root, job, args.get("discard", False))
    if name == "pack":
        if _active(_status(job)):
            raise ValueError("Wait for completion or cancel before collecting a consistent archive")
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
