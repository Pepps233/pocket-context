"""Digest-pinned snapshot provisioning and owned-resource cleanup for SWE-bench."""

import json
import os
import re
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import requests
import typer


@dataclass(frozen=True)
class RuntimeConfig:
    cpu: int = 2
    memory: int = 8
    disk: int = 10
    ttl_minutes: int = 240
    snapshot_timeout_seconds: int = 900
    startup_timeout_seconds: int = 180
    cleanup_timeout_seconds: int = 180
    snapshot_build_workers: int = 2

    def __post_init__(self) -> None:
        if any(type(value) is not int or value <= 0 for key, value in asdict(self).items() if key != "ttl_minutes"):
            raise ValueError("Sandbox resources and timeouts must be positive integers")
        if type(self.ttl_minutes) is not int or self.ttl_minutes < 0:
            raise ValueError("ttl_minutes must be nonnegative; 0 explicitly disables the absolute safety limit")
        if self.snapshot_build_workers > 10:
            raise ValueError("Use at most 10 concurrent snapshot imports")

    def preflight_timeout(self, command_timeout: int) -> int:
        if type(command_timeout) is not int or command_timeout <= 0:
            raise ValueError("Command timeout must be a positive integer")
        return self.startup_timeout_seconds + command_timeout * 8 + 60


@dataclass(frozen=True)
class PreparedSnapshot:
    name: str
    image: str


def immutable_image(image: str) -> str:
    match = re.fullmatch(r"docker\.io/(swebench/[a-z0-9_.-]+)(?::latest|@(sha256:[a-f0-9]{64}))", image)
    if not match:
        raise ValueError("Use a canonical official SWE-bench image, not an arbitrary registry")
    if match[2]:
        return image
    repository = match[1]
    auth = requests.get(
        "https://auth.docker.io/token",
        params={"service": "registry.docker.io", "scope": f"repository:{repository}:pull"},
        timeout=30,
    )
    auth.raise_for_status()
    manifest = requests.head(
        f"https://registry-1.docker.io/v2/{repository}/manifests/latest",
        headers={
            "Authorization": "Bearer " + auth.json()["token"],
            "Accept": "application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json, application/vnd.docker.distribution.manifest.list.v2+json",
        },
        timeout=30,
    )
    manifest.raise_for_status()
    digest = manifest.headers.get("Docker-Content-Digest", "")
    if not re.fullmatch(r"sha256:[a-f0-9]{64}", digest):
        raise ValueError("Registry did not return an immutable manifest digest")
    return f"docker.io/{repository}@{digest}"


class ResourceJournal:
    def __init__(self, path: Path):
        self.path = path
        self.records: dict[str, dict[str, dict]] = {"sandboxes": {}, "snapshots": {}}
        self.lock = threading.Lock()

    def record(self, kind: str, name: str, **fields) -> None:
        with self.lock:
            self.records[kind].setdefault(name, {"name": name}).update(fields)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(self.path.suffix + ".tmp")
            temporary.write_text(
                json.dumps({key: list(rows.values()) for key, rows in self.records.items()}, indent=2), encoding="utf-8"
            )
            temporary.replace(self.path)

    def failure(self, kind: str, name: str, error: BaseException) -> dict:
        message = str(error)
        for key, value in os.environ.items():
            if key.endswith("_API_KEY") and value:
                message = message.replace(value, "[REDACTED]")
        diagnostic = {"error_type": type(error).__name__, "error": message}
        self.record(kind, name, phase="failed", **diagnostic)
        typer.echo(f"{name}: {type(error).__name__}: {message}", err=True)
        return diagnostic

    def document(self) -> dict:
        with self.lock:
            return {key: list(rows.values()) for key, rows in self.records.items()}


def snapshots(api) -> list:
    first = api.snapshot.list(page=1, limit=100)
    return first.items + [
        item for page in range(2, first.total_pages + 1) for item in api.snapshot.list(page=page, limit=100).items
    ]


def delete_resources(api, document: dict, timeout: int = 180, *, preserve_retained: bool = False) -> None:
    retained = [row for row in document.get("sandboxes", []) if preserve_retained and row.get("retain")]
    targets = [row for row in document.get("sandboxes", []) if row not in retained]
    ids = {row["id"] for row in targets if row.get("id")}
    names = {row["name"] for row in targets if row.get("name")}

    def live_sandboxes() -> list:
        return [
            row
            for row in api.list()
            if (row.id in ids or row.name in names) and getattr(row.state, "value", row.state) != "destroyed"
        ]

    for sandbox in live_sandboxes():
        if getattr(sandbox.state, "value", sandbox.state) != "destroying":
            api.delete(sandbox, wait=True, timeout=timeout)
    deadline = time.monotonic() + timeout
    while live_sandboxes():
        if time.monotonic() >= deadline:
            raise TimeoutError("Sandbox deletion was not confirmed")
        time.sleep(3)
    snapshot_names = {row["name"] for row in document.get("snapshots", [])} - {row.get("snapshot") for row in retained}
    for snapshot in snapshots(api) if snapshot_names else []:
        if snapshot.name in snapshot_names and getattr(snapshot.state, "value", snapshot.state) != "removing":
            api.snapshot.delete(snapshot.name)
    deadline = time.monotonic() + timeout
    if snapshot_names:
        while snapshot_names & {row.name for row in snapshots(api)}:
            if time.monotonic() >= deadline:
                raise TimeoutError("Snapshot deletion was not confirmed; retry cleanup using the resource journal")
            time.sleep(3)


def create_snapshot(api, config: RuntimeConfig, journal: ResourceJournal, instance: dict) -> PreparedSnapshot:
    from daytona import CreateSnapshotParams, Resources

    name = f"pca-swe-snapshot-{uuid4().hex[:16]}"
    journal.record(
        "snapshots", name, phase="resolving", requested_image=instance["image"], query_id=instance["query_id"]
    )
    success = False
    try:
        image = immutable_image(instance["image"])
        journal.record("snapshots", name, phase="importing", image=image)
        typer.echo(f"{instance['query_id']}: importing shared snapshot {name}")
        snapshot = api.snapshot.create(
            CreateSnapshotParams(
                name=name, image=image, resources=Resources(cpu=config.cpu, memory=config.memory, disk=config.disk)
            ),
            timeout=config.snapshot_timeout_seconds,
        )
        if getattr(snapshot.state, "value", snapshot.state) != "active" or (
            snapshot.cpu,
            snapshot.mem,
            snapshot.disk,
        ) != (config.cpu, config.memory, config.disk):
            raise RuntimeError("Snapshot is not active with the requested resources")
        journal.record("snapshots", name, phase="active", image=image)
        success = True
        return PreparedSnapshot(name, image)
    finally:
        if not success and sys.exc_info()[1] is not None:
            journal.failure("snapshots", name, sys.exc_info()[1])


def sandbox_params(snapshot: PreparedSnapshot, method: str, config: RuntimeConfig, *, name: str | None = None):
    from daytona import CreateSandboxFromSnapshotParams

    return CreateSandboxFromSnapshotParams(
        snapshot=snapshot.name,
        name=name or f"pca-swe-{method.replace('_', '-')}-{uuid4().hex[:12]}",
        os_user="root",
        public=False,
        network_block_all=True,
        ephemeral=False,
        env_vars={"BASH_ENV": "/root/.bashrc", "PAGER": "cat", "GIT_TERMINAL_PROMPT": "0"},
        labels={"project": "pocket-context", "benchmark": "swebench-verified", "method": method},
        auto_stop_interval=0,
        auto_pause_interval=0,
        auto_delete_interval=-1,
        ttl_minutes=config.ttl_minutes,
    )


def sandbox_metadata(sandbox: dict, config: RuntimeConfig) -> dict:
    if tuple(sandbox[key] for key in ("cpu", "memory", "disk")) != (config.cpu, config.memory, config.disk):
        raise RuntimeError("Sandbox did not inherit the snapshot's requested resources")
    if tuple(sandbox[key] for key in ("auto_stop_interval", "auto_pause_interval", "auto_delete_interval")) != (
        0,
        0,
        -1,
    ):
        raise RuntimeError("Sandbox lifecycle policy differs from the requested persistent, no-idle-stop policy")
    if config.ttl_minutes:
        if sandbox["auto_destroy_at"] is None:
            raise RuntimeError("Sandbox absolute safety lifetime was not applied")
        deadline = datetime.fromisoformat(str(sandbox["auto_destroy_at"]).replace("Z", "+00:00"))
        if deadline.tzinfo is None:
            raise RuntimeError("Sandbox lifetime metadata lacks a timezone")
        remaining = (deadline - datetime.now(timezone.utc)).total_seconds()
        if remaining < config.ttl_minutes * 60 - config.startup_timeout_seconds - 60:
            raise RuntimeError("Sandbox absolute lifetime is shorter than requested")
    elif sandbox["auto_destroy_at"] is not None:
        raise RuntimeError("Organization policy imposes a lifetime despite ttl_minutes=0")
    return {
        key: str(sandbox[key]) if key == "auto_destroy_at" and sandbox[key] is not None else sandbox[key]
        for key in (
            "cpu",
            "memory",
            "disk",
            "auto_stop_interval",
            "auto_pause_interval",
            "auto_delete_interval",
            "auto_destroy_at",
        )
    }
