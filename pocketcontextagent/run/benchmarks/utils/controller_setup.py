"""Local setup helpers for the trusted Daytona controller; no network calls."""

import json
import os
import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from daytona import CreateSandboxFromSnapshotParams


@dataclass(frozen=True)
class ControllerConfig:
    cpu: int = 2
    memory: int = 8
    disk: int = 10
    ttl_minutes: int = 0

    def __post_init__(self) -> None:
        if any(type(value) is not int or value <= 0 for value in (self.cpu, self.memory, self.disk)):
            raise ValueError("Controller resources must be positive integers")
        if type(self.ttl_minutes) is not int or self.ttl_minutes < 0:
            raise ValueError("ttl_minutes must be a nonnegative integer; 0 disables the absolute safety limit")


def sandbox_params(
    snapshot: str, name: str, config: ControllerConfig, *, volume: dict | None = None
) -> "CreateSandboxFromSnapshotParams":
    from daytona import CreateSandboxFromSnapshotParams

    from pocketcontextagent.run.benchmarks.utils.controller_volumes import volume_mount

    return CreateSandboxFromSnapshotParams(
        snapshot=snapshot,
        name=name,
        volumes=[volume_mount(volume)] if volume is not None else None,
        os_user="root",
        public=False,
        network_block_all=False,
        ephemeral=False,
        auto_stop_interval=0,
        auto_pause_interval=0,
        auto_delete_interval=-1,
        ttl_minutes=config.ttl_minutes,
        labels={"project": "pocket-context", "role": "trusted-controller"},
        env_vars={
            "MSWEA_SILENT_STARTUP": "1",
            "MSWEA_GLOBAL_CONFIG_DIR": "/workspace/pocket-controller/config",
            "PYTHONUNBUFFERED": "1",
        },
    )


def write_private_json(path: Path, value: dict) -> None:
    """Atomically replace private JSON; leave existing parent permissions alone."""
    contents = json.dumps(value, indent=2).encode("utf-8")
    for parent in reversed(path.parents):
        parent.mkdir(mode=0o700, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def provider_endpoint(model_config: dict) -> str:
    """Return an official authenticated readiness URL, never a custom host/path."""
    provider = str(model_config.get("model_name", "")).partition("/")[0]
    if model_config.get("model_class") in {"openrouter", "openrouter_textbased", "openrouter_response"}:
        provider = "openrouter"
    bases = {
        "deepseek": "https://api.deepseek.com",
        "meta": "https://api.meta.ai/v1",
        "openrouter": "https://openrouter.ai/api/v1",
    }
    if provider not in bases:
        raise ValueError("Unsupported provider: use deepseek/..., meta/..., or openrouter/...")
    official = urlsplit(bases[provider])
    for base in (model_config.get("api_base"), model_config.get("model_kwargs", {}).get("api_base")):
        if base is None:
            continue
        if not isinstance(base, str) or any(char.isspace() for char in base) or "?" in base or "#" in base:
            raise ValueError(
                "api_base must be the official HTTPS provider base without credentials, query, or fragment"
            )
        parsed = urlsplit(base)
        if (
            parsed.scheme != "https"
            or parsed.netloc.lower() != official.netloc
            or parsed.path not in {official.path, official.path + "/"}
        ):
            raise ValueError("Custom api_base is unsupported; use the official HTTPS provider host/path without a port")
    return bases[provider] + ("/auth/key" if provider == "openrouter" else "/models")


def extract_artifacts(archive: Path, destination: Path) -> None:
    """Validate all tar members before extracting bytes into a fresh directory.

    Links, special files, duplicate normalized paths, and file/directory conflicts
    are rejected. Archive permissions/ownership are intentionally not restored.
    """
    if destination.is_symlink() or (destination.exists() and (not destination.is_dir() or any(destination.iterdir()))):
        raise ValueError("Artifact destination must be a nonexistent or empty directory")
    root = destination.resolve()
    with tarfile.open(archive, "r:*") as bundle:
        members: dict[PurePosixPath, tarfile.TarInfo] = {}
        for member in bundle.getmembers():
            relative = PurePosixPath(member.name)
            if (
                not member.name
                or "\0" in member.name
                or "\\" in member.name
                or PureWindowsPath(member.name).drive
                or relative.is_absolute()
                or ".." in relative.parts
                or not (root / relative).resolve().is_relative_to(root)
                or not (member.isfile() or member.isdir())
                or (relative == PurePosixPath(".") and not member.isdir())
            ):
                raise ValueError("Unsafe artifact archive member")
            if relative in members:
                raise ValueError("Duplicate artifact archive member")
            members[relative] = member
        for relative in members:
            if any(parent in members and not members[parent].isdir() for parent in relative.parents):
                raise ValueError("Artifact archive file/directory conflict")
        destination.mkdir(parents=True, exist_ok=True)
        for relative, member in members.items():
            target = destination / relative
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.extractfile(member) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)
