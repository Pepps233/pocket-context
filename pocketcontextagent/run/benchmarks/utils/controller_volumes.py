"""Persistent controller artifact volumes; never fall back to the root disk."""

import re
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from daytona import Daytona, VolumeMount
    from daytona.common.volume import Volume

VOLUME_MOUNT = "/mnt/pocket-context-artifacts"
STORE_ROOT = VOLUME_MOUNT + "/artifacts-v1"


def validate_volume_name(name: str) -> None:
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", name):
        raise ValueError("Invalid volume name: use 1-64 ASCII letters, digits, underscores or hyphens")


def _validate_record(record: dict) -> None:
    if not isinstance(record, dict) or set(record) != {"id", "name", "mount_path", "store_path", "created"}:
        raise ValueError("Invalid volume record fields")
    validate_volume_name(record["name"])
    if not isinstance(record["id"], str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", record["id"]):
        raise ValueError("Invalid volume id")
    if record["mount_path"] != VOLUME_MOUNT or record["store_path"] != STORE_ROOT:
        raise ValueError("Volume record paths must use the managed artifact mount and store")
    if type(record["created"]) is not bool:
        raise ValueError("Volume record created must be a boolean")


def volume_mount(record: dict) -> "VolumeMount":
    from daytona import VolumeMount

    _validate_record(record)
    return VolumeMount(volume_id=record["id"], mount_path=VOLUME_MOUNT)


def _volume_is_ready(volume: "Volume") -> bool:
    if volume.state == "ready":
        return True
    if volume.state in {"pending_create", "creating"}:
        return False
    raise RuntimeError(f"Volume {volume.id} cannot become ready from state {volume.state}")


def resolve_volume(api: "Daytona", name: str, *, create: bool, timeout: int = 120) -> dict:
    """Resolve once, creating only with permission; caller journals name beforehand.

    Failures leave any created volume intact for explicit reconciliation by name.
    The timeout bounds readiness polling, not individual SDK network requests.
    """
    validate_volume_name(name)
    if type(create) is not bool:
        raise ValueError("create must be a boolean")
    if type(timeout) is not int or timeout <= 0:
        raise ValueError("Volume readiness timeout must be a positive integer")
    matches = [volume for volume in api.volume.list() if volume.name == name]
    if len(matches) > 1:
        raise RuntimeError("Multiple volumes have the requested name")
    if not matches and not create:
        raise ValueError("Volume does not exist; explicitly allow creation")
    created = not matches
    volume = api.volume.create(name) if created else matches[0]
    record = {"id": volume.id, "name": name, "mount_path": VOLUME_MOUNT, "store_path": STORE_ROOT, "created": created}
    _validate_record(record)
    deadline = time.monotonic() + timeout
    while True:
        if volume.id != record["id"] or volume.name != name:
            raise RuntimeError("Volume identity changed while waiting for readiness")
        if _volume_is_ready(volume):
            return record
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Timed out waiting for artifact volume readiness")
        time.sleep(min(1, remaining))
        if time.monotonic() >= deadline:
            raise TimeoutError("Timed out waiting for artifact volume readiness")
        volume = api.volume.get(name, create=False)


def _verify_fuse_mountinfo(mountinfo: str, path: Path) -> None:
    filesystems = []
    for line in mountinfo.splitlines():
        entry, separator, filesystem = line.partition(" - ")
        fields, filesystem_fields = entry.split(), filesystem.split()
        if not separator or len(fields) < 6 or len(filesystem_fields) < 3:
            raise ValueError("Invalid artifact volume mountinfo entry")
        mountpoint = re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), fields[4])
        if mountpoint == str(path):
            filesystems.append(filesystem_fields[0])
    if len(filesystems) != 1 or not filesystems[0].startswith("fuse"):
        raise ValueError("Artifact volume must have exactly one FUSE mountinfo entry")


def verify_volume_mount(path: Path) -> None:
    """Require the exact managed FUSE mount, never a root-disk fallback."""
    if not path.is_absolute():
        raise ValueError("Artifact volume must use an absolute mount path")
    if any(parent.is_symlink() for parent in (path, *path.parents)) or not path.is_dir():
        raise ValueError("Artifact volume mount must be a directory without symlink ancestry")
    if not path.is_mount():
        raise ValueError("Artifact volume mount is missing; refusing root-disk storage")
    if path != Path(VOLUME_MOUNT):
        raise ValueError("Artifact volume must use the managed mount path, not a system mount")
    _verify_fuse_mountinfo(Path("/proc/self/mountinfo").read_text(), path)


def archive_environment(record: dict, jobid: str, min_free_bytes: int = 1024**3) -> dict[str, str]:
    """Return only public archive settings; verify the mount on the controller."""
    _validate_record(record)
    if not isinstance(jobid, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", jobid):
        raise ValueError("Invalid job identifier")
    if type(min_free_bytes) is not int or min_free_bytes <= 0:
        raise ValueError("Minimum free bytes must be a positive integer")
    return {
        "MSWEA_ARTIFACT_STORE_ROOT": STORE_ROOT,
        "MSWEA_ARTIFACT_JOB_ID": jobid,
        "MSWEA_ARTIFACT_MIN_FREE_BYTES": str(min_free_bytes),
    }
