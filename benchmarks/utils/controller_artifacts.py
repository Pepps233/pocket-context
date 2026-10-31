"""Immutable filesystem archives, including on FUSE (no locks or volume renames).

V1 receipts contain ``version``, ``kind``, ``job_id``, an ``object`` descriptor
(path relative to the store, sha256, bytes), and a ``members`` fingerprint map.
Task receipts additionally contain method/task_id; job receipts contain the
original final status and full committed task receipts in ``tasks``. Only
jobs/<id>/job.json commits a job; task prefixes alone never imply completion.

Callers must stop the job / finalize the task before archiving. The store is a
trusted, single-writer namespace, not a coordination mechanism or a database.
Failed writes are deliberately not overwritten: repair/remove them explicitly.
"""

import gzip
import hashlib
import io
import json
import os
import re
import shutil
import stat
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import BinaryIO

_CHUNK = 1024 * 1024
_PRIVATE = {"input", "credentials", "request.json", "launcher.log", "launch.log"}
_JOB_FILES = {"plan.json", "run.log", "controller-cleanup.json", "status.json"}


def require_free_space(path: Path, min_bytes: int) -> None:
    """Fail closed if the real filesystem cannot accommodate the requested bytes."""
    if not isinstance(min_bytes, int) or isinstance(min_bytes, bool) or min_bytes < 0:
        raise ValueError("min_bytes must be a nonnegative integer")
    if shutil.disk_usage(path).free < min_bytes:
        raise OSError(f"Insufficient free space at {path}: need {min_bytes} bytes")


def _json(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"


def _identifier(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,199}", value):
        raise ValueError(f"Invalid artifact identifier: {value!r}")
    return value


def _name(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or "\0" in value
        or any(part in {"", ".", ".."} for part in value.split("/"))
        or PurePosixPath(value).is_absolute()
    ):
        raise ValueError(f"Unsafe archive path: {value!r}")
    return value


def _kind(path: Path) -> str:
    mode = path.lstat().st_mode
    if stat.S_ISDIR(mode):
        return "directory"
    if stat.S_ISREG(mode) and path.stat().st_nlink == 1:
        return "file"
    raise ValueError(f"Links and special files are not artifacts: {path}")


def _directory(path: Path) -> None:
    if ".." in path.parts:
        raise ValueError(f"Unsafe directory path: {path}")
    if _kind(path) != "directory":
        raise ValueError(f"Expected directory: {path}")


def _digest(stream: BinaryIO) -> dict:
    digest = hashlib.sha256()
    size = 0
    while data := stream.read(_CHUNK):
        digest.update(data)
        size += len(data)
    return {"sha256": digest.hexdigest(), "bytes": size}


def _read_receipt(path: Path) -> dict:
    if _kind(path) != "file":
        raise ValueError(f"Expected regular receipt: {path}")
    receipt = json.loads(path.read_bytes())
    if not isinstance(receipt, dict):
        raise ValueError(f"Invalid artifact receipt: {path}")
    return receipt


def _fingerprint(path: Path) -> dict:
    if _kind(path) != "file":
        raise ValueError(f"Expected regular file: {path}")
    with path.open("rb") as stream:
        return _digest(stream)


def _walk(root: Path, *, private: bool = False) -> dict[str, Path]:
    _directory(root)
    files = {}
    for child in sorted(root.iterdir()):
        _name(child.name)
        if private and child.name in _PRIVATE:
            continue
        if _kind(child) == "directory":
            files.update({f"{child.name}/{name}": path for name, path in _walk(child, private=private).items()})
        else:
            files[child.name] = child
    return files


def _task_files(task: Path, task_id: str, *, complete: bool) -> dict[str, Path]:
    _directory(task)
    files = {}
    for name in ("run.yaml", f"{task_id}.json", "trajectories.tar.gz", "trajectories"):
        path = task / name
        if path.exists() or path.is_symlink():
            if name == "trajectories":
                files.update({f"trajectories/{key}": value for key, value in _walk(path).items()})
            elif _kind(path) == "file":
                files[name] = path
            else:
                raise ValueError(f"Expected regular task artifact: {path}")
    if complete and not {f"{task_id}.json", "trajectories.tar.gz"} <= files.keys():
        raise ValueError("Finalized task requires result JSON and trajectories.tar.gz")
    return files


class _Reader:
    def __init__(self, stream: BinaryIO):
        self.stream = stream
        self.sha256 = hashlib.sha256()
        self.size = 0

    def read(self, size: int = -1) -> bytes:
        data = self.stream.read(size)
        self.sha256.update(data)
        self.size += len(data)
        return data

    def fingerprint(self) -> dict:
        return {"sha256": self.sha256.hexdigest(), "bytes": self.size}


def _add(archive: tarfile.TarFile, name: str, stream: BinaryIO, expected: dict) -> None:
    info = tarfile.TarInfo(_name(name))
    info.size = expected["bytes"]
    info.mode = 0o644
    reader = _Reader(stream)
    archive.addfile(info, reader)
    if reader.fingerprint() != expected or stream.read(1):
        raise ValueError(f"Artifact changed while copying: {name}")


def _bundle(path: Path, files: dict[str, Path | bytes], members: dict) -> None:
    with path.open("wb") as output, gzip.GzipFile(filename="", fileobj=output, mode="wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as archive:
            for name, source in sorted(files.items()):
                if isinstance(source, bytes):
                    _add(archive, name, io.BytesIO(source), members[name])
                else:
                    if _kind(source) != "file":
                        raise ValueError(f"Expected regular artifact: {source}")
                    with source.open("rb") as stream:
                        _add(archive, name, stream, members[name])


class ArtifactStore:
    def __init__(self, root: Path):
        if root.is_symlink():
            raise ValueError("Artifact store must not be a symlink")
        root.mkdir(parents=True, exist_ok=True)
        _directory(root)
        self.root = root.resolve()

    def _path(self, relative: str, *, parents: bool = False) -> Path:
        _name(relative)
        path = self.root
        for part in relative.split("/")[:-1]:
            path /= part
            if parents and not path.exists() and not path.is_symlink():
                path.mkdir()
            if path.exists() or path.is_symlink():
                _directory(path)
        path /= relative.split("/")[-1]
        if path.exists() or path.is_symlink():
            _kind(path)
        return path

    def _receipt_path(self, receipt: dict) -> str:
        job_id = _identifier(receipt["job_id"])
        if receipt["kind"] == "task":
            return f"jobs/{job_id}/tasks/{_identifier(receipt['method'])}/{_identifier(receipt['task_id'])}.json"
        if receipt["kind"] == "job":
            return f"jobs/{job_id}/job.json"
        raise ValueError("Unknown artifact receipt kind")

    def _validate(self, receipt: dict) -> None:
        if not isinstance(receipt, dict):
            raise ValueError("Invalid artifact receipt")
        fields = {"version", "kind", "job_id", "object", "members"}
        fields |= {"method", "task_id"} if receipt.get("kind") == "task" else {"status", "tasks"}
        if set(receipt) != fields or type(receipt["version"]) is not int or receipt["version"] != 1:
            raise ValueError("Invalid artifact receipt schema")
        self._receipt_path(receipt)
        obj = receipt["object"]
        if not isinstance(obj, dict):
            raise ValueError("Invalid artifact object descriptor")
        self._validate_fingerprint({key: value for key, value in obj.items() if key != "path"})
        if set(obj) != {"path", "sha256", "bytes"} or obj["path"] != f"objects/{obj['sha256']}.tar.gz":
            raise ValueError("Invalid content-addressed object path")
        if not isinstance(receipt["members"], dict):
            raise ValueError("Invalid member manifest")
        for name, fingerprint in receipt["members"].items():
            _name(name)
            self._validate_fingerprint(fingerprint)
            if receipt["kind"] == "task":
                if name not in {
                    "run.yaml",
                    f"{receipt['task_id']}.json",
                    "trajectories.tar.gz",
                } and not name.startswith("trajectories/"):
                    raise ValueError(f"Unexpected task member: {name}")
            elif name not in _JOB_FILES and not name.startswith("output/"):
                raise ValueError(f"Unexpected job member: {name}")
            if any(part in _PRIVATE for part in name.split("/")):
                raise ValueError(f"Private artifact path: {name}")
        if receipt["kind"] == "task":
            if not {f"{receipt['task_id']}.json", "trajectories.tar.gz"} <= receipt["members"].keys():
                raise ValueError("Task receipt is missing finalized payload")
        else:
            self._inactive(receipt["status"])
            if "status.json" not in receipt["members"] or not isinstance(receipt["tasks"], list):
                raise ValueError("Invalid job receipt")

    @staticmethod
    def _validate_fingerprint(value: dict) -> None:
        if (
            not isinstance(value, dict)
            or set(value) != {"sha256", "bytes"}
            or not isinstance(value["sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", value["sha256"])
            or type(value["bytes"]) is not int
            or value["bytes"] < 0
        ):
            raise ValueError("Invalid artifact fingerprint")

    @staticmethod
    def _inactive(status: dict) -> None:
        if (
            not isinstance(status, dict)
            or not isinstance(status.get("state"), str)
            or status.get("state") not in {"completed", "failed", "cancelled", "interrupted"}
            or status.get("runner_alive")
            or status.get("child_alive")
        ):
            raise ValueError("Job must have an inactive final status")

    def _verify_object(self, receipt: dict) -> None:
        obj = receipt["object"]
        path = self._path(obj["path"])
        if _fingerprint(path) != {key: obj[key] for key in ("sha256", "bytes")}:
            raise ValueError(f"Corrupt artifact object: {path}")
        seen = set()
        with tarfile.open(path, mode="r|gz") as archive:
            for member in archive:
                name = _name(member.name)
                if not member.isfile() or name in seen or name not in receipt["members"]:
                    raise ValueError(f"Unsafe or unexpected archive member: {name}")
                seen.add(name)
                with archive.extractfile(member) as stream:
                    if _digest(stream) != receipt["members"][name]:
                        raise ValueError(f"Corrupt artifact member: {name}")
                if receipt["kind"] == "job" and name == "status.json":
                    expected = _json(receipt["status"])
                    if receipt["members"][name] != _digest(io.BytesIO(expected)):
                        raise ValueError("Job status does not match its metadata bundle")
        if seen != receipt["members"].keys():
            raise ValueError("Artifact object is missing members")

    def _verify_committed(self, receipt: dict) -> None:
        self._validate(receipt)
        path = self._path(self._receipt_path(receipt))
        if _kind(path) != "file" or path.read_bytes() != _json(receipt):
            raise ValueError(f"Committed receipt mismatch: {path}")
        self._verify_object(receipt)

    def _validate_task_references(self, receipt: dict) -> None:
        seen = set()
        for task in receipt["tasks"]:
            if not isinstance(task, dict) or task.get("kind") != "task" or task.get("job_id") != receipt["job_id"]:
                raise ValueError("Invalid job task reference")
            self._validate(task)
            key = (task["method"], task["task_id"])
            if key in seen:
                raise ValueError("Duplicate job task reference")
            seen.add(key)
        self._export_members(receipt)

    def verify(self, receipt: dict) -> None:
        """Verify the committed receipt and all bytes, including referenced tasks."""
        self._verify_committed(receipt)
        if receipt["kind"] == "job":
            self._validate_task_references(receipt)
            for task in receipt["tasks"]:
                self.verify(task)

    def _immutable(self, relative: str, source: Path | bytes) -> None:
        path = self._path(relative, parents=True)
        if path.exists():
            if isinstance(source, bytes):
                if _kind(path) != "file" or path.read_bytes() != source:
                    raise ValueError(f"Immutable receipt mismatch: {path}")
            elif _fingerprint(path) != _fingerprint(source):
                raise ValueError(f"Immutable object mismatch: {path}")
            return
        with path.open("xb") as output:
            if isinstance(source, bytes):
                output.write(source)
            else:
                with source.open("rb") as stream:
                    shutil.copyfileobj(stream, output, _CHUNK)
        if isinstance(source, bytes) and path.read_bytes() != source:
            raise ValueError(f"Receipt readback mismatch: {path}")

    def _publish(self, fields: dict, files: dict[str, Path | bytes]) -> dict:
        members = {
            name: _digest(io.BytesIO(source)) if isinstance(source, bytes) else _fingerprint(source)
            for name, source in files.items()
        }
        with tempfile.TemporaryDirectory(prefix="controller-artifacts-") as temporary:
            bundle = Path(temporary) / "bundle.tar.gz"
            _bundle(bundle, files, members)
            fingerprint = _fingerprint(bundle)
            receipt = fields | {
                "version": 1,
                "object": {"path": f"objects/{fingerprint['sha256']}.tar.gz"} | fingerprint,
                "members": members,
            }
            self._validate(receipt)
            if receipt["kind"] == "job":
                self._export_members(receipt)
                for task in receipt["tasks"]:
                    self.verify(task)
            self._immutable(receipt["object"]["path"], bundle)
            self._verify_object(receipt)
            self._immutable(self._receipt_path(receipt), _json(receipt))
        self.verify(receipt)
        return receipt

    @staticmethod
    def _local_matches(task: Path, receipt: dict, *, pruned: bool) -> None:
        files = _task_files(task, receipt["task_id"], complete=not pruned)
        current = {name: _fingerprint(path) for name, path in files.items()}
        expected = receipt["members"]
        if any(expected.get(name) != fingerprint for name, fingerprint in current.items()):
            raise ValueError("Local task artifacts changed since archival")
        required = {
            name
            for name in expected
            if not pruned or not (name.startswith("trajectories/") or name == "trajectories.tar.gz")
        }
        if not required <= current.keys():
            raise ValueError("Local task artifacts are missing")

    @staticmethod
    def _local_receipt(task: Path, receipt: dict) -> None:
        path = task / "volume-artifacts.json"
        if path.exists() or path.is_symlink():
            if _kind(path) != "file" or path.read_bytes() != _json(receipt):
                raise ValueError("Local artifact receipt mismatch")
            return
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=task, prefix=".volume-artifacts-", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(_json(receipt))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)  # Local disk only; never used on the mounted store.
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def archive_task(self, task: Path, job_id: str, method: str, task_id: str, *, prune: bool = True) -> dict:
        fields = {
            "kind": "task",
            "job_id": _identifier(job_id),
            "method": _identifier(method),
            "task_id": _identifier(task_id),
        }
        _directory(task)
        local = task / "volume-artifacts.json"
        remote = self._path(self._receipt_path(fields))
        if local.exists() or local.is_symlink():
            if _kind(local) != "file":
                raise ValueError("Invalid local artifact receipt")
            receipt = _read_receipt(local)
            if any(receipt.get(key) != value for key, value in fields.items()):
                raise ValueError("Local artifact receipt identity mismatch")
            self.verify(receipt)
            self._local_matches(task, receipt, pruned=True)
        elif remote.exists():
            receipt = _read_receipt(remote)
            if any(receipt.get(key) != value for key, value in fields.items()):
                raise ValueError("Committed task receipt identity mismatch")
            self.verify(receipt)
            self._local_matches(task, receipt, pruned=False)
        else:
            receipt = self._publish(fields, _task_files(task, task_id, complete=True))
            self._local_matches(task, receipt, pruned=False)
        self._local_receipt(task, receipt)
        if prune:
            self.verify(receipt)
            self._local_matches(task, receipt, pruned=True)
            raw = task / "trajectories"
            if raw.exists():
                shutil.rmtree(raw)
            (task / "trajectories.tar.gz").unlink(missing_ok=True)
        return receipt

    def _tasks(self, job_id: str) -> list[dict]:
        prefix = self._path(f"jobs/{job_id}/tasks")
        if not prefix.exists():
            return []
        tasks = []
        for relative, path in _walk(prefix).items():
            receipt = _read_receipt(path)
            if receipt.get("kind") != "task" or receipt.get("job_id") != job_id:
                raise ValueError("Invalid committed task receipt")
            if self._receipt_path(receipt) != f"jobs/{job_id}/tasks/{relative}":
                raise ValueError("Task receipt path identity mismatch")
            self.verify(receipt)
            tasks.append(receipt)
        return tasks

    def finalize_job(self, job: Path, job_id: str, final_status: dict) -> dict:
        """Commit inactive job metadata; never read current status or private inputs."""
        _identifier(job_id)
        _directory(job)
        self._inactive(final_status)
        tasks = self._tasks(job_id)
        archived = {}
        for task in tasks:
            prefix = f"output/{task['method']}/tasks/{task['task_id']}/"
            archived.update({prefix + name: fingerprint for name, fingerprint in task["members"].items()})
        files: dict[str, Path | bytes] = {"status.json": _json(final_status)}
        for name in sorted(_JOB_FILES - {"status.json"}):
            path = job / name
            if path.exists() or path.is_symlink():
                if _kind(path) != "file":
                    raise ValueError(f"Expected regular job metadata: {path}")
                files[name] = path
        output = job / "output"
        if output.exists() or output.is_symlink():
            for name, path in _walk(output, private=True).items():
                name = f"output/{name}"
                if name in archived:
                    if _fingerprint(path) != archived[name]:
                        raise ValueError(f"Archived immutable output changed: {name}")
                else:
                    if path.name == "volume-artifacts.json":
                        receipt = _read_receipt(path)
                        self.verify(receipt)
                        if receipt["kind"] != "task":
                            raise ValueError("Expected task receipt in output")
                        expected = f"output/{receipt['method']}/tasks/{receipt['task_id']}/volume-artifacts.json"
                        if receipt.get("job_id") != job_id or name != expected or receipt not in tasks:
                            raise ValueError("Output task receipt identity mismatch")
                    files[name] = path
        return self._publish({"kind": "job", "job_id": job_id, "status": final_status, "tasks": tasks}, files)

    def task_receipt(self, job_id: str, method: str, task_id: str) -> dict | None:
        """Read a fully verified committed task receipt, or None if absent."""
        fields = {
            "kind": "task",
            "job_id": _identifier(job_id),
            "method": _identifier(method),
            "task_id": _identifier(task_id),
        }
        path = self._path(self._receipt_path(fields))
        if not path.exists():
            return None
        receipt = _read_receipt(path)
        if any(receipt.get(key) != value for key, value in fields.items()):
            raise ValueError("Committed task receipt identity mismatch")
        self.verify(receipt)
        return receipt

    def _job_receipt(self, job_id: str, *, full: bool = False) -> dict | None:
        path = self._path(f"jobs/{_identifier(job_id)}/job.json")
        if not path.exists():
            return None
        receipt = _read_receipt(path)
        if receipt.get("kind") != "job" or receipt.get("job_id") != job_id:
            raise ValueError("Job receipt identity mismatch")
        if full:
            self.verify(receipt)
        else:
            self._verify_committed(receipt)
            self._validate_task_references(receipt)
        return receipt

    def verify_job(self, job_id: str, *, local: Path | None = None) -> None:
        """Verify every payload and optional local-output coverage before cleanup."""
        receipt = self._job_receipt(job_id)
        if receipt is None:
            raise ValueError("Job has no committed archive")
        self.verify(receipt)
        if local is not None and (local / "output").exists():
            members = self._export_members(receipt)
            for name, path in _walk(local / "output", private=True).items():
                if members.get(f"output/{name}") != _fingerprint(path):
                    raise ValueError(
                        "Local output differs from its immutable archive; preserve recovery in a fresh job namespace before cleanup"
                    )

    def job_status(self, job_id: str) -> dict | None:
        """Read verified final metadata, without re-reading task payload objects."""
        receipt = self._job_receipt(job_id)
        if receipt is None:
            return None
        return receipt["status"] | {"archived_only": True, "runner_alive": False, "child_alive": False}

    def read_job_file(self, job_id: str, name: str) -> bytes | None:
        """Read one verified metadata-bundle file (not task-bundle contents)."""
        _name(name)
        receipt = self._job_receipt(job_id)
        if receipt is None or name not in receipt["members"]:
            return None
        with tarfile.open(self._path(receipt["object"]["path"]), mode="r|gz") as archive:
            for member in archive:
                if member.name == name:
                    with archive.extractfile(member) as stream:
                        data = stream.read()
                    if _digest(io.BytesIO(data)) != receipt["members"][name]:
                        raise ValueError(f"Corrupt artifact member: {name}")
                    return data
        raise ValueError(f"Artifact object is missing member: {name}")

    def list_jobs(self) -> dict[str, dict]:
        jobs = self._path("jobs")
        if not jobs.exists():
            return {}
        _directory(jobs)
        result = {}
        for path in sorted(jobs.iterdir()):
            _identifier(path.name)
            _directory(path)
            descriptor = self._path(f"jobs/{path.name}/job.json")
            if descriptor.exists():
                result[path.name] = self.job_status(path.name)
        return result

    @staticmethod
    def _export_members(receipt: dict) -> dict:
        members = dict(receipt["members"])
        for task in receipt["tasks"]:
            prefix = f"output/{task['method']}/tasks/{task['task_id']}/"
            for name, fingerprint in task["members"].items():
                name = prefix + name
                if name in members and members[name] != fingerprint:
                    raise ValueError(f"Conflicting immutable export member: {name}")
                members[name] = fingerprint
        names = set(members)
        for name in names:
            if any(parent.as_posix() in names for parent in PurePosixPath(name).parents):
                raise ValueError(f"File/directory export collision: {name}")
        return members

    def export_job(self, job_id: str, destination: Path) -> dict:
        """Stream verified bundles to a new combined tar.gz, without extraction."""
        receipt = self._job_receipt(job_id, full=True)
        if receipt is None:
            raise FileNotFoundError(f"No committed archive for job {job_id}")
        expected = self._export_members(receipt)
        written = set()
        success = False
        if destination.absolute().is_relative_to(self.root):
            destination = self._path(destination.absolute().relative_to(self.root).as_posix(), parents=True)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as output:
            try:
                with gzip.GzipFile(filename="", fileobj=output, mode="wb", mtime=0) as compressed:
                    with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as combined:
                        for bundle in [receipt, *receipt["tasks"]]:
                            prefix = (
                                ""
                                if bundle["kind"] == "job"
                                else f"output/{bundle['method']}/tasks/{bundle['task_id']}/"
                            )
                            with tarfile.open(self._path(bundle["object"]["path"]), mode="r|gz") as archive:
                                for member in archive:
                                    name = prefix + _name(member.name)
                                    if not member.isfile() or name not in expected:
                                        raise ValueError(f"Unsafe export member: {name}")
                                    if name not in written:
                                        with archive.extractfile(member) as stream:
                                            _add(combined, name, stream, expected[name])
                                        written.add(name)
                if written != expected.keys():
                    raise ValueError("Export is missing members")
                success = True
            finally:
                if not success:
                    destination.unlink(missing_ok=True)
        return {"path": str(destination)} | _fingerprint(destination)
