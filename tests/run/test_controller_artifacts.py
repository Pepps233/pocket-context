import copy
import hashlib
import io
import json
import os
import shutil
import tarfile
from pathlib import Path

import pytest

from pocketcontextagent.run.benchmarks.utils.controller_artifacts import ArtifactStore, require_free_space

FINAL = {"state": "failed", "exit_code": 1, "finished_at": "2026-01-01", "runner_pid": 123, "child_pid": 456}


def encoded(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"


def fingerprint(data: bytes) -> dict:
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def files(root: Path) -> dict[str, bytes]:
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def unpack(path: Path) -> dict[str, bytes]:
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        assert all(member.isfile() for member in members)
        assert len({member.name for member in members}) == len(members)
        return {member.name: archive.extractfile(member).read() for member in members}


def make_task(job: Path, method: str = "baseline", task_id: str = "case-1") -> Path:
    task = job / "output" / method / "tasks" / task_id
    write(task / "run.yaml", b"original: provenance\n")
    write(task / f"{task_id}.json", b'{"answer":"answer", "original":true}\n')
    write(task / "worker.yaml", b"worker: mutable\n")
    write(task / "trajectories" / "nested" / "trace.json", b'[{"step":1}]\n')
    write(task / "trajectories" / "binary", bytes(range(256)) * 5000)
    write(task / "trajectories" / ("long-" + "x" * 120), b"long path\n")
    with tarfile.open(task / "trajectories.tar.gz", "w:gz") as archive:
        archive.add(task / "trajectories", arcname="trajectories")
    return task


@pytest.fixture
def example(tmp_path: Path) -> tuple[ArtifactStore, Path, Path]:
    job = tmp_path / "local-job"
    task = make_task(job)
    write(job / "plan.json", b'{"tasks": ["case-1", "case-2"]}\n')
    write(job / "status.json", b'{"state":"running","runner_alive":true}\n')
    write(job / "run.log", b"redacted: [REDACTED]\n")
    write(job / "controller-cleanup.json", b'{"cleanup":"complete"}\n')
    write(job / "output" / "baseline" / "summary.csv", b"case,result\ncase-1,answer\n")
    write(job / "output" / "baseline" / "tasks" / "case-2" / "worker.yaml", b"state: failed\n")
    write(job / "output" / "baseline" / "tasks" / "case-2" / "trajectories" / "failure", b"unfinished trace\n")
    for name in ("input/gold.json", "request.json", "credentials/keys.json", "launcher.log", "launch.log"):
        write(job / name, b"PRIVATE-GOLD-OR-CREDENTIAL\n")
        write(job / "output" / name, b"PRIVATE-GOLD-OR-CREDENTIAL\n")
    return ArtifactStore(tmp_path / "volume" / "prefix"), job, task


def assert_bulk(task: Path) -> None:
    assert (task / "trajectories.tar.gz").is_file()
    assert (task / "trajectories" / "binary").read_bytes() == bytes(range(256)) * 5000


def test_full_roundtrip_pruning_fingerprints_and_fresh_store(example: tuple, tmp_path: Path) -> None:
    store, job, task = example
    expected = {
        name: data
        for name, data in files(job).items()
        if name in {"plan.json", "run.log", "controller-cleanup.json"} or name.startswith("output/baseline/")
    }
    originals = {
        name: (fingerprint((task / name).read_bytes()), (task / name).stat().st_mtime_ns)
        for name in ("run.yaml", "case-1.json", "worker.yaml")
    }
    status_before = (job / "status.json").read_bytes()
    receipt = store.archive_task(task, "job-1", "baseline", "case-1")
    assert receipt["kind"] == "task" and receipt["version"] == 1
    assert set(receipt["members"]) == set(files(task)) - {"worker.yaml", "volume-artifacts.json"} | {
        "trajectories.tar.gz",
        "trajectories/nested/trace.json",
        "trajectories/binary",
        "trajectories/" + "long-" + "x" * 120,
    }
    assert not (task / "trajectories").exists() and not (task / "trajectories.tar.gz").exists()
    assert json.loads((task / "volume-artifacts.json").read_bytes()) == receipt
    task_receipt_time = (task / "volume-artifacts.json").stat().st_mtime_ns
    object_time = (store.root / receipt["object"]["path"]).stat().st_mtime_ns
    worker = task / "worker.yaml"
    worker.write_bytes(b"worker: updated after archival\n")
    originals["worker.yaml"] = (fingerprint(worker.read_bytes()), worker.stat().st_mtime_ns)
    assert store.archive_task(task, "job-1", "baseline", "case-1") == receipt
    assert (task / "volume-artifacts.json").stat().st_mtime_ns == task_receipt_time
    assert (store.root / receipt["object"]["path"]).stat().st_mtime_ns == object_time
    assert {
        name: (fingerprint((task / name).read_bytes()), (task / name).stat().st_mtime_ns) for name in originals
    } == originals
    expected["output/baseline/tasks/case-1/worker.yaml"] = worker.read_bytes()
    expected["output/baseline/tasks/case-1/volume-artifacts.json"] = (task / "volume-artifacts.json").read_bytes()
    expected["status.json"] = encoded(FINAL)
    descriptor = store.finalize_job(job, "job-1", FINAL)
    assert store.finalize_job(job, "job-1", FINAL) == descriptor
    assert (job / "status.json").read_bytes() == status_before
    assert descriptor["tasks"] == [receipt] and descriptor["status"] == FINAL
    shutil.rmtree(job)
    fresh = ArtifactStore(store.root)
    assert fresh.list_jobs() == {"job-1": FINAL | {"archived_only": True, "runner_alive": False, "child_alive": False}}
    assert fresh.job_status("job-1") == fresh.list_jobs()["job-1"]
    fresh.verify(receipt)
    fresh.verify(descriptor)
    destination = fresh.root / "exports" / "unique-export.tar.gz"
    exported = fresh.export_job("job-1", destination)
    assert exported == {"path": str(destination)} | fingerprint(destination.read_bytes())
    assert unpack(destination) == expected
    assert b"PRIVATE-GOLD-OR-CREDENTIAL" not in b"".join(unpack(destination).values())
    with pytest.raises(FileExistsError):
        fresh.export_job("job-1", destination)
    assert unpack(destination) == expected
    assert not list(tmp_path.glob("**/.volume-artifacts-*"))


def test_keep_bulk_and_recover_remote_commit_without_local_receipt(example: tuple) -> None:
    store, _, task = example
    original = files(task)
    receipt = store.archive_task(task, "job", "baseline", "case-1", prune=False)
    assert_bulk(task)
    assert {name: value for name, value in files(task).items() if name != "volume-artifacts.json"} == original
    (task / "volume-artifacts.json").unlink()
    assert store.archive_task(task, "job", "baseline", "case-1") == receipt
    assert not (task / "trajectories").exists()
    assert (task / "case-1.json").read_bytes() == original["case-1.json"]


@pytest.mark.parametrize(
    ("damage",),  # noqa: PT006
    [("object",), ("remote-receipt",), ("local-receipt",), ("raw",), ("archive",), ("result",), ("run",)],
)
def test_corruption_or_local_changes_never_prune(example: tuple, damage: str) -> None:
    store, _, task = example
    receipt = store.archive_task(task, "job", "baseline", "case-1", prune=False)
    if damage == "object":
        (store.root / receipt["object"]["path"]).write_bytes(b"corrupt")
    elif damage == "remote-receipt":
        (store.root / "jobs/job/tasks/baseline/case-1.json").write_bytes(b"{}")
    elif damage == "local-receipt":
        changed = copy.deepcopy(receipt)
        changed["object"]["path"] = "../../outside.tar.gz"
        (task / "volume-artifacts.json").write_bytes(encoded(changed))
    else:
        name = {
            "raw": "trajectories/nested/trace.json",
            "archive": "trajectories.tar.gz",
            "result": "case-1.json",
            "run": "run.yaml",
        }[damage]
        (task / name).write_bytes(b"changed")
    with pytest.raises(ValueError):
        store.archive_task(task, "job", "baseline", "case-1")
    assert (task / "trajectories.tar.gz").exists() and (task / "trajectories" / "binary").exists()


def test_existing_object_conflict_is_not_overwritten_or_committed(example: tuple, tmp_path: Path) -> None:
    source_store, _, task = example
    receipt = source_store.archive_task(task, "job", "baseline", "case-1", prune=False)
    (task / "volume-artifacts.json").unlink()
    target = ArtifactStore(tmp_path / "conflicting-volume")
    write(target.root / receipt["object"]["path"], b"existing-corrupt-object")
    with pytest.raises(ValueError, match="Immutable object mismatch"):
        target.archive_task(task, "job", "baseline", "case-1")
    assert_bulk(task)
    assert not (task / "volume-artifacts.json").exists()
    assert not (target.root / "jobs/job/tasks/baseline/case-1.json").exists()
    assert (target.root / receipt["object"]["path"]).read_bytes() == b"existing-corrupt-object"


def test_existing_receipt_conflict_is_not_overwritten(example: tuple) -> None:
    store, _, task = example
    receipt = store.archive_task(task, "job", "baseline", "case-1", prune=False)
    (task / "volume-artifacts.json").unlink()
    damaged = copy.deepcopy(receipt)
    damaged["members"]["run.yaml"] = fingerprint(b"different immutable run")
    path = store.root / "jobs/job/tasks/baseline/case-1.json"
    path.write_bytes(encoded(damaged))
    with pytest.raises(ValueError, match="Corrupt artifact member"):
        store.archive_task(task, "job", "baseline", "case-1")
    assert path.read_bytes() == encoded(damaged)
    assert_bulk(task)
    assert not (task / "volume-artifacts.json").exists()


@pytest.mark.parametrize(("missing",), [("case-1.json",), ("trajectories.tar.gz",)])  # noqa: PT006
def test_requires_finalized_payload_without_modifying_local_files(example: tuple, missing: str) -> None:
    store, _, task = example
    (task / missing).unlink()
    before = files(task)
    with pytest.raises(ValueError, match="requires result JSON"):
        store.archive_task(task, "job", "baseline", "case-1")
    assert files(task) == before
    assert not (store.root / "jobs").exists()


@pytest.mark.parametrize(("link_kind",), [("symlink",), ("directory-link",), ("hardlink",), ("fifo",)])  # noqa: PT006
def test_links_and_special_files_rejected_before_commit(example: tuple, tmp_path: Path, link_kind: str) -> None:
    store, _, task = example
    outside = tmp_path / "outside"
    outside.write_bytes(b"private")
    path = task / "trajectories" / "unsafe"
    if link_kind == "symlink":
        path.symlink_to(outside)
    elif link_kind == "directory-link":
        path.symlink_to(tmp_path, target_is_directory=True)
    elif link_kind == "hardlink":
        path.hardlink_to(outside)
    else:
        os.mkfifo(path)
    with pytest.raises(ValueError, match="Links and special"):
        store.archive_task(task, "job", "baseline", "case-1")
    assert_bulk(task)
    assert not (task / "volume-artifacts.json").exists()
    assert not (store.root / "jobs").exists()
    assert outside.read_bytes() == b"private"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("job", "../escape"),
        ("method", "a/b"),
        ("task", "/absolute"),
        ("task", ".."),
        ("job", ""),
        ("method", "a\\b"),
        ("task", "x\0y"),
    ],
)
def test_identifiers_reject_traversal(example: tuple, field: str, value: str) -> None:
    store, _, task = example
    identifiers = {"job": "job", "method": "baseline", "task": "case-1"} | {field: value}
    with pytest.raises(ValueError, match="identifier"):
        store.archive_task(task, identifiers["job"], identifiers["method"], identifiers["task"])
    assert_bulk(task)
    assert store.list_jobs() == {}


def test_symlink_roots_volume_components_and_local_task_rejected(example: tuple, tmp_path: Path) -> None:
    store, _, task = example
    linked_task = tmp_path / "linked-task"
    linked_task.symlink_to(task, target_is_directory=True)
    with pytest.raises(ValueError, match="Links and special"):
        store.archive_task(linked_task, "job", "baseline", "case-1")
    linked_store = tmp_path / "linked-store"
    linked_store.symlink_to(store.root, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        ArtifactStore(linked_store)
    outside = tmp_path / "outside"
    outside.mkdir()
    (store.root / "objects").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="Links and special"):
        store.archive_task(task, "job", "baseline", "case-1")
    assert_bulk(task)
    assert not list(outside.iterdir())


def install_object(store: ArtifactStore, receipt: dict, entries: list[tuple[tarfile.TarInfo, bytes]]) -> dict:
    """Write deliberately crafted but content-addressed filesystem objects."""
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        for info, data in entries:
            archive.addfile(info, io.BytesIO(data))
    data = output.getvalue()
    changed = copy.deepcopy(receipt)
    changed["object"] = {"path": f"objects/{hashlib.sha256(data).hexdigest()}.tar.gz"} | fingerprint(data)
    write(store.root / changed["object"]["path"], data)
    if changed["kind"] == "task":
        path = f"jobs/{changed['job_id']}/tasks/{changed['method']}/{changed['task_id']}.json"
    else:
        path = f"jobs/{changed['job_id']}/job.json"
    write(store.root / path, encoded(changed))
    return changed


def entry(name: str, data: bytes) -> tuple[tarfile.TarInfo, bytes]:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    return info, data


@pytest.mark.parametrize(
    ("unsafe",),  # noqa: PT006
    [
        ("traversal",),
        ("absolute",),
        ("normalized",),
        ("symlink",),
        ("hardlink",),
        ("fifo",),
        ("duplicate",),
        ("member-hash",),
    ],
)
def test_verify_checks_members_even_when_whole_object_hash_matches(example: tuple, unsafe: str) -> None:
    store, _, task = example
    receipt = store.archive_task(task, "job", "baseline", "case-1", prune=False)
    payload = unpack(store.root / receipt["object"]["path"])
    entries = [entry(name, data) for name, data in payload.items()]
    if unsafe in {"traversal", "absolute", "normalized"}:
        entries.append(
            entry(
                {"traversal": "../escape", "absolute": "/escape", "normalized": "trajectories/./binary"}[unsafe], b"bad"
            )
        )
    elif unsafe in {"symlink", "hardlink", "fifo"}:
        info = tarfile.TarInfo("trajectories/link")
        info.type = {"symlink": tarfile.SYMTYPE, "hardlink": tarfile.LNKTYPE, "fifo": tarfile.FIFOTYPE}[unsafe]
        info.linkname = "/outside"
        entries.append((info, b""))
    elif unsafe == "duplicate":
        entries.append(entry("run.yaml", payload["run.yaml"]))
    else:
        entries = [entry(name, b"bad" if name == "run.yaml" else data) for name, data in payload.items()]
    changed = install_object(store, receipt, entries)
    (task / "volume-artifacts.json").write_bytes(encoded(changed))
    with pytest.raises(ValueError):
        store.verify(changed)
    with pytest.raises(ValueError):
        store.archive_task(task, "job", "baseline", "case-1")
    assert_bulk(task)


@pytest.mark.parametrize(
    ("path",),  # noqa: PT006
    [("../outside.tar.gz",), ("/outside.tar.gz",), ("objects/../escape",), ("objects/not-the-hash.tar.gz",)],
)
def test_receipt_object_paths_must_be_content_addressed_and_bound(example: tuple, path: str) -> None:
    store, _, task = example
    receipt = store.archive_task(task, "job", "baseline", "case-1", prune=False)
    receipt["object"]["path"] = path
    with pytest.raises(ValueError, match="object path"):
        store.verify(receipt)
    assert_bulk(task)


def test_only_job_descriptor_commits_completion_and_missing_original_task_is_supported(example: tuple) -> None:
    store, job, task = example
    receipt = store.archive_task(task, "partial", "baseline", "case-1")
    assert store.list_jobs() == {}
    assert store.job_status("partial") is None
    assert store.read_job_file("partial", "run.log") is None
    with pytest.raises(FileNotFoundError):
        store.export_job("partial", store.root / "not-created.tar.gz")
    assert not (store.root / "not-created.tar.gz").exists()
    shutil.rmtree(task)
    descriptor = store.finalize_job(job, "partial", FINAL)
    assert descriptor["tasks"] == [receipt]
    assert store.list_jobs()["partial"]["state"] == "failed"
    assert (
        unpack(Path(store.export_job("partial", store.root / "export.tar.gz")["path"]))[
            "output/baseline/tasks/case-1/case-1.json"
        ]
        == b'{"answer":"answer", "original":true}\n'
    )


@pytest.mark.parametrize(
    ("status",),  # noqa: PT006
    [
        ({"state": "running"},),
        ({"state": "starting"},),
        ({"state": "completed", "runner_alive": True},),
        ({"state": "interrupted", "child_alive": True},),
    ],
)
def test_finalization_rejects_live_job(example: tuple, status: dict) -> None:
    store, job, task = example
    with pytest.raises(ValueError, match="inactive"):
        store.finalize_job(job, "job", status)
    assert_bulk(task)
    assert store.list_jobs() == {}


def test_finalize_rejects_changed_immutable_output_and_output_links(example: tuple, tmp_path: Path) -> None:
    store, job, task = example
    store.archive_task(task, "job", "baseline", "case-1")
    original = (task / "case-1.json").read_bytes()
    (task / "case-1.json").write_bytes(b"rewritten result")
    with pytest.raises(ValueError, match="immutable output changed"):
        store.finalize_job(job, "job", FINAL)
    (task / "case-1.json").write_bytes(original)
    (job / "output" / "unsafe-link").symlink_to(tmp_path)
    with pytest.raises(ValueError, match="Links and special"):
        store.finalize_job(job, "job", FINAL)
    assert store.list_jobs() == {}


def test_job_receipt_immutable_status_not_rewritten(example: tuple) -> None:
    store, job, _ = example
    descriptor = store.finalize_job(job, "job", FINAL)
    path = store.root / "jobs/job/job.json"
    before = path.read_bytes(), path.stat().st_mtime_ns
    with pytest.raises(ValueError, match="Immutable receipt mismatch"):
        store.finalize_job(job, "job", FINAL | {"exit_code": 9})
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before
    assert store.job_status("job")["exit_code"] == 1
    assert descriptor["status"] == FINAL


@pytest.mark.parametrize(("conflict",), [(False,), (True,)])  # noqa: PT006
def test_export_deduplicates_agreeing_overlaps_and_rejects_conflicts(example: tuple, conflict: bool) -> None:
    store, job, task = example
    task_receipt = store.archive_task(task, "job", "baseline", "case-1")
    receipt = store.finalize_job(job, "job", FINAL)
    payload = unpack(store.root / receipt["object"]["path"])
    name = "output/baseline/tasks/case-1/case-1.json"
    payload[name] = b"different result" if conflict else (task / "case-1.json").read_bytes()
    receipt["members"][name] = fingerprint(payload[name])
    changed = install_object(store, receipt, [entry(key, data) for key, data in payload.items()])
    destination = store.root / "overlap.tar.gz"
    if conflict:
        with pytest.raises(ValueError, match="Conflicting immutable"):
            store.export_job("job", destination)
        assert not destination.exists()
    else:
        store.verify(changed)
        assert unpack(Path(store.export_job("job", destination)["path"]))[name] == (task / "case-1.json").read_bytes()
        assert task_receipt["members"]["case-1.json"] == changed["members"][name]


def test_metadata_and_task_readers_verify_and_return_absent_files(example: tuple) -> None:
    store, job, task = example
    assert store.task_receipt("job", "baseline", "case-1") is None
    assert store.job_status("job") is None
    assert store.read_job_file("job", "run.log") is None
    task_receipt = store.archive_task(task, "job", "baseline", "case-1")
    assert store.task_receipt("job", "baseline", "case-1") == task_receipt
    assert store.task_receipt("job", "baseline", "absent") is None
    write(job / "output" / "sandbox-cleanup.json", b'{"deleted":true}\n')
    descriptor = store.finalize_job(job, "job", FINAL)
    assert store.read_job_file("job", "run.log") == b"redacted: [REDACTED]\n"
    assert store.read_job_file("job", "output/sandbox-cleanup.json") == b'{"deleted":true}\n'
    assert store.read_job_file("job", "absent") is None
    assert store.read_job_file("job", "request.json") is None
    assert store.read_job_file("job", "output/baseline/tasks/case-1/case-1.json") is None
    with pytest.raises(ValueError, match="Unsafe archive path"):
        store.read_job_file("job", "../request.json")
    (store.root / descriptor["object"]["path"]).write_bytes(b"broken metadata")
    with pytest.raises(ValueError, match="Corrupt artifact object"):
        store.read_job_file("job", "run.log")
    with pytest.raises(ValueError, match="Corrupt artifact object"):
        store.job_status("job")


def test_missing_task_payload_keeps_metadata_readers_available_but_not_verify_or_export(example: tuple) -> None:
    store, job, task = example
    task_receipt = store.archive_task(task, "job", "baseline", "case-1")
    write(job / "output" / "sandbox-cleanup.json", b'{"deleted":true}\n')
    descriptor = store.finalize_job(job, "job", FINAL)
    (store.root / task_receipt["object"]["path"]).unlink()
    shutil.rmtree(job)
    fresh = ArtifactStore(store.root)
    assert fresh.job_status("job") == FINAL | {"archived_only": True, "runner_alive": False, "child_alive": False}
    assert fresh.list_jobs() == {"job": fresh.job_status("job")}
    assert fresh.read_job_file("job", "run.log") == b"redacted: [REDACTED]\n"
    assert fresh.read_job_file("job", "output/sandbox-cleanup.json") == b'{"deleted":true}\n'
    with pytest.raises(FileNotFoundError):
        fresh.verify(descriptor)
    with pytest.raises(FileNotFoundError):
        fresh.task_receipt("job", "baseline", "case-1")
    with pytest.raises(FileNotFoundError):
        fresh.export_job("job", store.root / "exports" / "missing-task.tar.gz")
    assert not (store.root / "exports" / "missing-task.tar.gz").exists()


@pytest.mark.parametrize(
    ("damage",),  # noqa: PT006
    [("duplicate",), ("other-job",), ("unsafe-path",)],
)
def test_metadata_readers_validate_task_reference_schema_and_uniqueness(example: tuple, damage: str) -> None:
    store, job, task = example
    store.archive_task(task, "job", "baseline", "case-1")
    descriptor = store.finalize_job(job, "job", FINAL)
    payload = unpack(store.root / descriptor["object"]["path"])
    if damage == "duplicate":
        descriptor["tasks"].append(copy.deepcopy(descriptor["tasks"][0]))
    elif damage == "other-job":
        descriptor["tasks"][0]["job_id"] = "other"
    else:
        descriptor["tasks"][0]["object"]["path"] = "../outside.tar.gz"
    install_object(store, descriptor, [entry(name, data) for name, data in payload.items()])
    with pytest.raises(ValueError):
        store.job_status("job")
    with pytest.raises(ValueError):
        store.read_job_file("job", "run.log")


def test_disk_guard_uses_real_usage_fail_closed_without_patches(tmp_path: Path) -> None:
    require_free_space(tmp_path, 0)
    with pytest.raises(OSError, match="Insufficient free space"):
        require_free_space(tmp_path, shutil.disk_usage(tmp_path).free + 1)
    with pytest.raises(FileNotFoundError):
        require_free_space(tmp_path / "missing", 0)
    with pytest.raises(ValueError):
        require_free_space(tmp_path, -1)
