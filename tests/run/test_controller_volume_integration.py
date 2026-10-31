import hashlib
import json
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import traceback
from contextlib import closing
from pathlib import Path

import pytest
from click import unstyle
from typer.testing import CliRunner

from benchmarks.benchmark_controller import app
from benchmarks.utils.controller_artifacts import ArtifactStore
from benchmarks.utils.controller_job import _artifact_store
from benchmarks.utils.controller_remote import (
    _active,
    _archive_lock,
    _import_controller,
    action,
    archive_directory,
)
from benchmarks.utils.controller_setup import ControllerConfig, extract_artifacts, sandbox_params
from benchmarks.utils.controller_volumes import STORE_ROOT, VOLUME_MOUNT
from benchmarks.utils.swebench_daytona import (
    PreparedSnapshot,
    ResourceJournal,
    RuntimeConfig,
)
from benchmarks.utils.swebench_daytona import (
    sandbox_params as coding_sandbox_params,
)

PRIVATE = b"PRIVATE_INPUT_AND_CREDENTIAL_628349"
TASK = "output/baseline/tasks/case-1"


def write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def files(root: Path) -> dict[str, bytes]:
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def fingerprints(root: Path) -> dict[str, tuple[str, int, int]]:
    return {
        name: (hashlib.sha256(data).hexdigest(), len(data), (root / name).stat().st_mtime_ns)
        for name, data in files(root).items()
    }


@pytest.fixture
def legacy_job(tmp_path: Path) -> Path:
    job = tmp_path / "legacy-controller" / "jobs" / "legacy-1"
    task = job / TASK
    write(task / "case-1.json", b'{"answer":"original answer","exit_status":"Submitted"}\n')
    write(task / "run.yaml", b"seed: 42\nimplementation_sha256: original-benchmark-fingerprint\n")
    write(task / "worker.yaml", b"state: completed\ncleanup_confirmed: true\n")
    write(task / "trajectories" / "trajectory.json", b'{"messages":[{"role":"user","content":"issue"}]}\n')
    write(task / "trajectories" / "heavy.bin", bytes(range(256)) * 1024)
    with closing(sqlite3.connect(task / "trajectories" / "context.sqlite")) as database:
        database.execute("CREATE TABLE contexts (id INTEGER PRIMARY KEY, record TEXT, payload BLOB)")
        database.execute(
            "INSERT INTO contexts VALUES (?, ?, ?)", (1, '{"summary":"original λ context"}', b"\0\xff\r\n")
        )
        database.commit()
    with tarfile.open(task / "trajectories.tar.gz", "w:gz") as archive:
        archive.add(task / "trajectories", arcname="trajectories")
    write(job / "plan.json", b'{"seed":42,"implementation_sha256":"original-plan-fingerprint"}\n')
    write(job / "run.log", b"worker completed; credential=[REDACTED]\n")
    write(job / "output" / "baseline" / "summary.csv", b"task,answer\ncase-1,original answer\n")
    write(job / "status.json", b'{"state":"failed","exit_code":17}\n')
    for name in (
        "input/grading_dataset.jsonl",
        "credentials/keys.json",
        "request.json",
        "launcher.log",
        "output/input/gold.json",
        "output/credentials/keys.json",
        f"{TASK}/credentials/keys.json",
    ):
        write(job / name, PRIVATE)
    return job


@pytest.fixture
def store(tmp_path: Path) -> ArtifactStore:
    return ArtifactStore(tmp_path / "volume" / "artifacts-v1")


@pytest.fixture
def volume_record() -> dict:
    return {
        "id": "d8662458-761a-4ca0-ae66-4ed12990a306",
        "name": "controller-artifacts",
        "mount_path": VOLUME_MOUNT,
        "store_path": STORE_ROOT,
        "created": False,
    }


@pytest.mark.parametrize(("state", "exit_code"), [("failed", 17), ("completed", 0)])
def test_legacy_archive_roundtrip_preserves_benchmark_and_closed_sqlite(
    legacy_job: Path, store: ArtifactStore, tmp_path: Path, state: str, exit_code: int
) -> None:
    status = {"state": state, "exit_code": exit_code, "finished_at": "2026-01-01T00:00:00Z"}
    (legacy_job / "status.json").write_text(json.dumps(status))
    before = files(legacy_job)
    original_fingerprints = fingerprints(legacy_job)
    private_names = {name for name, data in before.items() if data == PRIVATE}
    expected = {name: data for name, data in before.items() if name not in private_names}
    expected["status.json"] = json.dumps(status, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    assert not _active(status)
    archived = archive_directory(store, legacy_job, "legacy-1", status)
    assert archived == status | {"archived_only": True, "runner_alive": False, "child_alive": False}
    assert status == {"state": state, "exit_code": exit_code, "finished_at": "2026-01-01T00:00:00Z"}
    task = legacy_job / TASK
    receipt = json.loads((task / "volume-artifacts.json").read_bytes())
    store.verify(receipt)
    assert receipt == store.task_receipt("legacy-1", "baseline", "case-1")
    for name in (
        "trajectories/context.sqlite",
        "trajectories/trajectory.json",
        "trajectories.tar.gz",
        "run.yaml",
        "case-1.json",
    ):
        data = before[f"{TASK}/{name}"]
        assert receipt["members"][name] == {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    assert not (task / "trajectories").exists() and not (task / "trajectories.tar.gz").exists()
    assert files(legacy_job) == {
        name: data
        for name, data in before.items()
        if not name.startswith(f"{TASK}/trajectories/") and name != f"{TASK}/trajectories.tar.gz"
    } | {f"{TASK}/volume-artifacts.json": (task / "volume-artifacts.json").read_bytes()}
    assert {name: value for name, value in fingerprints(legacy_job).items() if name in original_fingerprints} == {
        name: value
        for name, value in original_fingerprints.items()
        if not name.startswith(f"{TASK}/trajectories/") and name != f"{TASK}/trajectories.tar.gz"
    }
    expected[f"{TASK}/volume-artifacts.json"] = (task / "volume-artifacts.json").read_bytes()
    local_after, remote_after = fingerprints(legacy_job), fingerprints(store.root)
    assert archive_directory(store, legacy_job, "legacy-1", status) == archived
    assert fingerprints(legacy_job) == local_after and fingerprints(store.root) == remote_after
    shutil.rmtree(legacy_job.parent.parent)
    fresh = ArtifactStore(store.root)
    assert fresh.job_status("legacy-1") == archived
    exported = fresh.export_job("legacy-1", tmp_path / "restored.tar.gz")
    assert exported == {
        "path": str(tmp_path / "restored.tar.gz"),
        "bytes": (tmp_path / "restored.tar.gz").stat().st_size,
        "sha256": hashlib.sha256((tmp_path / "restored.tar.gz").read_bytes()).hexdigest(),
    }
    destination = tmp_path / "restored"
    extract_artifacts(Path(exported["path"]), destination)
    assert files(destination) == expected
    assert not private_names & files(destination).keys()
    assert PRIVATE not in b"".join(files(destination).values())
    with closing(sqlite3.connect(destination / TASK / "trajectories" / "context.sqlite")) as database:
        assert database.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
        assert database.execute("SELECT * FROM contexts").fetchall() == [
            (1, '{"summary":"original λ context"}', b"\0\xff\r\n")
        ]
    extract_artifacts(destination / TASK / "trajectories.tar.gz", tmp_path / "nested")
    assert files(tmp_path / "nested") == {
        name.removeprefix(f"{TASK}/"): data for name, data in before.items() if name.startswith(f"{TASK}/trajectories/")
    }


def test_imported_export_republishes_receipts_into_independent_store(
    legacy_job: Path, store: ArtifactStore, tmp_path: Path
) -> None:
    original = files(legacy_job)
    status = {"state": "failed", "exit_code": 17}
    archived = archive_directory(store, legacy_job, "legacy-1", status)
    source_bytes, source_fingerprints = files(store.root), fingerprints(store.root)
    shutil.rmtree(legacy_job.parent.parent)
    fresh_source = ArtifactStore(store.root)
    fresh_source.verify_job("legacy-1")
    exported = fresh_source.export_job("legacy-1", tmp_path / "source-export.tar.gz")
    staging = tmp_path / "verified-import"
    extract_artifacts(Path(exported["path"]), staging)
    expected = files(staging)
    source_receipt = json.loads((staging / TASK / "volume-artifacts.json").read_bytes())
    target = ArtifactStore(tmp_path / "independent-volume" / "artifacts-v1")
    assert files(target.root) == {}
    assert (
        archive_directory(
            target, staging, "legacy-1", json.loads((staging / "status.json").read_bytes()), imported=True
        )
        == archived
    )
    target_receipt = target.task_receipt("legacy-1", "baseline", "case-1")
    assert target_receipt["members"] == source_receipt["members"]
    assert json.loads((staging / TASK / "volume-artifacts.json").read_bytes()) == target_receipt
    assert not (staging / TASK / "trajectories").exists()
    assert not (staging / TASK / "trajectories.tar.gz").exists()
    shutil.rmtree(staging)
    fresh_target = ArtifactStore(target.root)
    fresh_target.verify_job("legacy-1")
    restored_export = fresh_target.export_job("legacy-1", tmp_path / "target-export.tar.gz")
    restored = tmp_path / "target-restored"
    extract_artifacts(Path(restored_export["path"]), restored)
    assert files(restored) == expected
    for name, data in original.items():
        if data != PRIVATE and name != "status.json":
            assert (restored / name).read_bytes() == data
            assert fingerprints(restored)[name][:2] == (hashlib.sha256(data).hexdigest(), len(data))
    assert json.loads((restored / "status.json").read_bytes()) == status
    assert PRIVATE not in b"".join(files(restored).values())
    assert files(store.root) == source_bytes
    assert fingerprints(store.root) == source_fingerprints


def test_archive_verification_failure_keeps_heavy_files_and_repair_never_reruns_work(
    legacy_job: Path, store: ArtifactStore
) -> None:
    receipt = store.archive_task(legacy_job / TASK, "legacy-1", "baseline", "case-1", prune=False)
    before = files(legacy_job)
    immutable = fingerprints(legacy_job)
    object_path = store.root / receipt["object"]["path"]
    original_object = object_path.read_bytes()
    object_path.write_bytes(b"corrupt volume object")
    with pytest.raises(ValueError, match="Corrupt artifact object"):
        archive_directory(store, legacy_job, "legacy-1", {"state": "failed", "exit_code": 17})
    assert files(legacy_job) == before and fingerprints(legacy_job) == immutable
    assert not (store.root / "jobs" / "legacy-1" / "job.json").exists()
    object_path.write_bytes(original_object)
    assert archive_directory(store, legacy_job, "legacy-1", {"state": "failed", "exit_code": 17})["exit_code"] == 17
    store.verify(receipt)
    assert not (legacy_job / TASK / "trajectories").exists()
    assert not (legacy_job / TASK / "trajectories.tar.gz").exists()
    assert (legacy_job / TASK / "case-1.json").read_bytes() == before[f"{TASK}/case-1.json"]
    assert (legacy_job / TASK / "run.yaml").read_bytes() == before[f"{TASK}/run.yaml"]


@pytest.mark.parametrize(("damage", "error"), [("corrupt", ValueError), ("missing", FileNotFoundError)])
@pytest.mark.parametrize(("operation",), [("archive",), ("import",)])  # noqa: PT006
def test_existing_archive_requires_complete_task_payload_before_idempotent_success(
    legacy_job: Path, store: ArtifactStore, damage: str, error: type[Exception], operation: str
) -> None:
    status = {"state": "failed", "exit_code": 17}
    receipt = store.archive_task(legacy_job / TASK, "legacy-1", "baseline", "case-1", prune=False)
    store.finalize_job(legacy_job, "legacy-1", status)
    store.verify_job("legacy-1", local=legacy_job)
    object_path = store.root / receipt["object"]["path"]
    original_object = object_path.read_bytes()
    if damage == "corrupt":
        object_path.write_bytes(b"corrupt committed task object")
    else:
        object_path.unlink()
    fresh = ArtifactStore(store.root)
    archived = status | {"archived_only": True, "runner_alive": False, "child_alive": False}
    assert fresh.job_status("legacy-1") == archived
    local_before, store_before = fingerprints(legacy_job), fingerprints(store.root)
    with pytest.raises(error):
        fresh.verify_job("legacy-1", local=legacy_job)
    assert fingerprints(legacy_job) == local_before
    assert fingerprints(store.root) == store_before
    if operation == "archive":
        with pytest.raises(error) as failure:
            archive_directory(fresh, legacy_job, "legacy-1", status)
    else:
        with pytest.raises(error) as failure:
            _import_controller(legacy_job.parent.parent, fresh, {"job": "legacy-1"})
    if damage == "missing":
        assert failure.value.filename == str(object_path)
    else:
        assert "Corrupt artifact object" in str(failure.value)
    assert fingerprints(legacy_job) == local_before
    assert fingerprints(store.root) == store_before
    assert (legacy_job / TASK / "trajectories" / "context.sqlite").is_file()
    assert (legacy_job / TASK / "trajectories.tar.gz").is_file()
    object_path.write_bytes(original_object)
    fresh.verify_job("legacy-1", local=legacy_job)
    assert archive_directory(fresh, legacy_job, "legacy-1", status) == archived
    assert _import_controller(legacy_job.parent.parent, fresh, {"job": "legacy-1"}) == archived
    assert fingerprints(legacy_job) == local_before


def test_existing_archive_rejects_late_recovered_output_without_pruning_or_replacing_commit(
    store: ArtifactStore, tmp_path: Path
) -> None:
    job = tmp_path / "failed-job"
    task = job / TASK
    status = {"state": "failed", "exit_code": 17}
    write(task / "worker.yaml", b"phase: failed\nretain: true\n")
    store.finalize_job(job, "legacy-1", status)
    store.verify_job("legacy-1", local=job)
    old_export = store.export_job("legacy-1", tmp_path / "before-recovery.tar.gz")
    extract_artifacts(Path(old_export["path"]), tmp_path / "before-recovery")
    original_export = files(tmp_path / "before-recovery")
    assert set(original_export) == {"status.json", f"{TASK}/worker.yaml"}
    write(task / "case-1.json", b'{"answer":"recovered after failed archive"}\n')
    write(task / "trajectories" / "trace.json", b'{"messages":["recovered trace"]}\n')
    with tarfile.open(task / "trajectories.tar.gz", "w:gz") as archive:
        archive.add(task / "trajectories", arcname="trajectories")
    local_before, store_before = fingerprints(job), fingerprints(store.root)
    store.verify_job("legacy-1")
    with pytest.raises(ValueError, match="Local output differs"):
        store.verify_job("legacy-1", local=job)
    with pytest.raises(ValueError, match="Local output differs"):
        archive_directory(store, job, "legacy-1", status)
    assert fingerprints(job) == local_before
    assert fingerprints(store.root) == store_before
    assert (task / "case-1.json").is_file()
    assert (task / "trajectories" / "trace.json").is_file()
    assert (task / "trajectories.tar.gz").is_file()
    assert not (task / "volume-artifacts.json").exists()
    retained_export = store.export_job("legacy-1", tmp_path / "after-recovery.tar.gz")
    extract_artifacts(Path(retained_export["path"]), tmp_path / "after-recovery")
    assert files(tmp_path / "after-recovery") == original_export
    assert fingerprints(job) == local_before
    assert fingerprints(store.root) == store_before


def test_journal_failure_persistence_error_preserves_previously_retained_worker(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.json"
    journal = ResourceJournal(ledger)
    journal.record("snapshots", "owned-snapshot", phase="active")
    journal.record(
        "sandboxes", "owned-worker", id="worker-id", snapshot="owned-snapshot", phase="running_agent", retain=True
    )
    previous = ledger.read_bytes()
    ledger_fingerprint = fingerprints(tmp_path)["ledger.json"]
    obstruction = tmp_path / "ledger.json.tmp"
    obstruction.mkdir()
    error = RuntimeError("artifact persistence failed before verified capture")
    with pytest.raises(IsADirectoryError):
        journal.failure("sandboxes", "owned-worker", error)
    assert ledger.read_bytes() == previous
    assert fingerprints(tmp_path)["ledger.json"] == ledger_fingerprint
    for document in (journal.document(), json.loads(ledger.read_bytes())):
        assert document["sandboxes"][0]["retain"] is True
        assert document["sandboxes"][0]["snapshot"] == "owned-snapshot"
        assert document["sandboxes"][0]["id"] == "worker-id"
        assert document["snapshots"] == [{"name": "owned-snapshot", "phase": "active"}]
    obstruction.rmdir()
    assert journal.failure("sandboxes", "owned-worker", error) == {"error_type": "RuntimeError", "error": str(error)}
    assert json.loads(ledger.read_bytes()) == journal.document()
    assert journal.document()["sandboxes"][0]["retain"] is True


def test_archive_lock_excludes_real_child_and_releases_without_changing_job(legacy_job: Path) -> None:
    root = legacy_job.parent.parent
    (root / "jobs" / "active.lock").touch()
    before = fingerprints(root)
    command = [
        sys.executable,
        "-c",
        "from pathlib import Path\n"
        "import sys\n"
        "from benchmarks.utils.controller_remote import _archive_lock\n"
        "with _archive_lock(Path(sys.argv[1])):\n"
        "    print('archive lock acquired')\n",
        str(root),
    ]
    with _archive_lock(root):
        rejected = subprocess.run(command, capture_output=True, text=True, timeout=15, check=False)
        assert rejected.returncode != 0 and "BlockingIOError" in rejected.stderr
        assert "archive lock acquired" not in rejected.stdout
    accepted = subprocess.run(command, capture_output=True, text=True, timeout=15, check=False)
    assert accepted.returncode == 0, accepted.stderr
    assert "archive lock acquired" in accepted.stdout
    assert fingerprints(root) == before


def test_maintenance_gate_denies_archive_admission_except_locked_internal_cleanup(tmp_path: Path) -> None:
    root = tmp_path / "controller"
    (root / "jobs").mkdir(parents=True)
    write(root / "retained.json", b'{"must":"remain unchanged"}\n')
    assert action("maintenance", str(root)) == {"disabled": True}
    before = fingerprints(root)
    with pytest.raises(ValueError, match="maintenance; archive admission is disabled"):
        _archive_lock(root).__enter__()
    command = [
        sys.executable,
        "-c",
        "from pathlib import Path\n"
        "import sys\n"
        "from benchmarks.utils.controller_remote import _archive_lock\n"
        "with _archive_lock(Path(sys.argv[1])):\n"
        "    print('external archive admitted')\n",
        str(root),
    ]
    with _archive_lock(root, allow_maintenance=True):
        assert (root / "jobs" / "disabled").exists()
        raced = subprocess.run(command, capture_output=True, text=True, timeout=15, check=False)
        assert raced.returncode != 0 and "BlockingIOError" in raced.stderr
        assert "external archive admitted" not in raced.stdout
    denied = subprocess.run(command, capture_output=True, text=True, timeout=15, check=False)
    assert denied.returncode != 0 and "maintenance; archive admission is disabled" in denied.stderr
    assert "external archive admitted" not in denied.stdout
    assert fingerprints(root) == before
    assert action("enable", str(root)) == {"disabled": False}
    with _archive_lock(root):
        assert not (root / "jobs" / "disabled").exists()
    assert fingerprints(root) == {name: value for name, value in before.items() if name != "jobs/disabled"}


@pytest.mark.parametrize(("workload_state", "exit_code"), [("failed", 23), ("completed", 0)])
def test_archive_failed_recovers_original_workload_status(
    legacy_job: Path, store: ArtifactStore, workload_state: str, exit_code: int
) -> None:
    status = {
        "state": "archive_failed",
        "workload_state": workload_state,
        "exit_code": exit_code,
        "archive_state": "failed",
        "error_type": "OSError",
    }
    (legacy_job / "status.json").write_text(json.dumps(status))
    before = (legacy_job / "status.json").read_bytes()
    archived = archive_directory(store, legacy_job, "legacy-1", status)
    assert archived == status | {
        "state": workload_state,
        "archive_state": "complete",
        "recovered_archive_failure": True,
        "archived_only": True,
        "runner_alive": False,
        "child_alive": False,
    }
    assert status["state"] == "archive_failed"
    assert (legacy_job / "status.json").read_bytes() == before
    assert json.loads(store.read_job_file("legacy-1", "status.json"))["exit_code"] == exit_code
    assert archive_directory(store, legacy_job, "legacy-1", status) == archived


@pytest.mark.parametrize(
    ("status",),  # noqa: PT006
    [
        ({"state": "running"},),
        ({"state": "starting"},),
        ({"state": "completed", "runner_alive": True},),
        ({"state": "failed", "child_alive": True},),
        ({"state": "archive_failed", "workload_state": "failed", "child_alive": True},),
    ],
)
def test_active_archive_rejected_before_writes(legacy_job: Path, store: ArtifactStore, status: dict) -> None:
    before = fingerprints(legacy_job)
    with pytest.raises(ValueError, match="inactive terminal"):
        archive_directory(store, legacy_job, "legacy-1", status)
    assert fingerprints(legacy_job) == before
    assert files(store.root) == {}


def test_only_controller_params_mount_real_volume_and_coding_has_no_secrets(volume_record: dict) -> None:
    daytona = pytest.importorskip("daytona")
    controller = sandbox_params("trusted-snapshot", "controller", ControllerConfig(), volume=volume_record)
    worker = coding_sandbox_params(
        PreparedSnapshot("coding-snapshot", "image@sha256:" + "a" * 64), "baseline", RuntimeConfig()
    )
    legacy = sandbox_params("trusted-snapshot", "legacy-controller", ControllerConfig(), volume=None)
    assert all(isinstance(params, daytona.CreateSandboxFromSnapshotParams) for params in (controller, worker, legacy))
    assert len(controller.volumes) == 1 and isinstance(controller.volumes[0], daytona.VolumeMount)
    assert controller.volumes[0].model_dump(exclude_unset=True, by_alias=True) == {
        "volumeId": volume_record["id"],
        "mountPath": VOLUME_MOUNT,
    }
    assert worker.volumes is None and legacy.volumes is None
    assert worker.network_block_all is True and controller.network_block_all is False
    for params in (controller, worker, legacy):
        assert not {"secretEnv", "secret_env", "secrets"} & params.model_dump(by_alias=True, exclude_none=True).keys()
        assert not {"secretEnv", "secret_env", "secrets", "additional_properties"} & params.model_fields_set
        assert not any(
            key.endswith("_API_KEY") or key.startswith(("DAYTONA_", "MSWEA_ARTIFACT_")) for key in params.env_vars
        )
        assert PRIVATE.decode() not in params.model_dump_json()
    assert volume_record["id"] not in worker.model_dump_json() + legacy.model_dump_json()


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        (["--help"], ("volume-info", "archive", "import-results", "volume-delete")),
        (["create", "--help"], ("--volume-name",)),
        (["volume-delete", "--help"], ("--name", "--confirm-destroy-data")),
        (["import-results", "--help"], ("--source-controller", "--job", "--max-bytes")),
    ],
)
def test_volume_commands_and_flags_have_offline_help(arguments: list[str], expected: tuple[str, ...]) -> None:
    result = CliRunner().invoke(app, arguments, color=False, env={"COLUMNS": "160"})
    assert result.exit_code == 0, result.output
    assert all(text in unstyle(result.output) for text in expected)


def test_volume_delete_without_destructive_confirmation_fails_before_client() -> None:
    result = CliRunner().invoke(app, ["volume-delete", "--name", "controller-artifacts"])
    assert result.exit_code != 0 and isinstance(result.exception, ValueError)
    assert "--confirm-destroy-data" in str(result.exception)
    frames = traceback.extract_tb(result.exc_info[2])
    assert frames[-1].name == "volume_delete"
    assert not any(frame.name == "client" or "daytona" in frame.filename for frame in frames)


@pytest.mark.parametrize(("record",), [(None,), (b"not json",), (b"[]",), (b"{}",), (b'{"store_path":"/"}',)])  # noqa: PT006
def test_missing_or_bad_volume_record_fails_closed_without_root_fallback(
    legacy_job: Path, record: bytes | None
) -> None:
    pytest.importorskip("daytona")
    root = legacy_job.parent.parent
    if record is not None:
        (root / "volume.json").write_bytes(record)
    before = fingerprints(root)
    with pytest.raises(ValueError):
        action("archive", str(root), {"job": "legacy-1"})
    assert fingerprints(root) == before
    assert not (root / "artifacts-v1").exists()
    assert not list(root.rglob("volume-artifacts.json"))


def test_unmanaged_temp_volume_record_and_unmounted_runner_store_never_create_fallback(
    legacy_job: Path, volume_record: dict, tmp_path: Path
) -> None:
    pytest.importorskip("daytona")
    root = legacy_job.parent.parent
    mount = tmp_path / "ordinary-directory-not-fuse"
    mount.mkdir()
    record = volume_record | {"mount_path": str(mount), "store_path": str(mount / "artifacts-v1")}
    (root / "volume.json").write_text(json.dumps(record))
    before = fingerprints(root)
    with pytest.raises(ValueError, match="managed artifact mount"):
        action("archive", str(root), {"job": "legacy-1"})
    with pytest.raises(ValueError, match="refusing root-disk storage"):
        _artifact_store(
            {
                "MSWEA_ARTIFACT_STORE_ROOT": record["store_path"],
                "MSWEA_ARTIFACT_JOB_ID": "legacy-1",
                "MSWEA_ARTIFACT_MIN_FREE_BYTES": "1",
            },
            legacy_job,
        )
    assert fingerprints(root) == before
    assert list(mount.iterdir()) == []
