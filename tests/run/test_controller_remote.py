import hashlib
import json
import os
import signal
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import pytest

from pocketcontextagent.run.benchmarks.utils.controller_job import MODULE, SETUP_FAILED, read_status
from pocketcontextagent.run.benchmarks.utils.controller_remote import action, job_path
from pocketcontextagent.run.benchmarks.utils.controller_setup import extract_artifacts

SECRET = "controller-private-key-38192"
GATED = """
import os
import time
from pathlib import Path
print('progress:' + os.environ['TEST_API_KEY'], flush=True)
Path('ready').touch()
deadline = time.monotonic() + 60
while not Path('release').exists() and time.monotonic() < deadline:
    time.sleep(0.02)
Path('finished').touch()
"""


@pytest.fixture
def controller(tmp_path: Path):
    (tmp_path / "jobs").mkdir()
    originals = {"credentials.json": json.dumps({"TEST_API_KEY": SECRET}).encode(), "original.txt": b"unchanged"}
    for name, contents in originals.items():
        (tmp_path / name).write_bytes(contents)
    (tmp_path / "credentials.json").chmod(0o600)
    yield tmp_path
    assert {name: (tmp_path / name).read_bytes() for name in originals} == originals


@pytest.fixture
def processes(controller: Path):
    children = []
    yield children
    for job in (controller / "jobs").iterdir():
        if job.is_dir():
            (job / "release").touch()
    for child in children:
        if child.poll() is None:
            child.wait(timeout=10)


def request(root: Path, identifier: str, code: str = "print('done')") -> Path:
    job = job_path(root, identifier)
    job.mkdir()
    path = job / "request.json"
    path.write_text(
        json.dumps(
            {
                "argv": [sys.executable, "-c", code],
                "cwd": str(job),
                "secrets_path": str(root / "credentials.json"),
                "env": {"PUBLIC_MARKER": "visible"},
            }
        )
    )
    return path


def start(path: Path, processes: list) -> subprocess.Popen:
    child = subprocess.Popen([sys.executable, "-m", MODULE, str(path)])
    processes.append(child)
    return child


def wait_status(job: Path, state: str) -> dict:
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if (job / "status.json").exists():
            status = read_status(job)
            if status["state"] == state and (state == "running" or not status["runner_alive"]):
                return status
        time.sleep(0.02)
    pytest.fail(f"No {state} status for {job}")


@pytest.mark.parametrize(
    ("identifier",),  # noqa: PT006
    [("../escape",), ("a/../../escape",), ("/absolute",), ("..",), ("",), ("UPPER",), ("a\\b",), ("a" * 65,)],
)
def test_job_path_rejects_traversal_and_invalid_identifiers(tmp_path: Path, identifier: str) -> None:
    with pytest.raises(ValueError, match="Invalid job identifier"):
        job_path(tmp_path, identifier)
    with pytest.raises(ValueError, match="Invalid job identifier"):
        action("status", str(tmp_path), {"job": identifier})
    assert not list(tmp_path.iterdir())


def test_view_status_and_maintenance_use_live_runner_and_disabled_gate(controller: Path, processes: list) -> None:
    staged = request(controller, "staged")
    assert action("view", str(controller)) == {"disabled": False, "jobs": {"staged": {"state": "staged"}}}
    path = request(controller, "active", GATED)
    runner = start(path, processes)
    try:
        live = wait_status(path.parent, "running")
        viewed = action("view", str(controller))["jobs"]
        assert set(viewed) == {"staged", "active"}
        assert viewed["active"]["runner_pid"] == runner.pid
        assert viewed["active"]["runner_alive"] and viewed["active"]["child_alive"]
        assert live["child_pid"] == live["child_pgid"] != runner.pid
        assert action("status", str(controller), {"job": "active"})["state"] == "running"
        with pytest.raises(BlockingIOError):
            action("maintenance", str(controller))
        assert not (controller / "jobs" / "disabled").exists()
        with pytest.raises(ValueError, match="completion or cancel"):
            action("pack", str(controller), {"job": "active"})
    finally:
        (path.parent / "release").touch()
    assert runner.wait(timeout=10) == 0
    completed = action("status", str(controller), {"job": "active"})
    assert completed["state"] == "completed" and completed["exit_code"] == 0
    assert not completed["runner_alive"] and not completed["child_alive"]
    persisted = (path.parent / "status.json").read_bytes()
    assert action("maintenance", str(controller)) == {"disabled": True}
    assert action("view", str(controller))["disabled"] is True
    assert (controller / "jobs" / "disabled").exists()
    assert start(staged, processes).wait(timeout=10) == SETUP_FAILED
    disabled = action("status", str(controller), {"job": "staged"})
    assert disabled["state"] == "failed" and disabled["error_type"] == "PermissionError"
    assert disabled["child_pid"] is None and not (staged.parent / "run.log").exists()
    assert action("enable", str(controller)) == {"disabled": False}
    assert not (controller / "jobs" / "disabled").exists()
    enabled = request(controller, "enabled")
    assert start(enabled, processes).wait(timeout=10) == 0
    assert action("status", str(controller), {"job": "enabled"})["state"] == "completed"
    assert (path.parent / "status.json").read_bytes() == persisted


def test_maintenance_rejects_orphaned_child_even_after_runner_lock_releases(controller: Path, processes: list) -> None:
    path = request(controller, "orphan", GATED)
    runner = start(path, processes)
    try:
        wait_status(path.parent, "running")
        deadline = time.monotonic() + 15
        while not (path.parent / "ready").exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert (path.parent / "ready").exists()
        runner.kill()
        assert runner.wait(timeout=10) == -signal.SIGKILL
        original = (path.parent / "status.json").read_bytes()
        interrupted = action("status", str(controller), {"job": "orphan"})
        assert interrupted["state"] == "interrupted"
        assert interrupted["child_alive"] and not interrupted["runner_alive"]
        with pytest.raises(ValueError, match="active runner or orphaned child"):
            action("maintenance", str(controller))
        assert not (controller / "jobs" / "disabled").exists()
        assert (path.parent / "status.json").read_bytes() == original
    finally:
        (path.parent / "release").touch()
        deadline = time.monotonic() + 10
        while read_status(path.parent)["child_alive"] and time.monotonic() < deadline:
            time.sleep(0.02)
    assert not read_status(path.parent)["child_alive"]
    assert action("maintenance", str(controller)) == {"disabled": True}


@pytest.mark.skipif(os.name != "posix" or not Path("/bin/kill").is_file(), reason="Requires POSIX process groups")
def test_launch_acceptance_and_cancel_signal_only_owned_child_group(controller: Path) -> None:
    path = request(controller, "detached", GATED)
    accepted = action("launch", str(controller), {"job": "detached"})
    try:
        status = accepted["acceptance"]
        assert status["state"] == "running" and status["runner_pid"] == accepted["runner_pid"]
        assert status["runner_alive"] and status["child_alive"]
        assert status["child_pgid"] == status["child_pid"] != os.getpgrp()
        assert os.getpgid(accepted["runner_pid"]) == accepted["runner_pid"]
        cancelled = action("cancel", str(controller), {"job": "detached"})
        assert cancelled["state"] == "cancelled" and cancelled["exit_code"] == -signal.SIGTERM
        assert cancelled["finished_at"] and not cancelled["runner_alive"] and not cancelled["child_alive"]
        assert not (path.parent / "finished").exists()
        assert action("cancel", str(controller), {"job": "detached"}) == cancelled
    finally:
        (path.parent / "release").touch()
        deadline = time.monotonic() + 15
        while read_status(path.parent)["runner_alive"] and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not read_status(path.parent)["runner_alive"]


def test_redacted_logs_and_artifact_allowlist_extract_unchanged(controller: Path, processes: list) -> None:
    path = request(
        controller,
        "artifacts",
        "import os; print('public progress'); print(os.environ['TEST_API_KEY']); print('last line')",
    )
    assert start(path, processes).wait(timeout=10) == 0
    job = path.parent
    (job / "output" / "tasks").mkdir(parents=True)
    (job / "output" / "tasks" / "patch.diff").write_bytes(b"diff --git a/x b/x\n+fix\n")
    (job / "output" / "binary").write_bytes(bytes(range(256)))
    (job / "plan.json").write_text('{"attempts":4}')
    (job / "controller-cleanup.json").write_text('{"confirmed":true}')
    (job / "input").mkdir()
    for target in (job / "credentials.json", job / "input" / "grading_dataset.jsonl", job / "input" / "config.yaml"):
        target.write_text("private-grading-input-8391")
    (job / "launcher.log").write_text("private launcher details")
    assert action("logs", str(controller), {"job": "artifacts", "lines": 2})["log"] == "[REDACTED]\nlast line\n"
    allowed = {"output", "status.json", "plan.json", "run.log", "controller-cleanup.json"}
    expected = {
        str(item.relative_to(job)): item.read_bytes()
        for item in job.rglob("*")
        if item.is_file() and item.relative_to(job).parts[0] in allowed
    }
    packed = action("pack", str(controller), {"job": "artifacts"})
    archive = Path(packed["path"])
    assert packed["bytes"] == archive.stat().st_size
    assert packed["sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    with tarfile.open(archive) as bundle:
        assert {Path(item.name).parts[0] for item in bundle.getmembers()} == allowed
        assert all(item.isfile() or item.isdir() for item in bundle.getmembers())
    destination = controller / "collected"
    extract_artifacts(archive, destination)
    assert {
        str(item.relative_to(destination)): item.read_bytes() for item in destination.rglob("*") if item.is_file()
    } == expected
    assert not any(
        item.name in {"credentials.json", "grading_dataset.jsonl", "launcher.log", "input"}
        for item in destination.rglob("*")
    )
    assert all(
        SECRET.encode() not in contents and b"private-grading-input" not in contents for contents in expected.values()
    )
    assert action("journal", str(controller), {"job": "artifacts"}) == {}
    assert action("cleanup", str(controller), {"job": "artifacts"}) == {"confirmed": True, "journal": False}


def test_historical_logs_are_redacted_without_rewriting_files(controller: Path) -> None:
    path = request(controller, "historical")
    log = path.parent / "run.log"
    original = ("old progress\nhistorical:" + SECRET + "\n").encode()
    log.write_bytes(original)
    result = action("logs", str(controller), {"job": "historical", "lines": 1})
    assert result == {"log": "historical:[REDACTED]\n", "status": {"state": "staged"}}
    assert log.read_bytes() == original
    log.unlink()
    assert action("logs", str(controller), {"job": "historical"})["log"] == ""


def test_provider_check_rejects_control_plane_credential_before_http_or_filesystem(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Select a model provider credential"):
        action("check", str(tmp_path / "nonexistent"), {"provider_key": "DAYTONA_API_KEY", "model": {}})
    assert not list(tmp_path.iterdir())
    with pytest.raises(ValueError, match="root must be absolute"):
        action("view", "relative")
