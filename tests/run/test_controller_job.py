import fcntl
import hashlib
import json
import os
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

from pocketcontextagent.run.benchmarks.utils.controller_job import DUPLICATE, MODULE, SETUP_FAILED, read_status


@pytest.fixture
def processes():
    children = []
    yield children
    for process in children:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=10)


def request(tmp_path: Path, code: str, name: str = "job", **fields) -> Path:
    jobdir = tmp_path / name
    jobdir.mkdir()
    secrets_path = tmp_path / "credentials.json"
    if not secrets_path.exists():
        secrets_path.write_text(json.dumps({"TEST_API_KEY": "private-test-value-92381"}))
        secrets_path.chmod(0o600)
    path = jobdir / "request.json"
    path.write_text(
        json.dumps(
            {
                "argv": [sys.executable, "-c", code],
                "cwd": str(jobdir),
                "secrets_path": str(secrets_path),
                "env": {},
            }
            | fields
        )
    )
    return path


def start(path: Path, processes: list, **kwargs) -> subprocess.Popen:
    process = subprocess.Popen([sys.executable, "-m", MODULE, str(path)], **kwargs)
    processes.append(process)
    return process


def wait_status(jobdir: Path, state: str) -> dict:
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if (jobdir / "status.json").exists():
            status = read_status(jobdir)
            if status["state"] == state:
                return status
        time.sleep(0.02)
    pytest.fail(f"No {state} status in {jobdir}: {list(jobdir.iterdir())}")


def wait_file(path: Path) -> None:
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.02)
    pytest.fail(f"Missing file: {path}")


GATED = """
import time
from pathlib import Path
print('live progress', flush=True)
Path('started').touch()
while not Path('release').exists():
    time.sleep(0.02)
Path('finished').touch()
"""


def test_detached_runner_survives_launcher_death_and_keeps_private_logs(tmp_path, processes):
    path = request(tmp_path, GATED)
    launcher = subprocess.Popen(
        [
            sys.executable,
            "-c",
            f"from pathlib import Path; from {MODULE} import launch; import time; "
            "import sys; print(launch(Path(sys.argv[1])), flush=True); time.sleep(60)",
            str(path),
        ],
        stdout=subprocess.PIPE,
        text=True,
        env=dict(os.environ) | {"MSWEA_SILENT_STARTUP": "1"},
    )
    processes.append(launcher)
    assert launcher.stdout is not None
    runner_pid = int(launcher.stdout.readline())
    try:
        status = wait_status(path.parent, "running")
        wait_file(path.parent / "started")
        assert status["runner_pid"] == runner_pid
        assert status["runner_alive"] and status["child_alive"]
        assert status["child_pgid"] == status["child_pid"] != runner_pid
        assert os.getpgid(runner_pid) == runner_pid
        deadline = time.monotonic() + 10
        while "live progress" not in (path.parent / "run.log").read_text() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert "live progress" in (path.parent / "run.log").read_text()
        assert read_status(path.parent)["state"] == "running"
        launcher.kill()
        assert launcher.wait(timeout=10) == -signal.SIGKILL
        assert read_status(path.parent)["state"] == "running"
    finally:
        (path.parent / "release").touch()
    completed = wait_status(path.parent, "completed")
    assert completed["exit_code"] == 0 and completed["finished_at"] >= completed["started_at"]
    assert (path.parent / "finished").exists()
    for name in ["launcher.log", "run.log", "status.json", "claim"]:
        assert stat.S_IMODE((path.parent / name).stat().st_mode) == 0o600
    assert not (path.parent / "status.json.tmp").exists()


def test_claims_are_permanent_and_controller_lock_is_nonblocking(tmp_path, processes):
    first = request(tmp_path, GATED, "first")
    runner = start(first, processes)
    try:
        wait_status(first.parent, "running")
        assert start(first, processes).wait(timeout=10) == DUPLICATE
        assert read_status(first.parent)["runner_pid"] == runner.pid
        second = request(tmp_path, "from pathlib import Path; Path('unexpected').touch()", "second")
        assert start(second, processes).wait(timeout=10) == SETUP_FAILED
        busy = read_status(second.parent)
        assert busy["state"] == "failed" and busy["error_type"] == "BlockingIOError"
        assert busy["child_pid"] is None and busy["finished_at"] and not busy["runner_alive"]
        assert not (second.parent / "unexpected").exists()
    finally:
        (first.parent / "release").touch()
    assert runner.wait(timeout=10) == 0
    original = (first.parent / "status.json").read_bytes()
    assert start(first, processes).wait(timeout=10) == DUPLICATE
    assert (first.parent / "status.json").read_bytes() == original
    assert start(second, processes).wait(timeout=10) == DUPLICATE
    third = request(tmp_path, "print('next job')", "third")
    assert start(third, processes).wait(timeout=10) == 0
    assert read_status(third.parent)["state"] == "completed"


def test_disabled_controller_rejects_before_credentials_or_child_launch(tmp_path, processes):
    path = request(
        tmp_path,
        "from pathlib import Path; Path('unexpected').touch()",
        secrets_path=str(tmp_path / "missing-credentials.json"),
    )
    with (tmp_path / "active.lock").open("a") as active:
        fcntl.flock(active, fcntl.LOCK_EX)
        (tmp_path / "disabled").touch()
    assert start(path, processes).wait(timeout=10) == SETUP_FAILED
    status = read_status(path.parent)
    assert status["state"] == "failed" and status["error_type"] == "PermissionError"
    assert status["exit_code"] == SETUP_FAILED and status["child_pid"] is None
    assert status["finished_at"] and not status["runner_alive"]
    assert not (path.parent / "unexpected").exists() and not (path.parent / "run.log").exists()
    with (tmp_path / "active.lock").open("a") as active:
        fcntl.flock(active, fcntl.LOCK_EX)
        (tmp_path / "disabled").unlink()
    assert start(path, processes).wait(timeout=10) == DUPLICATE
    enabled = request(tmp_path, "print('enabled')", "enabled")
    assert start(enabled, processes).wait(timeout=10) == 0
    assert read_status(enabled.parent)["state"] == "completed"


def test_credentials_only_enter_child_environment_and_are_redacted(tmp_path, processes):
    secret = "private-test-value-92381"
    digest = hashlib.sha256(secret.encode()).hexdigest()
    code = f"""
import hashlib
import os
import sys
assert hashlib.sha256(os.environ['TEST_API_KEY'].encode()).hexdigest() == {digest!r}
assert os.environ['PUBLIC_MARKER'] == 'visible'
assert 'INHERITED_API_KEY' not in os.environ
assert os.environ['DAYTONA_API_URL'].startswith('https://')
assert len(os.environ['DAYTONA_TARGET']) == 14
print('child_env_ok=True', flush=True)
print('stdout:' + os.environ['TEST_API_KEY'], flush=True)
sys.stderr.write('stderr:' + os.environ['TEST_API_KEY'] + '\\n')
print(os.environ['DAYTONA_API_URL'], os.environ['DAYTONA_TARGET'], flush=True)
"""
    path = request(tmp_path, code, env={"PUBLIC_MARKER": "visible"})
    secrets = {
        "TEST_API_KEY": secret,
        "DAYTONA_API_URL": "https://private.invalid/api",
        "DAYTONA_TARGET": "private-region",
    }
    (tmp_path / "credentials.json").write_text(json.dumps(secrets))
    runner = start(path, processes, env=dict(os.environ) | {"INHERITED_API_KEY": "must-not-be-inherited"})
    assert runner.wait(timeout=10) == 0
    assert read_status(path.parent)["state"] == "completed"
    assert (path.parent / "run.log").read_text().count("[REDACTED]") == 4
    assert "child_env_ok=True" in (path.parent / "run.log").read_text()
    for artifact in path.parent.iterdir():
        assert all(value.encode() not in artifact.read_bytes() for value in secrets.values())
    assert stat.S_IMODE((path.parent / "run.log").stat().st_mode) == 0o600
    assert (tmp_path / "credentials.json").exists()


@pytest.mark.parametrize(
    ("fields", "error_type"),
    [
        ({"env": {"TEST_API_KEY": "override"}}, "ValueError"),
        ({"env": {"DAYTONA_API_URL": "override"}}, "ValueError"),
        ({"env": {"DAYTONA_TARGET": "override"}}, "ValueError"),
        ({"argv": ["/nonexistent/controller-test-executable"]}, "FileNotFoundError"),
        ({"argv": "not a list"}, "ValueError"),
        ({"cwd": "relative"}, "ValueError"),
        ({"secrets_path": "relative"}, "ValueError"),
    ],
)
def test_setup_failures_are_private_terminal_and_never_execute(tmp_path, processes, fields, error_type):
    path = request(tmp_path, "from pathlib import Path; Path('unexpected').touch()", **fields)
    runner = start(path, processes, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = runner.communicate(timeout=10)
    assert runner.returncode == SETUP_FAILED and b"override" not in stdout + stderr
    status = read_status(path.parent)
    assert status["state"] == "failed" and status["error_type"] == error_type
    assert status["exit_code"] == SETUP_FAILED and status["finished_at"]
    assert status["child_pid"] is None and not status["child_alive"]
    assert not (path.parent / "unexpected").exists()
    assert "override" not in (path.parent / "status.json").read_text()
    assert start(path, processes).wait(timeout=10) == DUPLICATE


@pytest.mark.parametrize(
    ("credential", "error_type"),
    [
        ({"TEST_API_KEY": "line-one\nline-two"}, "ValueError"),
        ({"TEST_API_KEY": "line-one\rline-two"}, "ValueError"),
        ({"UNSUPPORTED_SECRET": "value"}, "ValueError"),
        ({"TEST_API_KEY": 123}, "ValueError"),
        ("invalid json", "JSONDecodeError"),
    ],
)
def test_invalid_credentials_never_launch_or_leak(tmp_path, processes, credential, error_type):
    path = request(tmp_path, "from pathlib import Path; Path('unexpected').touch()")
    (tmp_path / "credentials.json").write_text(credential if isinstance(credential, str) else json.dumps(credential))
    assert start(path, processes).wait(timeout=10) == SETUP_FAILED
    status = read_status(path.parent)
    assert status["state"] == "failed" and status["error_type"] == error_type
    assert status["child_pid"] is None and not (path.parent / "unexpected").exists()
    assert "line-one" not in (path.parent / "status.json").read_text()


def test_request_paths_enforce_job_boundary_and_credentials_stay_external(tmp_path, processes):
    path = request(tmp_path, "pass", "inside", secrets_path=str(tmp_path / "inside" / "credentials.json"))
    (path.parent / "credentials.json").write_bytes((tmp_path / "credentials.json").read_bytes())
    assert start(path, processes).wait(timeout=10) == SETUP_FAILED
    assert read_status(path.parent)["error_type"] == "ValueError"
    descendant = request(tmp_path, "pass", "descendant", cwd=str(tmp_path / "descendant" / "child"))
    (descendant.parent / "child").mkdir()
    assert start(descendant, processes).wait(timeout=10) == SETUP_FAILED
    assert read_status(descendant.parent)["child_pid"] is None
    ancestor = request(tmp_path, "pass", "ancestor", cwd=str(tmp_path))
    assert start(ancestor, processes).wait(timeout=10) == 0
    assert read_status(ancestor.parent)["state"] == "completed"


@pytest.mark.parametrize(("exit_code", "state"), [(0, "completed"), (7, "failed"), (43, "failed")])
def test_exit_codes_and_merged_output_are_preserved(tmp_path, processes, exit_code, state):
    path = request(tmp_path, f"import sys; print('out'); print('err', file=sys.stderr); sys.exit({exit_code})")
    assert start(path, processes).wait(timeout=10) == exit_code
    status = read_status(path.parent)
    assert status["state"] == state
    assert status["exit_code"] == exit_code and not status["child_alive"]
    assert status["finished_at"] and "error_type" not in status
    assert sorted((path.parent / "run.log").read_text().splitlines()) == ["err", "out"]


def test_dead_runner_is_interrupted_not_completed_and_lock_is_released(tmp_path, processes):
    path = request(tmp_path, GATED)
    runner = start(path, processes)
    try:
        status = wait_status(path.parent, "running")
        wait_file(path.parent / "started")
        runner.kill()
        assert runner.wait(timeout=10) == -signal.SIGKILL
        interrupted = read_status(path.parent)
        assert interrupted["state"] == "interrupted" and not interrupted["runner_alive"]
        assert interrupted["child_alive"] and interrupted["exit_code"] is None
        assert json.loads((path.parent / "status.json").read_text())["state"] == "running"
        assert start(path, processes).wait(timeout=10) == DUPLICATE
        next_job = request(tmp_path, "pass", "next")
        assert start(next_job, processes).wait(timeout=10) == 0
    finally:
        (path.parent / "release").touch()
        wait_file(path.parent / "finished")
    assert status["child_pid"] == status["child_pgid"]


def test_signal_termination_records_cancelled_without_signalling_other_processes(tmp_path, processes):
    path = request(tmp_path, GATED)
    runner = start(path, processes)
    try:
        status = wait_status(path.parent, "running")
        wait_file(path.parent / "started")
        os.killpg(status["child_pgid"], signal.SIGTERM)
        assert runner.wait(timeout=10) == 256 - signal.SIGTERM
        cancelled = read_status(path.parent)
        assert cancelled["state"] == "cancelled" and cancelled["exit_code"] == -signal.SIGTERM
        assert cancelled["finished_at"] and not cancelled["child_alive"]
    finally:
        (path.parent / "release").touch()
