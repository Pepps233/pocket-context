"""POSIX, trusted-controller jobs; no retries, providers, or model dependencies.

Run with ``python -m pocketcontextagent.run.benchmarks.utils.controller_job /abs/job/request.json``.
``launch`` detaches that runner and returns its PID, not the workload's exit code.
Requests contain argv, cwd (job directory or an ancestor), secrets_path (outside
that directory), and public env. Credential values must be single-line strings.
Child output is merged and credential-redacted into a private, line-buffered log.
"""

import fcntl
import json
import os
import subprocess
import sys
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO

MODULE = "pocketcontextagent.run.benchmarks.utils.controller_job"
SETUP_FAILED = 125
DUPLICATE = 73


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _private(path: Path, flags: int) -> BinaryIO:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | flags, 0o600)
    os.fchmod(fd, 0o600)
    return os.fdopen(fd, "wb")


def _save(jobdir: Path, status: dict) -> None:
    with _private(jobdir / "status.json.tmp", os.O_TRUNC) as stream:
        stream.write(json.dumps(status).encode())
        stream.flush()
        os.fsync(stream.fileno())
    (jobdir / "status.json.tmp").replace(jobdir / "status.json")
    fd = os.open(jobdir, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _secret_key(key: str) -> bool:
    return key.endswith("_API_KEY") or key in {"DAYTONA_API_URL", "DAYTONA_TARGET"}


def _environment(request: dict, jobdir: Path) -> tuple[dict[str, str], list[bytes]]:
    if set(request) != {"argv", "cwd", "secrets_path", "env"}:
        raise ValueError("Invalid request fields")
    if not isinstance(request["argv"], list) or not request["argv"]:
        raise ValueError("argv must be a nonempty list")
    if any(not isinstance(arg, str) or "\0" in arg for arg in request["argv"]):
        raise ValueError("Invalid argv")
    for key in ("cwd", "secrets_path"):
        if not isinstance(request[key], str) or not Path(request[key]).is_absolute():
            raise ValueError("Paths must be absolute")
    if Path(request["cwd"]).resolve() not in (jobdir, *jobdir.parents):
        raise ValueError("cwd must be the job directory or an ancestor")
    secrets_path = Path(request["secrets_path"]).resolve()
    if secrets_path.is_relative_to(jobdir):
        raise ValueError("Credentials must be outside the job directory")
    public = request["env"]
    if not isinstance(public, dict) or any(
        not isinstance(key, str)
        or not isinstance(value, str)
        or not key
        or "=" in key
        or "\0" in key + value
        or _secret_key(key)
        for key, value in public.items()
    ):
        raise ValueError("Invalid public environment")
    secrets = json.loads(secrets_path.read_text())
    if not isinstance(secrets, dict) or any(
        not isinstance(key, str)
        or not _secret_key(key)
        or not isinstance(value, str)
        or not value
        or any(char in key + value for char in "\0\r\n")
        or "=" in key
        for key, value in secrets.items()
    ):
        raise ValueError("Invalid credentials")
    if any(value in arg for value in secrets.values() for arg in request["argv"]):
        raise ValueError("Credentials must not appear in argv")
    environment = {key: value for key, value in os.environ.items() if not _secret_key(key)}
    return environment | public | secrets, sorted({value.encode() for value in secrets.values()}, key=len, reverse=True)


def _process(pid: int | None) -> dict | None:
    if pid is None:
        return None
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "lstart=", "-o", "stat=", "-o", "pgid="],
        capture_output=True,
        text=True,
        env={"PATH": os.defpath, "LC_ALL": "C"},
    )
    fields = result.stdout.split()
    if result.returncode or len(fields) != 7 or fields[5].startswith("Z"):
        return None
    return {"birth": " ".join(fields[:5]), "pgid": int(fields[6])}


def read_status(jobdir: Path) -> dict:
    """Return persisted results, or an interrupted *view* if the runner died.

    A claim lock proves ownership independently of PID reuse or machine restarts.
    Reading never rewrites status or invents a successful exit code.
    """
    status = json.loads((jobdir / "status.json").read_text())
    with (jobdir / "claim").open("rb") as claim:
        try:
            fcntl.flock(claim, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            status["runner_alive"] = True
        else:
            status["runner_alive"] = False
    child = _process(status["child_pid"])
    status["child_alive"] = child is not None and child == status.get("child_identity")
    if status["state"] in {"starting", "running"} and not status["runner_alive"]:
        status["state"] = "interrupted"
    return status


def run(request_path: Path) -> int:
    """Claim once, take the controller-wide lock, and synchronously supervise."""
    if not request_path.is_absolute():
        return SETUP_FAILED
    jobdir = request_path.resolve().parent
    try:
        claim = _private(jobdir / "claim", os.O_EXCL)
    except FileExistsError:
        return DUPLICATE
    with claim, ExitStack() as stack:
        fcntl.flock(claim, fcntl.LOCK_EX)
        status = {
            "state": "starting",
            "runner_pid": os.getpid(),
            "child_pid": None,
            "child_pgid": None,
            "child_identity": None,
            "started_at": _now(),
            "finished_at": None,
            "exit_code": None,
        }
        _save(jobdir, status)
        try:
            active = stack.enter_context(_private(jobdir.parent / "active.lock", 0))
            fcntl.flock(active, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if (jobdir.parent / "disabled").exists():
                raise PermissionError("Controller is disabled")
            request = json.loads(request_path.read_text())
            environment, secrets = _environment(request, jobdir)
            log = stack.enter_context(_private(jobdir / "run.log", os.O_TRUNC))
            child = subprocess.Popen(
                request["argv"],
                cwd=request["cwd"],
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except (OSError, ValueError, TypeError, KeyError) as error:
            status.update(state="failed", finished_at=_now(), exit_code=SETUP_FAILED, error_type=type(error).__name__)
            _save(jobdir, status)
            return SETUP_FAILED
        status.update(state="running", child_pid=child.pid, child_pgid=child.pid, child_identity=_process(child.pid))
        _save(jobdir, status)
        assert child.stdout is not None
        with child.stdout:
            for line in child.stdout:
                for value in secrets:
                    line = line.replace(value, b"[REDACTED]")
                log.write(line)
                log.flush()
        exit_code = child.wait()
        os.fsync(log.fileno())
        status.update(
            state="completed" if exit_code == 0 else "cancelled" if exit_code < 0 else "failed",
            finished_at=_now(),
            exit_code=exit_code,
        )
        _save(jobdir, status)
        return exit_code


def launch(request: Path) -> int:
    """Return a detached runner PID; invoke this helper in a short-lived launcher.

    Poll status.json for acceptance/results; the returned PID alone is not proof
    of a successful claim. The interpreter/package must already exist remotely.
    """
    if not request.is_absolute():
        raise ValueError("Request path must be absolute")
    with _private(request.parent / "launcher.log", os.O_APPEND) as log:
        return subprocess.Popen(
            [sys.executable, "-m", MODULE, str(request)],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        ).pid


if __name__ == "__main__":
    sys.exit(run(Path(sys.argv[1])) if len(sys.argv) == 2 else SETUP_FAILED)
