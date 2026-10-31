"""Run the pinned official SWE-bench tests in fresh Daytona sandboxes."""

import hashlib
import json
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from pathlib import Path
from shlex import quote
from typing import TYPE_CHECKING
from uuid import uuid4

import typer
from dotenv import load_dotenv

from pocketcontextagent.environments.swebench_verified import SWEBenchVerifiedEnvironment
from pocketcontextagent.run.benchmarks.utils.swebench_daytona import (
    ResourceJournal,
    RuntimeConfig,
    create_snapshot,
    delete_resources,
    sandbox_metadata,
    sandbox_params,
)
from pocketcontextagent.run.benchmarks.utils.swebench_verified import HARNESS_REVISION, prediction, public_task

if TYPE_CHECKING:
    from daytona import Sandbox
    from swebench.harness.test_spec.test_spec import TestSpec

app = typer.Typer(add_completion=False)

HTTPBIN_REQUIREMENTS = """attrs==26.1.0 blinker==1.9.0 brotlicffi==1.2.1.0 cffi==2.1.1
click==8.5.0 decorator==5.3.1 flasgger==0.9.7.1 Flask==3.1.3 greenlet==2.0.2
gunicorn==23.0.0 httpbin==0.10.2 itsdangerous==2.2.0 Jinja2==3.1.6 jsonschema==4.26.0
jsonschema-specifications==2025.9.1 MarkupSafe==3.0.4 mistune==3.4.0 packaging==26.3
pycparser==3.12 PyYAML==6.0.3 referencing==0.37.0 rpds-py==2026.9.1 six==1.17.0
typing_extensions==4.16.0 Werkzeug==3.1.9""".split()


def prepare_httpbin(sandbox: "Sandbox", instance: dict, directory: Path) -> str:
    """Provide the legacy tests' supported external fixture without public network flakiness."""
    if instance["repo"] != "psf/requests" or instance["version"] not in {"1.1", "2.0"}:
        return ""
    setup = (
        "set -eu\n"
        "test -f /testbed/requests/cacert.pem && ! test -L /testbed/requests/cacert.pem\n"
        "/opt/miniconda3/bin/python -m venv /tmp/pca-httpbin-env\n"
        f"/tmp/pca-httpbin-env/bin/pip install --disable-pip-version-check {' '.join(HTTPBIN_REQUIREMENTS)}\n"
        "openssl req -x509 -newkey rsa:2048 -nodes -days 2 -subj /CN=localhost "
        "-addext subjectAltName=DNS:localhost,IP:127.0.0.1 "
        "-keyout /tmp/pca-httpbin.key -out /tmp/pca-httpbin.crt\n"
        "sha256sum /testbed/requests/cacert.pem\n"
        "cat /tmp/pca-httpbin.crt >> /testbed/requests/cacert.pem\n"
        "sha256sum /testbed/requests/cacert.pem\n"
        "/tmp/pca-httpbin-env/bin/gunicorn -D -b 127.0.0.1:80 --workers 2 "
        "--pid /tmp/pca-httpbin-http.pid --error-logfile /tmp/pca-httpbin-http.log httpbin:app\n"
        "/tmp/pca-httpbin-env/bin/gunicorn -D -b 127.0.0.1:443 --workers 2 "
        "--pid /tmp/pca-httpbin-https.pid --error-logfile /tmp/pca-httpbin-https.log "
        "--certfile /tmp/pca-httpbin.crt --keyfile /tmp/pca-httpbin.key httpbin:app\n"
        "/tmp/pca-httpbin-env/bin/pip freeze\n"
    )
    (directory / "httpbin-setup.sh").write_text(setup)
    response = sandbox.process.exec("/bin/bash -c " + quote(setup), cwd="/tmp", timeout=300)
    (directory / "httpbin-setup.log").write_text(response.result)
    if response.exit_code != 0:
        raise RuntimeError("Local httpbin fixture setup failed; see httpbin-setup.log")
    sandbox.fs.download_file("/tmp/pca-httpbin.crt", str(directory / "httpbin-ca.pem"))
    (directory / "httpbin-provenance.json").write_text(
        json.dumps(
            {
                "requirements": HTTPBIN_REQUIREMENTS,
                "HTTPBIN_URL": "http://localhost/",
                "certificate_verification": True,
                "trust_adjustment": "Append the isolated localhost fixture CA to requests/cacert.pem",
                "official_tests_modified": False,
            },
            indent=2,
        )
    )
    return "HTTPBIN_URL=http://localhost/ NO_PROXY=localhost,127.0.0.1 "


def verify_harness() -> None:
    import swebench

    checkout = Path(swebench.__file__).resolve().parent.parent
    if (
        version("swebench") != "4.1.0"
        or subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
        != HARNESS_REVISION
    ):
        raise ValueError("Install the official editable SWE-bench checkout at the pinned v4.1.0 revision")
    subprocess.run(["git", "-C", str(checkout), "diff", "--exit-code", "HEAD", "--", "swebench"], check=True)


def load_inputs(
    dataset: Path, predictions: Path, dataset_sha256: str, predictions_sha256: str
) -> list[tuple[dict, dict]]:
    contents = []
    for path, expected in ((dataset, dataset_sha256), (predictions, predictions_sha256)):
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f"Exported input checksum mismatch: {path}")
        contents.append([json.loads(line) for line in content.decode().splitlines() if line.strip()])
    instances = {row["instance_id"]: row for row in contents[0]}
    rows = [prediction(row) for row in contents[1]]
    if len(instances) != len(contents[0]) or len({row["instance_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate instance IDs in grading inputs")
    if len({row["model_name_or_path"] for row in rows}) > 1:
        raise ValueError("Grade one model/method per run")
    for row in rows:
        model_dir = row["model_name_or_path"].replace("/", "__")
        if model_dir in (".", "..") or "\\" in model_dir:
            raise ValueError("Unsafe model name in report path")
        public_task(instances[row["instance_id"]])
    return [(instances[row["instance_id"]], row) for row in rows]


def evaluation_command(timeout: int) -> str:
    if timeout <= 0:
        raise ValueError("Use a positive test timeout")
    return f"timeout --signal=TERM --kill-after=10s {timeout}s /bin/bash /eval.sh > /tmp/pca-test-output.txt 2>&1"


def publish_report(spec: "TestSpec", pred: dict, directory: Path, exit_code: int) -> dict:
    from swebench.harness.grading import get_eval_report

    (directory / "status.json").write_text(json.dumps({"state": "finished", "exit_code": exit_code}))
    if exit_code in (124, 137):
        raise TimeoutError("Official tests exceeded the execution limit; no grade published")
    if exit_code in (125, 126, 127):
        raise RuntimeError("Failed to launch the official test script; no grade published")
    report = get_eval_report(spec, pred, str(directory / "test_output.txt"), include_tests_status=True)
    if not report[pred["instance_id"]]["patch_successfully_applied"]:
        raise RuntimeError("Official evaluation did not reach test output; no grade published")
    (directory / "report.json").write_text(json.dumps(report, indent=2))
    return report


def evaluate(instance: dict, pred: dict, root: Path, timeout: int, runtime: RuntimeConfig) -> None:
    from daytona import Daytona, SessionExecuteRequest
    from swebench.harness.run_evaluation import GIT_APPLY_CMDS
    from swebench.harness.test_spec.test_spec import make_test_spec

    directory = root / pred["model_name_or_path"].replace("/", "__") / pred["instance_id"]
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "patch.diff").write_text(pred["model_patch"])
    if not pred["model_patch"].strip():
        (directory / "status.json").write_text(json.dumps({"state": "empty_patch", "tests_executed": False}))
        return
    spec = make_test_spec(instance, namespace="swebench", arch="x86_64")
    (directory / "eval.sh").write_text(spec.eval_script)
    journal = ResourceJournal(directory / "sandbox-cleanup.json")
    api = Daytona()
    try:
        snapshot = create_snapshot(api, runtime, journal, public_task(instance))
        name = f"pca-swe-grading-{uuid4().hex[:16]}"
        journal.record("sandboxes", name, phase="creating", snapshot=snapshot.name)
        sandbox = api.create(
            sandbox_params(snapshot, "grading", runtime, name=name).model_copy(update={"network_block_all": False}),
            timeout=runtime.startup_timeout_seconds,
        )
        journal.record("sandboxes", name, id=sandbox.id, phase="running")
        journal.record("sandboxes", name, lifecycle=sandbox_metadata(api.get(sandbox.id).model_dump(), runtime))
        initial = sandbox.process.exec("git rev-parse HEAD && git status --porcelain", cwd="/testbed", timeout=60)
        (directory / "initial-repository-state.log").write_text(initial.result)

        def command_backend(command: str, *, cwd: str, timeout: int) -> dict:
            response = sandbox.process.exec(command, cwd=cwd, timeout=timeout)
            return {"output": response.result, "returncode": response.exit_code}

        environment = SWEBenchVerifiedEnvironment(command_backend, base_commit=instance["base_commit"])
        environment.prepare(fresh_isolated=True)
        check = sandbox.process.exec(
            "/bin/bash -c "
            + quote(
                'test "$(uname -m)" = x86_64 && command -v timeout && '
                f'test "$(git -C /testbed rev-parse HEAD)" = {quote(instance["base_commit"])} && '
                'test -z "$(git -C /testbed status --porcelain)"'
            ),
            timeout=60,
        )
        if check.exit_code != 0:
            raise RuntimeError(f"Grading sandbox failed base-image validation: {check.result}")
        sandbox.fs.upload_file(pred["model_patch"].encode(), "/tmp/patch.diff")
        applied = False
        patch_log = ""
        for command in GIT_APPLY_CMDS:
            response = sandbox.process.exec(f"{command} /tmp/patch.diff", cwd="/testbed", timeout=60)
            patch_log += f"{command}\n{response.result}\n"
            (directory / "patch-application.log").write_text(patch_log)
            if response.exit_code == 0:
                applied = True
                break
        if not applied:
            (directory / "status.json").write_text(json.dumps({"state": "patch_application_failed"}))
            return
        fixture_environment = prepare_httpbin(sandbox, instance, directory)
        sandbox.fs.upload_file(spec.eval_script.encode(), "/eval.sh")
        sandbox.process.create_session("evaluation")
        command = sandbox.process.execute_session_command(
            "evaluation",
            SessionExecuteRequest(
                command="/bin/bash -c " + quote(fixture_environment + evaluation_command(timeout)), run_async=True
            ),
        )
        deadline = time.monotonic() + timeout + 120
        while (result := sandbox.process.get_session_command("evaluation", command.cmd_id)).exit_code is None:
            if time.monotonic() >= deadline:
                raise TimeoutError("Daytona did not confirm evaluation completion before the deadline")
            time.sleep(3)
        sandbox.fs.download_file("/tmp/pca-test-output.txt", str(directory / "test_output.txt"))
        report = publish_report(spec, pred, directory, result.exit_code)
        typer.echo(f"{pred['instance_id']}: resolved={report[pred['instance_id']]['resolved']}")
    finally:
        delete_resources(api, journal.document(), runtime.cleanup_timeout_seconds)
        (directory / "cleanup-confirmed.json").write_text(json.dumps({"confirmed": True}))


@app.command()
def main(
    dataset_name: Path = typer.Option(..., "--dataset_name", exists=True, dir_okay=False),
    predictions_path: Path = typer.Option(..., "--predictions_path", exists=True, dir_okay=False),
    run_id: str = typer.Option(..., "--run_id"),
    dataset_sha256: str = typer.Option(...),
    predictions_sha256: str = typer.Option(...),
    max_workers: int = typer.Option(1, "--max_workers", min=1, max=6),
    timeout: int = typer.Option(1800, min=1),
    logs: Path = typer.Option(Path("logs/run_evaluation")),
    cpu: int = typer.Option(2, min=1),
    memory: int = typer.Option(8, min=1),
    disk: int = typer.Option(10, min=1),
    dry_run: bool = typer.Option(False, help="Validate inputs and official test specs without calling Daytona."),
) -> None:
    """Grade exported predictions using official tests on Daytona, without local Docker."""
    from swebench.harness.test_spec.test_spec import make_test_spec

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", run_id):
        raise ValueError("Use a fresh, safe run ID")
    verify_harness()
    rows = load_inputs(dataset_name, predictions_path, dataset_sha256, predictions_sha256)
    for instance, _ in rows:
        make_test_spec(instance, namespace="swebench", arch="x86_64")
    if dry_run:
        typer.echo(f"Validated {len(rows)} official test specifications; no Daytona resources created")
        return
    load_dotenv()
    root = logs / run_id
    root.mkdir(parents=True, exist_ok=False)
    (root / "provenance.json").write_text(
        json.dumps(
            {
                "backend": "daytona",
                "harness_revision": HARNESS_REVISION,
                "dataset_sha256": dataset_sha256,
                "predictions_sha256": predictions_sha256,
                "timeout": timeout,
                "max_workers": max_workers,
                "network_block_all": False,
            },
            indent=2,
        )
    )
    runtime = RuntimeConfig(cpu=cpu, memory=memory, disk=disk, ttl_minutes=(timeout + 1800) // 60 + 1)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        list(pool.map(lambda pair: evaluate(*pair, root, timeout, runtime), rows))


if __name__ == "__main__":
    app()
