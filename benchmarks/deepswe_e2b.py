"""Run the pinned DeepSWE cohort with E2B agent, controller, and verifier sandboxes."""

import json
import os
import re
import secrets
import shlex
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

import typer
import yaml
from dotenv import load_dotenv

from benchmarks.deepswe_common import COHORT, UPSTREAM, Task, load_cohort, method_config, record_run, save_json, sha256
from benchmarks.utils.metrics import episode_metrics
from pocketcontextagent.agents import get_agent
from pocketcontextagent.comparisons.registry import METHODS
from pocketcontextagent.environments.swebench_verified import SWEBenchVerifiedEnvironment
from pocketcontextagent.models import get_model

app = typer.Typer(add_completion=False, no_args_is_help=True)
MAX_SANDBOX_SECONDS = 3_600
CONTROLLERS_PER_METHOD = 2
WORKERS_PER_CONTROLLER = 6
_template_locks: dict[str, threading.Lock] = {}
_template_locks_guard = threading.Lock()
INFRASTRUCTURE_ERRORS = {
    "ConnectError",
    "ConnectionError",
    "ImportError",
    "ModuleNotFoundError",
    "ReadError",
    "SandboxNotFoundException",
    "SandboxTimeoutException",
    "TimeoutError",
    "TimeoutException",
}


def controller_assignments(tasks: list[Task]) -> dict[str, list[list[Task]]]:
    if len(tasks) != 50 or len({task.task_id for task in tasks}) != 50:
        raise ValueError("The E2B run requires the pinned 50 unique task IDs")
    return {method: [tasks[:25], tasks[25:]] for method in METHODS}


def bounded_command(command: str, timeout: int) -> str:
    if timeout <= 0:
        raise ValueError("Command timeout must be positive")
    return f"timeout --signal=TERM --kill-after=5s {timeout}s {command}"


def _start_heartbeat(sandbox_id: str) -> tuple[threading.Event, threading.Thread]:
    from e2b import Sandbox

    stop = threading.Event()

    def refresh() -> None:
        while not stop.wait(600):
            Sandbox.set_timeout(sandbox_id, MAX_SANDBOX_SECONDS)

    thread = threading.Thread(target=refresh, daemon=True)
    thread.start()
    return stop, thread


def run_instance(
    instance: dict,
    output: Path,
    config: dict,
    env: SWEBenchVerifiedEnvironment,
    manifest: dict,
    runtime_metadata: dict | None = None,
) -> dict:
    iid = instance["query_id"]
    trajectory = output / "trajectories" / iid
    output.mkdir(parents=True, exist_ok=True)
    (output / "run.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    fingerprint = sha256(json.dumps(manifest, sort_keys=True).encode())
    agent = get_agent(
        get_model(config=config["model"]),
        env,
        config["agent"] | {"output_path": trajectory / "trajectory.json", "context_db_path": trajectory / "context.sqlite"},
    )
    started = time.monotonic()
    ended = False
    try:
        agent.run(instance["query"])
        ended = True
    finally:
        elapsed = time.monotonic() - started
        info = agent.save(trajectory / "trajectory.json")["info"]
        error = type(sys.exc_info()[1]).__name__ if not ended and sys.exc_info()[1] is not None else None
        patch = env.capture_patch()
        messages = (
            agent.messages
            + [call["response"] for call in agent.compactions]
            + [call["response"] for call in getattr(agent, "memory_queries", [])]
        )
        metrics = episode_metrics(messages, model_stats=info["model_stats"], latency_seconds=elapsed)
        if error and episode_metrics(messages)["model_calls"] < info["model_stats"]["api_calls"]:
            info["model_stats"]["cost_available"] = False
            metrics["cost_usd"] = None
        if error:
            agent.save(trajectory / "trajectory.json", {"info": {"execution_error": error, "model_stats": info["model_stats"]}})
        result = {
            "query_id": iid,
            "instance_id": iid,
            "status": "completed" if info["exit_status"] == "Submitted" else info["exit_status"] or error or "interrupted",
            "execution_error": error,
            "agent_exit_status": info["exit_status"],
            "agent_submission": info["submission"],
            "model_name_or_path": agent.model.config.model_name,
            "model_patch": patch,
            "patch_capture_status": "captured",
            "result": [{"type": "output_text", "output": patch}],
            "tool_call_counts": metrics["tool_call_counts"],
            "model_stats": info["model_stats"],
            "metrics": metrics,
            "metadata": {
                "model": agent.model.config.model_name,
                "method": manifest["method"],
                "fingerprint": fingerprint,
                **(runtime_metadata or {}),
            },
        }
        save_json(output / f"{iid}.json", result)
        with tarfile.open(output / "trajectories.tar.gz.tmp", "w:gz") as archive:
            archive.add(output / "trajectories", arcname="trajectories")
        (output / "trajectories.tar.gz.tmp").replace(output / "trajectories.tar.gz")
    return result


def configure_key(key_file: Path | None) -> None:
    if key_file is not None:
        match = re.search(r"^E2B_API_KEY\s*=\s*(\S+)\s*$", key_file.read_text(), re.M)
        if match is None:
            raise ValueError("Key file has no E2B_API_KEY assignment")
        os.environ["E2B_API_KEY"] = match[1].strip("\"'")
    if not os.getenv("E2B_API_KEY"):
        raise ValueError("Set E2B_API_KEY or provide --e2b-key-file")


def _lock_for(alias: str) -> threading.Lock:
    with _template_locks_guard:
        return _template_locks.setdefault(alias, threading.Lock())


def _build_template(alias: str, builder, output: Path, metadata: dict) -> None:
    from e2b import Template

    intent = output / "templates" / f"{alias}.json"
    if intent.exists() and any(json.loads(intent.read_text()).get(key) != value for key, value in metadata.items()):
        raise ValueError(f"E2B template intent differs from this run: {alias}")
    if not intent.exists():
        save_json(intent, metadata)
    with _lock_for(alias):
        if not Template.alias_exists(alias):
            info = Template.build(builder, alias=alias, cpu_count=2, memory_mb=8192, min_free_disk_mb=2000)
            save_json(intent, metadata | {"template_id": info.template_id, "build_id": info.build_id})


def agent_template(task: Task, output: Path, revision: str) -> str:
    from e2b import Template

    alias = "pca-deepswe-agent-" + sha256(f"{revision}:{task.image}".encode())[:20]
    _build_template(alias, Template().from_image(task.image), output, {"alias": alias, "image": task.image})
    return alias


def grader_template(task: Task, output: Path, revision: str) -> str:
    dockerfile = (task.tests_dir / "Dockerfile").read_text().splitlines()
    base_lines = [line.split(maxsplit=1)[1].strip() for line in dockerfile if line.upper().startswith("FROM ")]
    if base_lines != [task.image]:
        raise ValueError(f"DeepSWE verifier Dockerfile does not use the pinned task image: {task.task_id}")
    prerequisite_commands = grader_prerequisite_commands(task)
    if prerequisite_commands:
        metadata = {
            "alias": "pca-deepswe-grader-" + sha256(
                f"{revision}:{task.image}:{prerequisite_commands}".encode()
            )[:20],
            "image": task.image,
            "prerequisite_commands": prerequisite_commands,
        }
        alias = metadata["alias"]
        _build_template(
            alias,
            _template_with_commands(task.image, prerequisite_commands),
            output,
            metadata,
        )
        return alias
    return agent_template(task, output, revision)


def grader_prerequisite_commands(task: Task) -> list[str]:
    environment_dockerfile = task.tests_dir.parent / "environment" / "Dockerfile"
    return [
        line.split(maxsplit=1)[1].strip()
        for line in environment_dockerfile.read_text().splitlines()
        if line.upper().startswith("RUN ") and "go-ctrf-json-reporter" in line
    ]


def _template_with_commands(image: str, commands: list[str]):
    from e2b import Template

    builder = Template().from_image(image)
    for command in commands:
        builder = builder.run_cmd(command)
    return builder


def _kill_sandbox(sandbox_id: str, intent: Path, receipt: Path) -> None:
    from e2b import Sandbox

    save_json(intent, {"sandbox_id": sandbox_id})
    if not Sandbox.kill(sandbox_id):
        raise RuntimeError(f"E2B did not confirm sandbox deletion: {sandbox_id}")
    save_json(receipt, {"sandbox_id": sandbox_id, "deleted": True})


def run_task(task: Task, output: Path, config: dict, revision: str, method: str) -> Path:
    from e2b import Sandbox
    from e2b.sandbox.commands.command_handle import CommandExitException

    directory = output / "tasks" / task.task_id
    inference = directory / "inference"
    result = inference / f"{task.task_id}.json"
    intent = directory / "inference-intent.json"
    retry_of = None
    if result.exists():
        saved = json.loads(result.read_text())
        if saved.get("query_id") != task.task_id or saved.get("metadata", {}).get("method") != method:
            raise ValueError(f"Saved inference result does not match task and method: {result}")
        if not saved.get("execution_error") or saved["execution_error"] not in INFRASTRUCTURE_ERRORS:
            return result
    if intent.exists():
        previous = json.loads(intent.read_text())
        if previous.get("status") != "infrastructure_failed":
            raise RuntimeError(f"Started E2B inference needs reconciliation before retry: {intent}")
        retry_of = previous["attempt_id"]
        attempts = directory / "attempts"
        attempts.mkdir(parents=True, exist_ok=True)
        previous_dir = attempts / f"attempt-{len(list(attempts.iterdir())) + 1:03d}"
        previous_dir.mkdir()
        shutil.copy2(intent, previous_dir / intent.name)
        if inference.exists():
            shutil.copytree(inference, previous_dir / "inference", dirs_exist_ok=True)
        failure_path = directory / "infrastructure-error.json"
        if failure_path.exists():
            shutil.copy2(failure_path, previous_dir / failure_path.name)
    alias = agent_template(task, output, revision)
    attempt_id = uuid4().hex
    save_json(
        intent,
        {
            "task_id": task.task_id,
            "method": method,
            "template": alias,
            "attempt_id": attempt_id,
            "retry_of": retry_of,
        },
    )
    sandbox = Sandbox.create(
        alias,
        timeout=MAX_SANDBOX_SECONDS,
        secure=True,
        allow_internet_access=False,
        metadata={"project": "pocket-context", "benchmark": "deepswe", "method": method, "role": "agent"},
        idempotency_key=attempt_id,
    )
    save_json(intent, json.loads(intent.read_text()) | {"sandbox_id": sandbox.sandbox_id})
    stop_heartbeat, heartbeat = _start_heartbeat(sandbox.sandbox_id)
    try:

        def command_backend(command: str, *, cwd: str, timeout: int) -> dict:
            log_path = f"/tmp/pca-command-{uuid4().hex}.log"
            wrapped = (
                f"{bounded_command(command, timeout)} > {log_path} 2>&1; status=$?; "
                f"tail -c 8192 {log_path}; rm -f {log_path}; exit $status"
            )
            try:
                response = sandbox.commands.run(
                    wrapped, cwd=cwd, user="root", timeout=timeout + 15, request_timeout=timeout + 15
                )
            except CommandExitException as error:
                response = error
            return {"output": response.stdout, "returncode": response.exit_code}

        environment = SWEBenchVerifiedEnvironment(command_backend, **config["environment"], base_commit=task.base_commit)
        environment.prepare(fresh_isolated=True)
        if environment.capture_patch().strip() or environment._run("test ! -e /tests/grader.py")["returncode"]:
            raise RuntimeError("DeepSWE E2B agent image is not clean or contains the hidden verifier")
        run_instance(
            {"query_id": task.task_id, "query": task.instruction},
            inference,
            config,
            environment,
            {"method": method, "config": config, "benchmark": "deepswe", "upstream_revision": revision},
            {"sandbox_id": sandbox.sandbox_id, "image": task.image, "execution_backend": "e2b"},
        )
    finally:
        stop_heartbeat.set()
        heartbeat.join(timeout=5)
        if result.exists():
            _kill_sandbox(
                sandbox.sandbox_id,
                directory / "inference-cleanup-intent.json",
                directory / "inference-cleanup.json",
            )
    return result


def grade_task(task: Task, output: Path, result_path: Path, revision: str, method: str) -> dict:
    from e2b import Sandbox

    result = json.loads(result_path.read_text())
    if result["query_id"] != task.task_id:
        raise ValueError("Inference result belongs to another DeepSWE task")
    patch = result["model_patch"]
    patch_hash = sha256(patch.encode())
    directory = output / "tasks" / task.task_id
    report_path = directory / "grade.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if report["patch_sha256"] != patch_hash or report["upstream_revision"] != revision:
            raise ValueError("Saved grade does not match the saved patch or verifier revision")
        return report
    alias = grader_template(task, output, revision)
    attempt = directory / "grading-attempts" / uuid4().hex
    attempt.mkdir(parents=True)
    intent = attempt / "intent.json"
    save_json(intent, {"task_id": task.task_id, "method": method, "template": alias, "patch_sha256": patch_hash})
    sandbox = Sandbox.create(
        alias,
        timeout=MAX_SANDBOX_SECONDS,
        secure=True,
        allow_internet_access=False,
        metadata={"project": "pocket-context", "benchmark": "deepswe", "method": method, "role": "grader"},
        idempotency_key=attempt.name,
    )
    save_json(intent, json.loads(intent.read_text()) | {"sandbox_id": sandbox.sandbox_id})
    stop_heartbeat, heartbeat = _start_heartbeat(sandbox.sandbox_id)
    try:
        sandbox.commands.run("mkdir -p /tests /logs/artifacts /logs/verifier", user="root", timeout=30)
        for name in ("test.sh", "test.patch", "grader.py", "config.json"):
            sandbox.files.write(f"/tests/{name}", (task.tests_dir / name).read_bytes(), user="root")
        setup = sandbox.commands.run("chmod +x /tests/test.sh", user="root", timeout=30)
        if setup.exit_code:
            raise RuntimeError(f"Verifier setup failed: {setup.stdout}")
        sandbox.files.write("/logs/artifacts/model.patch", patch, user="root")
        response = sandbox.commands.run(
            "/bin/bash /tests/test.sh > /logs/verifier/test-stdout.txt 2>&1",
            cwd="/app",
            user="root",
            timeout=task.verifier_timeout + 60,
        )
        save_json(attempt / "execution.json", {"exit_code": response.exit_code, "output": response.stdout})
        remote = "/workspace/export/deepswe-verifier.tar.gz"
        archive_response = sandbox.commands.run(
            "mkdir -p /workspace/export && tar -czf /workspace/export/deepswe-verifier.tar.gz "
            "-C /logs verifier && sha256sum /workspace/export/deepswe-verifier.tar.gz && "
            "stat -c %s /workspace/export/deepswe-verifier.tar.gz",
            user="root",
            timeout=120,
            request_timeout=130,
        )
        if archive_response.exit_code:
            raise RuntimeError(f"Verifier artifact archival failed: {archive_response.stdout}")
        fields = archive_response.stdout.split()
        archive_sha256 = _verify_remote_sha256(archive_response.stdout, fields[0])
        archive_path = attempt / "verifier-artifacts.tar.gz"
        _download_controller_file(sandbox, remote, archive_path, archive_sha256, int(fields[-1]))
        with tarfile.open(archive_path, "r:gz") as archive:
            archive.extractall(attempt / "retrieved", filter="data")
        verifier_dir = attempt / "retrieved" / "verifier"
        reward = json.loads((verifier_dir / "reward.json").read_text())
        shutil.copy2(verifier_dir / "reward.json", attempt / "reward.json")
        shutil.copy2(verifier_dir / "test-stdout.txt", attempt / "test-stdout.txt")
        ctrf = verifier_dir / "ctrf.json"
        if ctrf.exists():
            shutil.copy2(ctrf, attempt / "ctrf.json")
        if reward.get("reward") not in (0, 1) or reward.get("apply_failed") and reward["reward"] != 0:
            raise RuntimeError(f"DeepSWE verifier produced an invalid score: {task.task_id}")
        report = {
            "task_id": task.task_id,
            "method": method,
            "upstream_revision": revision,
            "patch_sha256": patch_hash,
            "reward": reward,
            "verifier_exit_code": response.exit_code,
            "attempt": str(attempt.relative_to(output)),
            "verifier_artifacts_sha256": archive_sha256,
        }
        save_json(report_path, report)
        _kill_sandbox(sandbox.sandbox_id, attempt / "cleanup-intent.json", attempt / "cleanup.json")
        return report
    except Exception as error:
        save_json(attempt / "infrastructure-error.json", {"error": type(error).__name__, "message": str(error)})
        raise
    finally:
        stop_heartbeat.set()
        heartbeat.join(timeout=5)


def _task_archive(task_dir: Path) -> bytes:
    archive = task_dir / "task-artifacts.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for path in sorted(task_dir.iterdir()):
            if path != archive:
                bundle.add(path, arcname=path.name)
    return archive.read_bytes()


def _verify_remote_sha256(output: str, expected: str) -> str:
    fields = output.split()
    if not fields or fields[0] != expected:
        raise ValueError("E2B controller artifact checksum mismatch")
    return fields[0]


def _download_controller_file(sandbox, remote: str, local: Path, expected_sha256: str, expected_bytes: int) -> None:
    token = secrets.token_urlsafe(32)
    script_path = f"/workspace/export/serve-{token}.py"
    script = (
        "from http.server import BaseHTTPRequestHandler, HTTPServer\n"
        "import hmac, os\n"
        f"TOKEN = {token!r}\n"
        f"FILE = {remote!r}\n"
        "class Handler(BaseHTTPRequestHandler):\n"
        " def do_GET(self):\n"
        "  if not hmac.compare_digest(self.path, '/' + TOKEN): self.send_error(404); return\n"
        "  self.send_response(200); self.send_header('Content-Length', str(os.path.getsize(FILE))); self.end_headers()\n"
        "  with open(FILE, 'rb') as stream:\n"
        "   while chunk := stream.read(65536): self.wfile.write(chunk)\n"
        " def log_message(self, *args): pass\n"
        "HTTPServer(('0.0.0.0', 8000), Handler).serve_forever()\n"
    )
    sandbox.files.write(script_path, script, user="root")
    start = sandbox.commands.run(
        f"nohup python {shlex.quote(script_path)} >/workspace/export/http-export.log 2>&1 < /dev/null & echo $!",
        user="root",
        timeout=20,
        request_timeout=25,
    )
    if start.exit_code:
        raise RuntimeError("Could not start the E2B artifact export server")
    pid = start.stdout.strip().splitlines()[-1]
    temporary = local.with_suffix(local.suffix + ".tmp")
    try:
        url = f"https://{sandbox.get_host(8000)}/{token}"
        with urllib.request.urlopen(url, timeout=600) as response, temporary.open("wb") as destination:
            shutil.copyfileobj(response, destination)
        data = temporary.read_bytes()
        if len(data) != expected_bytes or sha256(data) != expected_sha256:
            raise ValueError("Downloaded E2B controller artifact failed its checksum or size verification")
        temporary.replace(local)
    finally:
        sandbox.commands.run(
            f"kill {shlex.quote(pid)} 2>/dev/null || true; rm -f {shlex.quote(script_path)}",
            user="root",
            timeout=20,
        )


def _set_controller_status(sandbox, status: dict, lock: threading.Lock) -> None:
    from e2b import Sandbox

    with lock:
        content = json.dumps(status, indent=2, ensure_ascii=False) + "\n"
        sandbox.files.write("/workspace/controller-status.json", content, user="root")
        Sandbox.set_timeout(sandbox.sandbox_id, MAX_SANDBOX_SECONDS)


def _build_controller_package(output: Path) -> tuple[str, Path]:
    from e2b import Template

    runtime = output / "controller-runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--no-isolation", "--outdir", str(runtime), "."],
        check=True,
        capture_output=True,
        text=True,
    )
    wheel = next(runtime.glob("*.whl"))
    wheel_hash = sha256(wheel.read_bytes())
    alias = "pca-deepswe-controller-" + wheel_hash[:20]
    intent = output / "templates" / f"controller-{wheel_hash[:20]}.json"
    expected = {"alias": alias, "wheel_sha256": wheel_hash, "base_image": "python:3.11-slim"}
    if intent.exists() and any(json.loads(intent.read_text()).get(key) != value for key, value in expected.items()):
        raise ValueError("E2B controller image intent differs from this package")
    if not Template.alias_exists(alias):
        info = Template.build(
            Template().from_image("python:3.11-slim"),
            alias=alias,
            cpu_count=4,
            memory_mb=8192,
            min_free_disk_mb=2000,
        )
        expected |= {"template_id": info.template_id, "build_id": info.build_id}
    save_json(intent, expected)
    return alias, wheel


def _task_payload(task: Task) -> dict:
    return {
        "task_id": task.task_id,
        "instruction": task.instruction,
        "image": task.image,
        "base_commit": task.base_commit,
        "tests_dir": f"/workspace/payload/tests/{task.task_id}",
        "verifier_timeout": task.verifier_timeout,
    }


def _start_controller(
    method: str,
    number: int,
    tasks: list[Task],
    output: Path,
    alias: str,
    wheel: Path,
    config: dict,
    revision: str,
    cohort: dict,
    reuse_sources: dict[str, Path],
):
    from e2b import Sandbox

    intent = output / "controllers" / f"{method}-{number}.json"
    if intent.exists():
        previous = json.loads(intent.read_text())
        expected_tasks = [task.task_id for task in tasks]
        if previous.get("method") != method or previous.get("controller") != number or previous.get("task_ids") != expected_tasks:
            raise ValueError(f"Saved controller intent differs from this assignment: {intent}")
        if previous.get("status") not in {"created", "running", "archived"} or not previous.get("sandbox_id"):
            raise RuntimeError(f"Ambiguous controller launch requires reconciliation: {intent}")
        sandbox = Sandbox.connect(previous["sandbox_id"])
        if previous["status"] in {"running", "archived"}:
            return sandbox
    else:
        attempt_id = uuid4().hex
        save_json(
            intent,
            {
                "method": method,
                "controller": number,
                "task_ids": [task.task_id for task in tasks],
                "worker_cap": WORKERS_PER_CONTROLLER,
                "attempt_id": attempt_id,
                "status": "creating",
            },
        )
        sandbox = Sandbox.create(
            alias,
            timeout=MAX_SANDBOX_SECONDS,
            secure=True,
            allow_internet_access=True,
            envs={"E2B_API_KEY": os.environ["E2B_API_KEY"], "DEEPSEEK_API_KEY": os.environ["DEEPSEEK_API_KEY"]},
            metadata={
                "project": "pocket-context",
                "benchmark": "deepswe",
                "method": method,
                "role": "controller",
                "controller": str(number),
            },
            idempotency_key=attempt_id,
        )
        save_json(intent, json.loads(intent.read_text()) | {"sandbox_id": sandbox.sandbox_id, "status": "created"})
    sandbox.commands.run("mkdir -p /workspace/payload/tests /workspace/run /workspace/artifacts", user="root", timeout=30)
    remote_wheel = f"/workspace/{wheel.name}"
    sandbox.files.write(remote_wheel, wheel.read_bytes(), user="root")
    install = sandbox.commands.run(
        f"python -m pip install --force-reinstall --no-cache-dir '{remote_wheel}[e2b]' && "
        "python -c 'import e2b, benchmarks.deepswe_e2b'",
        user="root",
        timeout=1200,
    )
    if install.exit_code:
        raise RuntimeError(f"E2B controller package install failed: {install.stdout}")
    payload = {
        "method": method,
        "controller": number,
        "sandbox_id": sandbox.sandbox_id,
        "task_ids": [task.task_id for task in tasks],
        "tasks": [_task_payload(task) for task in tasks],
        "reused_task_ids": sorted(set(reuse_sources) & {task.task_id for task in tasks}),
        "config": config,
        "upstream_revision": revision,
        "cohort": cohort,
        "output": f"/workspace/run/{method}/controller-{number}",
    }
    save_json(output / "controller-payloads" / f"{method}-{number}.json", payload)
    tar_path = output / "controller-payloads" / f"{method}-{number}.tar.gz"
    with tarfile.open(tar_path, "w:gz") as archive:
        archive.add(output / "controller-payloads" / f"{method}-{number}.json", arcname="controller.json")
        for task in tasks:
            archive.add(task.tests_dir, arcname=f"tests/{task.task_id}")
            if task.task_id in reuse_sources:
                source = reuse_sources[task.task_id]
                archive.add(source / "inference", arcname=f"reuse/{task.task_id}/inference")
                for name in ("inference-intent.json", "inference-cleanup-intent.json", "inference-cleanup.json"):
                    if (source / name).exists():
                        archive.add(source / name, arcname=f"reuse/{task.task_id}/{name}")
    sandbox.files.write("/workspace/payload.tar.gz", tar_path.read_bytes(), user="root")
    extract = sandbox.commands.run("tar -xzf /workspace/payload.tar.gz -C /workspace/payload", user="root", timeout=120)
    if extract.exit_code:
        raise RuntimeError(f"Could not unpack E2B controller payload: {extract.stdout}")
    response = sandbox.commands.run(
        "nohup python -m benchmarks.deepswe_e2b controller-run /workspace/payload/controller.json "
        "> /workspace/controller.log 2>&1 < /dev/null & echo $!",
        envs={"E2B_API_KEY": os.environ["E2B_API_KEY"], "DEEPSEEK_API_KEY": os.environ["DEEPSEEK_API_KEY"]},
        user="root",
        timeout=30,
    )
    if response.exit_code:
        raise RuntimeError(f"Could not start E2B controller: {response.stdout}")
    save_json(intent, json.loads(intent.read_text()) | {"status": "running", "worker_cap": WORKERS_PER_CONTROLLER})
    return sandbox


def _execute_group(
    method: str,
    number: int,
    tasks: list[Task],
    sandbox,
    status: dict,
    config: dict,
    revision: str,
    output: Path,
) -> list[dict]:
    status_lock = threading.Lock()
    sandbox_lock = threading.Lock()

    def record() -> None:
        with sandbox_lock:
            _set_controller_status(sandbox, status, status_lock)

    def one(task: Task) -> dict:
        with status_lock:
            status["active_workers"] += 1
            status["started"] = status.get("started", []) + [task.task_id]
        record()
        try:
            result_path = run_task(task, output, config, revision, method)
            inference = json.loads(result_path.read_text())
            if inference.get("execution_error") in INFRASTRUCTURE_ERRORS:
                raise RuntimeError(f"Verified E2B inference infrastructure failure: {inference['execution_error']}")
            report = grade_task(task, output, result_path, revision, method)
            artifact = _task_archive(output / "tasks" / task.task_id)
            remote = f"/workspace/artifacts/{task.task_id}.tar.gz"
            with sandbox_lock:
                sandbox.files.write(remote, artifact, user="root")
                response = sandbox.commands.run(
                    f"sha256sum {remote}", user="root", timeout=30, request_timeout=40
                )
            if response.exit_code:
                raise RuntimeError(f"E2B controller artifact checksum failed: {task.task_id}")
            _verify_remote_sha256(response.stdout, sha256(artifact))
            receipt = {
                "task_id": task.task_id,
                "remote_path": remote,
                "sha256": sha256(artifact),
                "bytes": len(artifact),
                "verified_remote_sha256": True,
            }
            save_json(output / "controllers" / f"{method}-{number}" / f"{task.task_id}-receipt.json", receipt)
            with status_lock:
                status["completed"].append(task.task_id)
                status["archive_receipts"].append(receipt)
            return {
                "task_id": task.task_id,
                "method": method,
                "reward": report["reward"]["reward"],
                "infrastructure_error": None,
            }
        except Exception as error:
            failure = {"task_id": task.task_id, "error": type(error).__name__, "message": str(error)}
            intent = output / "tasks" / task.task_id / "inference-intent.json"
            if intent.exists():
                saved_intent = json.loads(intent.read_text())
                saved_intent["status"] = "infrastructure_failed"
                saved_intent["failure"] = failure
                save_json(intent, saved_intent)
            with status_lock:
                status["failed"].append(failure)
            save_json(output / "controller-errors" / f"{method}-{number}" / f"{task.task_id}.json", failure)
            return {"task_id": task.task_id, "method": method, "reward": None, "infrastructure_error": failure}
        finally:
            with status_lock:
                status["active_workers"] -= 1
            record()

    with ThreadPoolExecutor(max_workers=WORKERS_PER_CONTROLLER) as executor:
        return list(executor.map(one, tasks))


def controller_run(payload_path: Path) -> None:
    from e2b import Sandbox

    payload = json.loads(payload_path.read_text())
    method = payload["method"]
    tasks = [Task(**(task | {"tests_dir": Path(task["tests_dir"])})) for task in payload["tasks"]]
    output = Path(payload["output"])
    output.mkdir(parents=True, exist_ok=True)
    cohort = payload["cohort"]
    config = payload["config"]
    revision = payload["upstream_revision"]
    for task_id in payload.get("reused_task_ids", []):
        source = payload_path.parent / "reuse" / task_id
        destination = output / "tasks" / task_id
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source / "inference", destination / "inference", dirs_exist_ok=True)
        for name in ("inference-intent.json", "inference-cleanup-intent.json", "inference-cleanup.json"):
            if (source / name).exists():
                shutil.copy2(source / name, destination / name)
    record_run(output, cohort, config, method=method, smoke=False, task_ids=[task.task_id for task in tasks])
    controller = Sandbox.connect(payload["sandbox_id"])
    status = {
        "method": method,
        "controller": payload["controller"],
        "sandbox_id": payload["sandbox_id"],
        "task_ids": payload["task_ids"],
        "worker_cap": WORKERS_PER_CONTROLLER,
        "active_workers": 0,
        "completed": [],
        "failed": [],
        "archive_receipts": [],
    }
    _set_controller_status(controller, status, threading.Lock())
    stop_heartbeat, heartbeat = _start_heartbeat(controller.sandbox_id)
    try:
        results = _execute_group(
            method,
            payload["controller"],
            tasks,
            controller,
            status,
            config,
            revision,
            output,
        )
    finally:
        stop_heartbeat.set()
        heartbeat.join(timeout=5)
    summary = {
        "method": method,
        "controller": payload["controller"],
        "attempted": len(results),
        "graded": sum(result["reward"] is not None for result in results),
        "resolved": sum(result["reward"] == 1 for result in results),
        "infrastructure_failed": sum(result["infrastructure_error"] is not None for result in results),
        "results": results,
    }
    save_json(output / "controller-summary.json", summary)
    status["state"] = "complete"
    status["summary"] = summary
    _set_controller_status(controller, status, threading.Lock())


def controller_retry(payload_path: Path) -> None:
    from e2b import Sandbox

    payload = json.loads(payload_path.read_text())
    tasks = [Task(**(task | {"tests_dir": Path(task["tests_dir"])})) for task in payload["tasks"]]
    output = Path(payload["output"])
    status = {
        "method": payload["method"],
        "controller": payload["controller"],
        "sandbox_id": payload["sandbox_id"],
        "task_ids": payload["task_ids"],
        "worker_cap": WORKERS_PER_CONTROLLER,
        "active_workers": 0,
        "completed": [],
        "failed": [],
        "archive_receipts": [],
    }
    controller = Sandbox.connect(payload["sandbox_id"])
    _set_controller_status(controller, status, threading.Lock())
    stop_heartbeat, heartbeat = _start_heartbeat(controller.sandbox_id)
    try:
        results = _execute_group(
            payload["method"],
            payload["controller"],
            tasks,
            controller,
            status,
            payload["config"],
            payload["upstream_revision"],
            output,
        )
    finally:
        stop_heartbeat.set()
        heartbeat.join(timeout=5)
    summary = {
        "method": payload["method"],
        "controller": payload["controller"],
        "attempted": len(results),
        "graded": sum(result["reward"] is not None for result in results),
        "resolved": sum(result["reward"] == 1 for result in results),
        "infrastructure_failed": sum(result["infrastructure_error"] is not None for result in results),
        "results": results,
    }
    save_json(output / f"controller-retry-{payload['retry_id']}-summary.json", summary)
    status["state"] = "retry_complete"
    status["summary"] = summary
    _set_controller_status(controller, status, threading.Lock())


def _poll_controller(sandbox, intent: Path, output: Path) -> dict:
    controller_id = json.loads(intent.read_text())["sandbox_id"]
    try:
        status = json.loads(sandbox.files.read("/workspace/controller-status.json", user="root"))
    except Exception:
        if not sandbox.is_running():
            try:
                detail = sandbox.files.read("/workspace/controller.log", user="root")
            except Exception as error:
                detail = str(error)
            return {"sandbox_id": controller_id, "state": "controller_exited", "log": detail}
        return {"sandbox_id": controller_id, "state": "starting"}
    save_json(output / "controllers" / (intent.stem + "-status.json"), status)
    if status.get("state") == "complete" and not (output / "controllers" / (intent.stem + "-bundle.json")).exists():
        response = sandbox.commands.run(
            "mkdir -p /workspace/export && tar -czf /workspace/export/controller-bundle.tar.gz "
            "-C /workspace/run . -C /workspace controller.log controller-status.json payload/controller.json && "
            "sha256sum /workspace/export/controller-bundle.tar.gz && "
            "stat -c %s /workspace/export/controller-bundle.tar.gz",
            user="root",
            timeout=300,
        )
        if response.exit_code:
            raise RuntimeError(f"Controller artifact archive failed: {response.stdout}")
        fields = response.stdout.split()
        expected_hash = fields[0]
        expected_size = int(fields[-1])
        archive_path = output / "controllers" / (intent.stem + "-bundle.tar.gz")
        _download_controller_file(
            sandbox,
            "/workspace/export/controller-bundle.tar.gz",
            archive_path,
            expected_hash,
            expected_size,
        )
        receipt = {"sandbox_id": controller_id, "sha256": expected_hash, "bytes": expected_size, "verified": True}
        save_json(output / "controllers" / (intent.stem + "-bundle.json"), receipt)
        with tarfile.open(archive_path, "r:gz") as archive:
            archive.extractall(output, filter="data")
        assignment = json.loads(intent.read_text())
        if not (
            output / assignment["method"] / f"controller-{assignment['controller']}" / "controller-summary.json"
        ).exists():
            raise ValueError(f"Controller bundle missed its completion summary: {intent.stem}")
        save_json(intent, json.loads(intent.read_text()) | {"status": "archived", "bundle": receipt})
    return status


def execute_four_methods(
    output: Path,
    upstream: Path,
    cohort_path: Path,
    key_file: Path | None,
) -> None:
    load_dotenv(Path(".env"))
    configure_key(key_file)
    if not os.getenv("DEEPSEEK_API_KEY"):
        raise ValueError("DEEPSEEK_API_KEY is required for the configured DeepSWE model")
    if output.exists() and (output / "run-manifest.json").exists():
        existing_status = json.loads((output / "run-status.json").read_text()) if (output / "run-status.json").exists() else {}
        if existing_status.get("status") != "creating_controllers":
            raise ValueError("Run output already has active or completed work; reconcile before resuming")
    cohort, tasks = load_cohort(upstream, cohort_path)
    configs = {method: method_config(method) for method in METHODS}
    assignments = controller_assignments(tasks)
    controller_template, controller_wheel = _build_controller_package(output)
    image_tasks = {task.image: task for task in tasks}
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda task: agent_template(task, output, cohort["upstream_revision"]), image_tasks.values()))
    reuse_sources: dict[str, dict[str, Path]] = {}
    previous = Path("runs/deepswe-v1.1-react-e2b-full-e2e-20261101")
    previous_result = previous / "tasks/adaptix-name-mapping-aliases/inference/adaptix-name-mapping-aliases.json"
    if previous_result.is_file():
        previous_manifest = json.loads((previous / "run-manifest.json").read_text())
        config_hash = sha256(json.dumps(configs["react"], sort_keys=True).encode())
        saved = json.loads(previous_result.read_text())
        if (
            previous_manifest.get("cohort_sha256") == cohort["task_ids_sha256"]
            and previous_manifest.get("upstream_revision") == cohort["upstream_revision"]
            and previous_manifest.get("method") == "react"
            and previous_manifest.get("context_token_limit") == 128000
            and previous_manifest.get("config_sha256") == config_hash
            and saved.get("query_id") == "adaptix-name-mapping-aliases"
            and saved.get("execution_error") is None
        ):
            reuse_sources["react"] = {"adaptix-name-mapping-aliases": previous_result.parent.parent}
            run_reused = {
                "method": "react",
                "task_id": "adaptix-name-mapping-aliases",
                "source_run": str(previous),
                "patch_sha256": sha256(saved["model_patch"].encode()),
            }
        else:
            run_reused = None
    else:
        run_reused = None
    output.mkdir(parents=True, exist_ok=True)
    run = {
        "benchmark": "DeepSWE v1.1",
        "upstream_revision": cohort["upstream_revision"],
        "cohort_sha256": cohort["task_ids_sha256"],
        "task_ids": [task.task_id for task in tasks],
        "methods": list(METHODS),
        "context_token_limit": 128000,
        "controllers_per_method": CONTROLLERS_PER_METHOD,
        "workers_per_controller": WORKERS_PER_CONTROLLER,
        "maximum_concurrent_task_sandboxes": len(METHODS) * CONTROLLERS_PER_METHOD * WORKERS_PER_CONTROLLER,
        "execution_backend": "e2b",
        "grader": "pinned upstream DeepSWE v1.1 tests/test.sh and tests/grader.py",
        "storage": "E2B controller sandbox filesystems; verified local artifact copies",
        "controller_execution": "two remote E2B controller sandboxes per method; six remote worker threads per controller",
        "model_configs_sha256": {
            method: sha256(json.dumps(config, sort_keys=True).encode()) for method, config in configs.items()
        },
        "reused_inference": [run_reused] if run_reused else [],
    }
    manifest_path = output / "run-manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != run:
            raise ValueError("Saved run manifest differs from this frozen experiment")
    else:
        save_json(manifest_path, run)
    status_path = output / "run-status.json"
    if status_path.exists():
        if json.loads(status_path.read_text()).get("status") != "creating_controllers":
            raise ValueError("Saved run status is not resumable controller provisioning")
    else:
        save_json(status_path, {"run": run, "status": "creating_controllers", "methods": {}})

    controllers = {}
    for method in METHODS:
        method_output = output / method
        record_run(
            method_output,
            cohort,
            configs[method],
            method=method,
            smoke=False,
            task_ids=[task.task_id for task in tasks],
            backend="e2b",
        )
        save_json(
            method_output / "controller-assignment.json",
            {
                "method": method,
                "controllers": [
                    {"controller": index + 1, "task_ids": [task.task_id for task in tasks[index * 25 : (index + 1) * 25]]}
                    for index in range(CONTROLLERS_PER_METHOD)
                ],
            },
        )
        controllers[method] = [
            _start_controller(
                method,
                index + 1,
                assignments[method][index],
                output,
                controller_template,
                controller_wheel,
                configs[method],
                cohort["upstream_revision"],
                cohort,
                reuse_sources.get(method, {}),
            )
            for index in range(CONTROLLERS_PER_METHOD)
        ]
    status = {"run": run, "status": "running", "controllers": {}}
    save_json(output / "run-status.json", status)
    while True:
        done = 0
        controller_status = {}
        for method in METHODS:
            for number, sandbox in enumerate(controllers[method], start=1):
                intent = output / "controllers" / f"{method}-{number}.json"
                state = _poll_controller(sandbox, intent, output)
                key = f"{method}-{number}"
                controller_status[key] = state
                done += state.get("state") == "complete"
        status = {
            "run": run,
            "status": "complete" if done == len(METHODS) * CONTROLLERS_PER_METHOD else "running",
            "controllers_complete": done,
            "controllers_total": len(METHODS) * CONTROLLERS_PER_METHOD,
            "controllers": controller_status,
        }
        save_json(output / "run-status.json", status)
        if done == len(METHODS) * CONTROLLERS_PER_METHOD:
            break
        time.sleep(30)
    reports = {}
    for method in METHODS:
        method_results = []
        for number in (1, 2):
            summary = json.loads(
                (output / method / f"controller-{number}" / "controller-summary.json").read_text()
            )
            method_results.extend(summary["results"])
        reports[method] = {
            "attempted": len(method_results),
            "graded": sum(result["reward"] is not None for result in method_results),
            "resolved": sum(result["reward"] == 1 for result in method_results),
            "infrastructure_failed": sum(result["infrastructure_error"] is not None for result in method_results),
            "results": method_results,
        }
        save_json(output / method / "summary.json", {"method": method, **reports[method]})
    save_json(output / "final-summary.json", {"run": run, "methods": reports})
    save_json(output / "run-status.json", {"run": run, "status": "complete", "methods": reports})


@app.command("run-four-methods")
def run_four_methods(
    output: Path = typer.Option(Path("runs/deepswe-v1.1-e2b-four-methods-20261101")),
    upstream: Path = typer.Option(UPSTREAM),
    cohort: Path = typer.Option(COHORT),
    e2b_key_file: Path | None = typer.Option(None),
) -> None:
    """Run all 50 pinned tasks for all four methods on eight E2B controllers."""
    execute_four_methods(output, upstream, cohort, e2b_key_file)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "controller-run":
        controller_run(Path(sys.argv[2]))
    elif len(sys.argv) > 1 and sys.argv[1] == "controller-retry":
        controller_retry(Path(sys.argv[2]))
    else:
        app()
