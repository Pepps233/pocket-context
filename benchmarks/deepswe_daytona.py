"""Run the pinned DeepSWE v1.1 cohort with ReAct and its official verifier on Daytona."""

import hashlib
import json
import random
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

import typer
from dotenv import load_dotenv

from benchmarks.benchmark_matrix import bounded_command, run_instance
from benchmarks.browsecomp_plus_daytona import client, provider_environment
from benchmarks.utils.swebench_daytona import delete_resources
from pocketcontextagent.comparisons.registry import METHODS, configure_method
from pocketcontextagent.config import get_config_from_spec
from pocketcontextagent.environments.swebench_verified import SWEBenchVerifiedEnvironment
from pocketcontextagent.utils.serialize import recursive_merge

app = typer.Typer(add_completion=False, no_args_is_help=True)
COHORT = Path("evals/deepswe-v1.1-50-task-ids.json")
UPSTREAM = Path("data/deepswe/upstream")
DAYTONA_DISK_GB = 10


@dataclass(frozen=True)
class Task:
    task_id: str
    instruction: str
    image: str
    base_commit: str
    tests_dir: Path
    verifier_timeout: int


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def upstream_checkout(root: Path, revision: str) -> None:
    if not root.exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--filter=blob:none", "https://github.com/datacurve-ai/deep-swe", str(root)], check=True
        )
    current = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if current != revision:
        subprocess.run(["git", "-C", str(root), "fetch", "origin", revision], check=True)
        subprocess.run(["git", "-C", str(root), "checkout", "--detach", revision], check=True)
    if subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True).strip():
        raise ValueError("DeepSWE upstream checkout has local changes")


def load_cohort(root: Path, cohort_path: Path = COHORT) -> tuple[dict, list[Task]]:
    cohort = json.loads(cohort_path.read_text())
    upstream_checkout(root, cohort["upstream_revision"])
    all_ids = sorted(path.name for path in (root / "tasks").iterdir() if (path / "task.toml").is_file())
    ids = cohort["task_ids"]
    if (
        len(all_ids) != cohort["source_task_count"]
        or len(ids) != 50
        or cohort["task_count"] != 50
        or len(set(ids)) != len(ids)
        or ids != random.Random(cohort["selection_seed"]).sample(all_ids, len(ids))
        or sha256(("\n".join(ids) + "\n").encode()) != cohort["task_ids_sha256"]
    ):
        raise ValueError("Pinned DeepSWE cohort differs from the upstream task set")
    return cohort, [load_task(root, task_id) for task_id in ids]


def load_task(root: Path, task_id: str) -> Task:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", task_id):
        raise ValueError("Invalid DeepSWE task ID")
    directory = root / "tasks" / task_id
    config = tomllib.loads((directory / "task.toml").read_text())
    base = config["metadata"]["base_commit_hash"]
    if (
        config["metadata"]["task_id"] != task_id
        or config["verifier"]["environment_mode"] != "separate"
        or config["verifier"]["network_mode"] != "no-network"
        or not re.fullmatch(r"[0-9a-f]{40}", base)
        or json.loads((directory / "tests" / "config.json").read_text())["base_commit"] != base
    ):
        raise ValueError(f"Unexpected DeepSWE task contract: {task_id}")
    return Task(
        task_id=task_id,
        instruction=(directory / "instruction.md").read_text(),
        image=config["environment"]["docker_image"],
        base_commit=base,
        tests_dir=directory / "tests",
        verifier_timeout=int(config["verifier"]["timeout_sec"]),
    )


def method_config(method: str, *, smoke: bool = False) -> dict:
    config = recursive_merge(
        get_config_from_spec("deepswe_context"), get_config_from_spec("context_benchmark_deepseek")
    )
    if smoke:
        config["agent"]["step_limit"] = 2
    config = configure_method(config, method)
    if config["agent"]["context_token_limit"] != 128000 or config["environment"]["cwd"] != "/app":
        raise ValueError("DeepSWE requires the pinned 128K /app configuration")
    provider_environment("DEEPSEEK_API_KEY")
    return config


def record_run(
    output: Path, cohort: dict, config: dict, *, method: str, smoke: bool, task_ids: list[str], backend: str = "daytona"
) -> None:
    manifest = {
        "benchmark": "DeepSWE v1.1",
        "upstream_revision": cohort["upstream_revision"],
        "cohort_sha256": cohort["task_ids_sha256"],
        "task_ids": task_ids,
        "method": method,
        "context_token_limit": 128000,
        "model": config["model"]["model_name"],
        "config_sha256": sha256(json.dumps(config, sort_keys=True).encode()),
        "smoke": smoke,
        "agent_step_limit": config["agent"]["step_limit"],
        "daytona_disk_gb": DAYTONA_DISK_GB,
        "upstream_requested_disk_gb": 20,
        "grader": "upstream DeepSWE v1.1 tests/test.sh and tests/grader.py; no LLM judge",
    }
    if backend != "daytona":
        manifest["execution_backend"] = backend
    path = output / "run-manifest.json"
    if path.exists():
        current = json.loads(path.read_text())
        if current == manifest:
            return
        if current == {key: value for key, value in manifest.items() if key != "config_sha256"}:
            save_json(path, manifest)
            return
        raise ValueError("Output directory belongs to a different DeepSWE run")
    save_json(path, manifest)


def clean_sandbox(intent: Path, receipt: Path) -> None:
    if receipt.exists():
        return
    identity = json.loads(intent.read_text())
    if not identity.get("sandbox_id"):
        raise ValueError(f"Cannot verify sandbox cleanup without its ID: {intent}")
    delete_resources(client(), {"sandboxes": [{"name": identity["sandbox_name"], "id": identity["sandbox_id"]}]})
    save_json(
        receipt, {"sandbox_name": identity["sandbox_name"], "sandbox_id": identity["sandbox_id"], "deleted": True}
    )


def run_task(task: Task, output: Path, config: dict, revision: str, method: str = "react") -> Path:
    directory = output / "tasks" / task.task_id
    result_path = directory / "inference" / f"{task.task_id}.json"
    intent = directory / "inference-intent.json"
    if result_path.exists():
        clean_sandbox(intent, directory / "inference-cleanup.json")
        if json.loads(result_path.read_text()).get("execution_error"):
            raise RuntimeError(f"Inference recorded an execution error; inspect before grading: {result_path}")
        return result_path
    if intent.exists():
        raise RuntimeError(f"Started inference needs recovery before retry: {intent}")
    from daytona import CreateSandboxFromImageParams, Resources

    name = f"pca-deepswe-agent-{uuid4().hex[:12]}"
    save_json(intent, {"task_id": task.task_id, "sandbox_name": name, "upstream_revision": revision})
    api = client()
    sandbox = api.create(
        CreateSandboxFromImageParams(
            image=task.image,
            resources=Resources(cpu=2, memory=8, disk=DAYTONA_DISK_GB),
            name=name,
            os_user="root",
            public=False,
            network_block_all=True,
            labels={"project": "pocket-context", "benchmark": "deepswe", "method": method, "role": "agent"},
            auto_stop_interval=0,
            auto_pause_interval=0,
            auto_delete_interval=-1,
            ttl_minutes=240,
        ),
        timeout=1800,
    )
    save_json(
        intent, {"task_id": task.task_id, "sandbox_name": name, "sandbox_id": sandbox.id, "upstream_revision": revision}
    )
    try:

        def command_backend(command: str, *, cwd: str, timeout: int) -> dict:
            response = sandbox.process.exec(bounded_command(command, timeout), cwd=cwd, timeout=timeout + 15)
            return {"output": response.result, "returncode": response.exit_code}

        environment = SWEBenchVerifiedEnvironment(
            command_backend, **config["environment"], base_commit=task.base_commit
        )
        environment.prepare(fresh_isolated=True)
        if environment.capture_patch().strip() or environment._run("test ! -e /tests/grader.py")["returncode"]:
            raise RuntimeError("DeepSWE agent image is not clean or contains the hidden verifier")
        run_instance(
            {"query_id": task.task_id, "query": task.instruction},
            directory / "inference",
            config,
            environment,
            {"method": method, "config": config, "benchmark": "deepswe", "upstream_revision": revision},
            {"sandbox_id": sandbox.id, "image": task.image},
        )
    finally:
        if result_path.exists():
            clean_sandbox(intent, directory / "inference-cleanup.json")
    return result_path


def grade_task(task: Task, output: Path, result_path: Path, revision: str, method: str = "react") -> dict:
    directory = output / "tasks" / task.task_id
    result = json.loads(result_path.read_text())
    patch = result["model_patch"]
    if result["query_id"] != task.task_id:
        raise ValueError("Inference result belongs to another DeepSWE task")
    patch_hash = sha256(patch.encode())
    report_path = directory / "grade.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if report["patch_sha256"] != patch_hash or report["upstream_revision"] != revision:
            raise ValueError("Saved grade does not match the saved patch or verifier revision")
        completed_attempt = output / report["attempt"]
        clean_sandbox(completed_attempt / "intent.json", completed_attempt / "cleanup.json")
        return report
    from daytona import CreateSandboxFromImageParams, Image, Resources

    attempt = directory / "grading-attempts" / uuid4().hex
    attempt.mkdir(parents=True)
    name = f"pca-deepswe-grader-{uuid4().hex[:12]}"
    save_json(attempt / "intent.json", {"task_id": task.task_id, "sandbox_name": name, "patch_sha256": patch_hash})
    api = client()
    sandbox = api.create(
        CreateSandboxFromImageParams(
            image=Image.from_dockerfile(task.tests_dir / "Dockerfile"),
            resources=Resources(cpu=2, memory=8, disk=DAYTONA_DISK_GB),
            name=name,
            os_user="root",
            public=False,
            network_block_all=True,
            labels={"project": "pocket-context", "benchmark": "deepswe", "method": method, "role": "grader"},
            auto_stop_interval=0,
            auto_pause_interval=0,
            auto_delete_interval=-1,
            ttl_minutes=120,
        ),
        timeout=1800,
    )
    save_json(
        attempt / "intent.json",
        {"task_id": task.task_id, "sandbox_name": name, "sandbox_id": sandbox.id, "patch_sha256": patch_hash},
    )
    complete = False
    try:
        setup = sandbox.process.exec("mkdir -p /logs/artifacts /logs/verifier", timeout=30)
        if setup.exit_code:
            raise RuntimeError(f"Verifier setup failed: {setup.result}")
        sandbox.fs.upload_file(patch.encode(), "/logs/artifacts/model.patch")
        response = sandbox.process.exec(
            "/bin/bash /tests/test.sh > /logs/verifier/test-stdout.txt 2>&1",
            cwd="/app",
            timeout=task.verifier_timeout + 30,
        )
        save_json(attempt / "execution.json", {"exit_code": response.exit_code, "output": response.result})
        for remote, local in (
            ("/logs/verifier/reward.json", "reward.json"),
            ("/logs/verifier/test-stdout.txt", "test-stdout.txt"),
        ):
            sandbox.fs.download_file(remote, str(attempt / local))
        reward = json.loads((attempt / "reward.json").read_text())
        if not reward.get("apply_failed"):
            sandbox.fs.download_file("/logs/verifier/ctrf.json", str(attempt / "ctrf.json"))
        archive = sandbox.process.exec("tar -czf /tmp/deepswe-verifier.tar.gz -C /logs verifier", timeout=60)
        if archive.exit_code:
            raise RuntimeError(f"Verifier artifact archival failed: {archive.result}")
        sandbox.fs.download_file("/tmp/deepswe-verifier.tar.gz", str(attempt / "verifier.tar.gz"))
        if (
            response.exit_code
            or reward.get("reward") not in (0, 1)
            or (reward.get("apply_failed") and reward["reward"] != 0)
        ):
            raise RuntimeError(f"DeepSWE verifier did not produce a conclusive score: {attempt}")
        report = {
            "task_id": task.task_id,
            "upstream_revision": revision,
            "patch_sha256": patch_hash,
            "reward": reward,
            "attempt": str(attempt.relative_to(output)),
            "verifier_artifacts_sha256": sha256((attempt / "verifier.tar.gz").read_bytes()),
        }
        save_json(report_path, report)
        complete = True
        return report
    finally:
        if complete:
            clean_sandbox(attempt / "intent.json", attempt / "cleanup.json")


def execute(
    output: Path, upstream: Path, cohort_path: Path, task_id: str | None, method: str, smoke: bool, workers: int
) -> None:
    load_dotenv(Path(".env"))
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    cohort, tasks = load_cohort(upstream, cohort_path)
    if task_id:
        tasks = [task for task in tasks if task.task_id == task_id]
        if not tasks:
            raise ValueError("Task ID is not in the pinned 50-task DeepSWE cohort")
    elif smoke:
        tasks = [next(task for task in tasks if task.task_id == "fastapi-deprecation-response-headers")]
    if not 1 <= workers <= 12:
        raise ValueError("Use 1 to 12 concurrent task workers")
    config = method_config(method, smoke=smoke)
    record_run(output, cohort, config, method=method, smoke=smoke, task_ids=[task.task_id for task in tasks])

    def one(task: Task) -> dict:
        result = run_task(task, output, config, cohort["upstream_revision"], method)
        return grade_task(task, output, result, cohort["upstream_revision"], method)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        reports = list(executor.map(one, tasks))
    summary = {
        "benchmark": "DeepSWE v1.1",
        "method": method,
        "context_token_limit": 128000,
        "task_count": len(tasks),
        "graded": len(reports),
        "resolved": sum(report["reward"]["reward"] == 1 for report in reports),
        "results": [{"task_id": report["task_id"], "reward": report["reward"]["reward"]} for report in reports],
    }
    save_json(output / "summary.json", summary)
    typer.echo(json.dumps(summary, indent=2))


@app.command("run")
def run(
    output: Path | None = typer.Option(None),
    upstream: Path = typer.Option(UPSTREAM),
    cohort: Path = typer.Option(COHORT),
    task_id: str | None = typer.Option(None),
    method: str = typer.Option("react", help="One of pocket_context, react, resum, acm."),
    smoke: bool = typer.Option(False, help="Run one pinned task with two model calls, then the official verifier."),
    workers: int = typer.Option(6),
) -> None:
    """Run one method and the official DeepSWE verifier for the pinned cohort."""
    target = output or Path(f"runs/deepswe-v1.1-{method}-{'smoke' if smoke else task_id or '50'}")
    execute(target, upstream, cohort, task_id, method, smoke, workers)


if __name__ == "__main__":
    app()
