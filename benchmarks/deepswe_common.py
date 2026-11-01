"""Shared pinned DeepSWE task and configuration helpers."""

import hashlib
import json
import random
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from dotenv import load_dotenv

from pocketcontextagent.comparisons.registry import METHODS, configure_method
from pocketcontextagent.config import get_config_from_spec
from pocketcontextagent.utils.serialize import recursive_merge

COHORT = Path("evals/deepswe-v1.1-50-task-ids.json")
UPSTREAM = Path("data/deepswe/upstream")


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
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
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


def method_config(method: str) -> dict:
    load_dotenv(Path(".env"))
    config = recursive_merge(
        get_config_from_spec("deepswe_context"), get_config_from_spec("context_benchmark_deepseek")
    )
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    config = configure_method(config, method)
    if config["agent"]["context_token_limit"] != 128000 or config["environment"]["cwd"] != "/app":
        raise ValueError("DeepSWE requires the pinned 128K /app configuration")
    if not __import__("os").getenv("DEEPSEEK_API_KEY"):
        raise ValueError("DEEPSEEK_API_KEY is required for the configured DeepSWE model")
    return config


def record_run(
    output: Path,
    cohort: dict,
    config: dict,
    *,
    method: str,
    smoke: bool,
    task_ids: list[str],
    backend: str = "e2b",
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
        "upstream_requested_disk_gb": 20,
        "grader": "upstream DeepSWE v1.1 tests/test.sh and tests/grader.py; no LLM judge",
        "execution_backend": backend,
    }
    path = output / "run-manifest.json"
    if path.exists():
        current = json.loads(path.read_text())
        if current != manifest:
            raise ValueError("Output directory belongs to a different DeepSWE run")
        return
    save_json(path, manifest)
