import json
from pathlib import Path

import pytest

from benchmarks.deepswe_daytona import COHORT, Task, grade_task, save_json, sha256
from pocketcontextagent.comparisons.registry import configure_method
from pocketcontextagent.config import get_config_from_spec


def test_pinned_cohort_and_react_prompt_use_the_deepswe_contract():
    cohort = json.loads(COHORT.read_text())
    assert len(cohort["task_ids"]) == len(set(cohort["task_ids"])) == 50
    assert sha256(("\n".join(cohort["task_ids"]) + "\n").encode()) == cohort["task_ids_sha256"]
    config = configure_method(get_config_from_spec("deepswe_context"), "react")
    assert config["agent"]["context_token_limit"] == 128000
    assert config["environment"]["cwd"] == "/app"
    assert "/app" in config["agent"]["system_template"]
    assert "/testbed" not in config["agent"]["system_template"]
    assert "commit your changes" in config["agent"]["system_template"]
    assert config["model"]["research_tools"] == "bash"


def test_existing_grade_is_bound_to_the_saved_patch_and_verifier_revision(tmp_path: Path):
    task = Task("example-task", "Fix it", "example/image", "a" * 40, tmp_path, 1800)
    directory = tmp_path / "tasks" / task.task_id
    result = directory / "inference" / f"{task.task_id}.json"
    save_json(result, {"query_id": task.task_id, "model_patch": "patch one"})
    report = {
        "task_id": task.task_id,
        "upstream_revision": "b" * 40,
        "patch_sha256": sha256(b"patch one"),
        "reward": {"reward": 0},
        "attempt": "tasks/example-task/grading-attempts/first",
    }
    save_json(directory / "grade.json", report)
    save_json(directory / "grading-attempts" / "first" / "cleanup.json", {"deleted": True})
    assert grade_task(task, tmp_path, result, "b" * 40) == report
    save_json(result, {"query_id": task.task_id, "model_patch": "patch two"})
    with pytest.raises(ValueError, match="does not match"):
        grade_task(task, tmp_path, result, "b" * 40)
