import json
from datetime import datetime, timedelta, timezone

import pytest

from pocketcontextagent.run.benchmarks.benchmark_matrix import benchmark_config
from pocketcontextagent.run.benchmarks.utils.swebench_daytona import (
    ResourceJournal,
    RuntimeConfig,
    immutable_image,
    sandbox_metadata,
)


def metadata() -> dict:
    return {
        "cpu": 2,
        "memory": 8,
        "disk": 10,
        "auto_stop_interval": 0,
        "auto_pause_interval": 0,
        "auto_delete_interval": -1,
        "auto_destroy_at": datetime.now(timezone.utc) + timedelta(hours=4),
    }


def test_lifecycle_requires_no_idle_shutdown_and_verifies_real_deadline():
    runtime = RuntimeConfig()
    assert sandbox_metadata(metadata(), runtime)["auto_stop_interval"] == 0
    assert sandbox_metadata(metadata(), runtime)["auto_pause_interval"] == 0
    for key, value in [("auto_stop_interval", 30), ("auto_pause_interval", 5), ("auto_delete_interval", 0)]:
        with pytest.raises(RuntimeError, match="lifecycle"):
            sandbox_metadata(metadata() | {key: value}, runtime)
    with pytest.raises(RuntimeError, match="shorter"):
        sandbox_metadata(metadata() | {"auto_destroy_at": datetime.now(timezone.utc) + timedelta(minutes=30)}, runtime)
    with pytest.raises(RuntimeError, match="not applied"):
        sandbox_metadata(metadata() | {"auto_destroy_at": None}, runtime)
    with pytest.raises(RuntimeError, match="resources"):
        sandbox_metadata(metadata() | {"memory": 4}, runtime)
    assert (
        sandbox_metadata(metadata() | {"auto_destroy_at": None}, RuntimeConfig(ttl_minutes=0))["auto_destroy_at"]
        is None
    )
    with pytest.raises(RuntimeError, match="Organization policy"):
        sandbox_metadata(metadata(), RuntimeConfig(ttl_minutes=0))


def test_official_digest_is_immutable_and_arbitrary_registry_inputs_are_rejected():
    image = "docker.io/swebench/sweb.eval.x86_64.fixture_1776_repo-1@sha256:" + "a" * 64
    assert immutable_image(image) == image
    for invalid in [
        "http://127.0.0.1/image",
        "docker.io/untrusted/image:latest",
        "docker.io/swebench/image:stable",
        image[:-1],
    ]:
        with pytest.raises(ValueError, match="canonical official"):
            immutable_image(invalid)


def test_resource_journal_persists_intended_names_before_ids_and_retains_errors(tmp_path):
    path = tmp_path / "cleanup.json"
    journal = ResourceJournal(path)
    journal.record("sandboxes", "planned-before-create", phase="creating", method="acm")
    assert json.loads(path.read_text())["sandboxes"] == [
        {"name": "planned-before-create", "phase": "creating", "method": "acm"}
    ]
    with pytest.raises(FileNotFoundError) as error:
        (tmp_path / "absent").read_text()
    journal.failure("sandboxes", "planned-before-create", error.value)
    journal.record("sandboxes", "planned-before-create", id="returned-id", cleanup_confirmed=True)
    journal.record("snapshots", "owned-snapshot", image="immutable-reference", phase="active")
    saved = json.loads(path.read_text())
    assert len(saved["sandboxes"]) == 1 and saved["sandboxes"][0]["id"] == "returned-id"
    assert saved["sandboxes"][0]["error_type"] == "FileNotFoundError"
    assert saved["sandboxes"][0]["cleanup_confirmed"] is True
    assert saved == journal.document() and not path.with_suffix(".json.tmp").exists()


def test_runtime_config_keeps_long_task_budget_below_absolute_lifetime():
    config = benchmark_config([])
    assert config["agent"]["wall_time_limit_seconds"] == 7200
    assert config["sandbox"]["ttl_minutes"] == 240
    assert config["agent"]["cost_limit"] == 0.5 and config["agent"]["step_limit"] == 100
    with pytest.raises(ValueError, match="exceed the task"):
        benchmark_config(["context_benchmark_deepseek", "sandbox.ttl_minutes=120"])
    assert (
        benchmark_config(
            ["context_benchmark_deepseek", "agent.wall_time_limit_seconds=14400", "sandbox.ttl_minutes=300"]
        )["sandbox"]["ttl_minutes"]
        == 300
    )
    for fields in [{"ttl_minutes": -1}, {"disk": 0}, {"snapshot_build_workers": 11}]:
        with pytest.raises(ValueError):
            RuntimeConfig(**fields)
