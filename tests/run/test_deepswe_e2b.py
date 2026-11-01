import json

import pytest

from benchmarks.deepswe_common import UPSTREAM, load_cohort, load_task, method_config
from benchmarks.deepswe_e2b import (
    CONTROLLERS_PER_METHOD,
    WORKERS_PER_CONTROLLER,
    _verify_remote_sha256,
    controller_assignments,
    grader_prerequisite_commands,
    sha256,
)
from pocketcontextagent.comparisons.registry import METHODS


def test_controller_assignments_cover_the_pinned_cohort_once_per_method():
    cohort, tasks = load_cohort(UPSTREAM)
    assignments = controller_assignments(tasks)
    task_ids = [task.task_id for task in tasks]

    assert list(assignments) == list(METHODS)
    assert all(len(groups) == CONTROLLERS_PER_METHOD and [len(group) for group in groups] == [25, 25] for groups in assignments.values())
    assert all([task.task_id for group in groups for task in group] == task_ids for groups in assignments.values())
    assert all({task.task_id for task in groups[0]}.isdisjoint(task.task_id for task in groups[1]) for groups in assignments.values())
    assert cohort["task_ids_sha256"] == "7d65a50798d527f4c8473cdc9de842583b1a47227f98f56c899cfa74f145bb88"
    assert CONTROLLERS_PER_METHOD * WORKERS_PER_CONTROLLER * len(METHODS) == 48


def test_controller_assignments_reject_duplicate_or_incomplete_task_sets():
    _, tasks = load_cohort(UPSTREAM)

    with pytest.raises(ValueError, match="50 unique"):
        controller_assignments(tasks[:-1])
    with pytest.raises(ValueError, match="50 unique"):
        controller_assignments([*tasks[:-1], tasks[0]])


def test_remote_artifact_receipt_requires_the_uploaded_checksum():
    checksum = sha256(b"artifact")

    assert _verify_remote_sha256(f"{checksum}  /workspace/artifact.tar.gz", checksum) == checksum
    with pytest.raises(ValueError, match="checksum mismatch"):
        _verify_remote_sha256(f"{sha256(b'other')}  /workspace/artifact.tar.gz", checksum)


def test_grader_template_installs_the_pinned_task_reporter_prerequisite():
    task = load_task(UPSTREAM, "goreleaser-retry-publish-auditing")

    assert grader_prerequisite_commands(task) == [
        "env GOFLAGS= go install github.com/ctrf-io/go-ctrf-json-reporter/cmd/go-ctrf-json-reporter@v0.1.0"
    ]


def test_each_method_keeps_the_128k_deepswe_configuration():
    for method in METHODS:
        config = method_config(method)
        assert config["agent"]["context_token_limit"] == 128000
        assert config["environment"]["cwd"] == "/app"
        assert config["agent"]["agent_class"] == METHODS[method].agent_class
        assert len(sha256(json.dumps(config, sort_keys=True).encode())) == 64
