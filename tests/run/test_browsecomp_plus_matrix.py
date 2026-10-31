import json
import os
import random
import shlex
import threading
import time
from collections import Counter
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from benchmarks.browsecomp_plus import app
from benchmarks.browsecomp_plus_daytona import (
    matrix_arguments,
    matrix_config,
    matrix_provider_key,
    matrix_worker_groups,
    provider_environment,
    run_matrix_workers,
    select_matrix_instances,
    worker_command,
)
from pocketcontextagent.config import builtin_config_dir


@pytest.fixture
def queries(tmp_path: Path) -> Path:
    held_out = json.loads((builtin_config_dir / "benchmarks/browsecomp_plus_eval150.json").read_text())["query_ids"][:5]
    path = tmp_path / "queries.jsonl"
    path.write_text(
        "".join(
            json.dumps({"query_id": str(iid), "query": "Which city?", "answer": "SECRET"}) + "\n"
            for iid in [*held_out, "development"]
        )
    )
    return path


def test_matrix_selection_is_shared_reproducible_and_excludes_answers(queries: Path):
    selected = select_matrix_instances(queries, "eval150", [], 3, 17)
    assert selected == random.Random(17).sample(select_matrix_instances(queries, "eval150", [], 0, None), 3)
    assert len({row["query_id"] for row in selected}) == 3
    assert all(set(row) == {"query_id", "query"} for row in selected)
    assert "SECRET" not in json.dumps(selected)
    assert select_matrix_instances(queries, "eval150", [row["query_id"] for row in selected], 0, None)
    assert len(select_matrix_instances(queries, "development680", [], 0, 1)) == 1
    with pytest.raises(ValueError, match="outside the selected split"):
        select_matrix_instances(queries, "eval150", [selected[0]["query_id"], "development"], 0, None)
    with pytest.raises(ValueError, match="between 0 and 5"):
        select_matrix_instances(queries, "eval150", [], 6, 1)
    with pytest.raises(ValueError, match="Unknown query IDs"):
        select_matrix_instances(queries, "all", ["missing"], 0, None)


def test_matrix_dry_run_needs_no_pool_and_writes_nothing(queries: Path, tmp_path: Path):
    output = tmp_path / "untouched"
    result = CliRunner().invoke(
        app,
        [
            "daytona",
            "matrix",
            "--data-dir",
            str(tmp_path),
            "--pool",
            str(tmp_path / "missing.json"),
            "--split",
            "eval150",
            "--seed",
            "42",
            "--limit",
            "2",
            "--output",
            str(output),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    selection = yaml.safe_load(result.output)
    assert selection["split"] == "eval150" and selection["seed"] == 42
    assert len(selection["selected_query_ids"]) == 2
    assert not output.exists() and "SECRET" not in result.output
    assert selection["workers_per_method"] == selection["active_workers_per_method"] == 1


def test_matrix_config_replaces_provider_default_instead_of_layering():
    assert matrix_config([])["model"]["model_name"] == "deepseek/deepseek-flash"
    assert matrix_provider_key(matrix_config([])) == "DEEPSEEK_API_KEY"
    meta = matrix_config(["browsecomp_plus_muse_spark_1_3_contributor"])
    assert meta["model"]["model_name"] == "meta/muse-spark-1.3-contributor"
    assert meta["model"]["model_class"] == "litellm_response"
    assert matrix_provider_key(meta) == "META_API_KEY"
    assert "thinking" not in meta["model"]["model_kwargs"] and "extra_body" not in meta["model"]["model_kwargs"]
    router = matrix_config(["browsecomp_plus_openrouter"])
    assert router["model"]["model_class"] == "openrouter"
    assert matrix_provider_key(router) == "OPENROUTER_API_KEY"
    with pytest.raises(ValueError, match="supports"):
        matrix_provider_key({"model": {"model_name": "unsupported/model"}})
    with pytest.raises(ValueError, match="provider profile"):
        matrix_provider_key(matrix_config(["agent.step_limit=2"]))
    assert "api_base" not in router["model"]["model_kwargs"]


def test_matrix_never_overwrites_recorded_attempts(queries: Path, tmp_path: Path):
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "recorded.json"
    sentinel.write_text('{"status":"LimitsExceeded"}')
    result = CliRunner().invoke(
        app, ["daytona", "matrix", "--data-dir", str(tmp_path), "--split", "eval150", "--output", str(output)]
    )
    assert isinstance(result.exception, ValueError) and "never replaces or retries" in str(result.exception)
    assert sentinel.read_text() == '{"status":"LimitsExceeded"}'
    assert list(output.iterdir()) == [sentinel]


def test_matrix_worker_command_preserves_method_split_and_safe_paths():
    arguments = matrix_arguments("/workspace/path with spaces", "acm", "eval150")
    assert arguments[arguments.index("--method") + 1] == "acm"
    assert arguments[arguments.index("--split") + 1] == "eval150"
    assert arguments[arguments.index("--workers") + 1] == "1"
    assert shlex.split(worker_command(arguments))[3:] == arguments
    assert arguments[arguments.index("--queries") + 1] == "/workspace/path with spaces/query.jsonl"
    for key in ("DAYTONA_API_KEY", "invalid", "OPENROUTER_API_KEY;echo secret"):
        with pytest.raises(ValueError, match="provider"):
            provider_environment(key)
    key = "POCKET_MATRIX_MISSING_" + str(os.getpid()) + "_API_KEY"
    assert key not in os.environ
    with pytest.raises(ValueError, match=key):
        provider_environment(key)


@pytest.mark.parametrize(
    ("invalid_pool", "message"),
    [
        ([], "4 distinct"),
        ([{"id": "same"}] * 4, "4 distinct"),
        ([{"id": str(i)} for i in range(4)], "credentials do not match"),
    ],
)
def test_matrix_rejects_bad_pools_before_any_remote_call(
    queries: Path, tmp_path: Path, invalid_pool: list, message: str
):
    pool = tmp_path / "pool.json"
    pool.write_text(json.dumps({"sandboxes": invalid_pool}))
    result = CliRunner().invoke(
        app,
        [
            "daytona",
            "matrix",
            "--data-dir",
            str(tmp_path),
            "--pool",
            str(pool),
            "--split",
            "eval150",
            "--output",
            str(tmp_path / "output"),
        ],
    )
    assert isinstance(result.exception, ValueError) and message in str(result.exception)
    assert not (tmp_path / "output").exists()


def test_matrix_allocates_disjoint_workers_and_caps_to_selected_tasks():
    manifest = {"sandboxes": [{"id": str(i)} for i in range(120)]}
    groups = matrix_worker_groups(manifest, 30, 150)
    assert len(groups) == 4 and all(len(workers) == 30 for workers in groups.values())
    assert len({worker["id"] for workers in groups.values() for worker in workers}) == 120
    assert all(len(workers) == 3 for workers in matrix_worker_groups(manifest, 30, 3).values())
    with pytest.raises(ValueError, match="12 distinct"):
        matrix_worker_groups({"sandboxes": manifest["sandboxes"][:11]}, 30, 3)
    with pytest.raises(ValueError, match="120 distinct"):
        matrix_worker_groups({"sandboxes": [{"id": "same"}] * 120}, 30, 150)
    for requested, tasks in [(0, 1), (31, 1), (30, 0)]:
        with pytest.raises(ValueError, match="1 to 30"):
            matrix_worker_groups(manifest, requested, tasks)


@pytest.mark.parametrize(("workers_per_method",), [(1,), (2,), (30,)])  # noqa: PT006
def test_matrix_scheduler_runs_once_per_task_with_one_active_task_per_sandbox(workers_per_method: int):
    groups = matrix_worker_groups(
        {"sandboxes": [{"id": str(i)} for i in range(4 * workers_per_method)]},
        workers_per_method,
        workers_per_method + 2,
    )
    instances = [{"query_id": str(i)} for i in range(workers_per_method + 2)]
    lock = threading.Lock()
    barrier = threading.Barrier(4 * workers_per_method)
    active_workers = set()
    seen_workers = set()
    active = Counter()
    peak = Counter()
    records = []

    def execute(method: str, worker: dict, instance: dict) -> None:
        with lock:
            assert worker["id"] not in active_workers
            active_workers.add(worker["id"])
            active[method] += 1
            peak[method] = max(peak[method], active[method])
            first = worker["id"] not in seen_workers
            seen_workers.add(worker["id"])
        if first:
            barrier.wait(timeout=30)
        time.sleep(0.001)
        with lock:
            records.append((method, instance["query_id"]))
            active_workers.remove(worker["id"])
            active[method] -= 1

    run_matrix_workers(groups, instances, execute)
    assert Counter(records) == Counter((method, instance["query_id"]) for method in groups for instance in instances)
    assert all(count == workers_per_method for count in peak.values())
    assert len(seen_workers) == 4 * workers_per_method and not active_workers


def test_matrix_free_worker_claims_next_task_without_waiting_for_a_slow_worker():
    blocked = threading.Event()
    next_task = threading.Event()
    assignments = {}

    def execute(method: str, worker: dict, instance: dict) -> None:
        assignments[instance["query_id"]] = worker["id"]
        if instance["query_id"] == "0":
            blocked.set()
            assert next_task.wait(timeout=10)
        elif instance["query_id"] == "1":
            assert blocked.wait(timeout=10)
        else:
            next_task.set()

    run_matrix_workers({"react": [{"id": "slow"}, {"id": "fast"}]}, [{"query_id": str(i)} for i in range(3)], execute)
    assert set(assignments) == {"0", "1", "2"}
    assert assignments["0"] != assignments["1"] == assignments["2"]


def test_parallel_matrix_dry_run_reports_effective_concurrency_without_a_pool(queries: Path, tmp_path: Path):
    result = CliRunner().invoke(
        app,
        [
            "daytona",
            "matrix",
            "--data-dir",
            str(tmp_path),
            "--split",
            "eval150",
            "--limit",
            "2",
            "--workers-per-method",
            "30",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    selection = yaml.safe_load(result.output)
    assert selection["workers_per_method"] == 30 and selection["active_workers_per_method"] == 2
    arguments = matrix_arguments("/workspace/task", "acm", "eval150", query_id="1055", workers=30)
    assert arguments[arguments.index("--execute-query-id") + 1] == "1055"
    assert arguments[arguments.index("--workers") + 1] == "30"
    assert "--query-id" not in arguments
    for count in ("0", "31"):
        invalid = CliRunner().invoke(app, ["daytona", "matrix", "--workers-per-method", count, "--dry-run"])
        assert invalid.exit_code == 2
    assert CliRunner().invoke(app, ["daytona", "create", "--workers", "121"]).exit_code == 2


@pytest.mark.parametrize(("explicit_method",), [(False,), (True,)])  # noqa: PT006
def test_distributed_execution_requires_a_valid_method_and_selected_id(
    queries: Path, tmp_path: Path, explicit_method: bool
):
    selected = select_matrix_instances(queries, "eval150", [], 1, None)[0]["query_id"]
    arguments = [
        "run",
        "--queries",
        str(queries),
        "--index",
        str(tmp_path / "missing-index"),
        "--output",
        str(tmp_path / "output"),
        "--model",
        "replay",
        "--split",
        "eval150",
        "--execute-query-id",
        "missing" if explicit_method else selected,
    ]
    if explicit_method:
        arguments += ["--method", "react"]
    result = CliRunner().invoke(app, arguments)
    assert isinstance(result.exception, ValueError) and "Distributed execution requires" in str(result.exception)
    assert not (tmp_path / "output").exists()
