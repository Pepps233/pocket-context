import json
import os
import random
import shlex
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from pocketcontextagent.config import builtin_config_dir
from pocketcontextagent.run.benchmarks.browsecomp_plus import app
from pocketcontextagent.run.benchmarks.browsecomp_plus_daytona import (
    matrix_arguments,
    matrix_config,
    matrix_provider_key,
    provider_environment,
    select_matrix_instances,
    worker_command,
)


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
        ([], "four distinct"),
        ([{"id": "same"}] * 4, "four distinct"),
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
