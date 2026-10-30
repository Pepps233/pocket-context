"""Offline preparation contracts; all question text and grading values are synthetic."""

import ast
import copy
import hashlib
import inspect
import json
import os
import random
import subprocess
import sys
import textwrap
from dataclasses import fields
from pathlib import Path

import pytest
import yaml
from jinja2 import StrictUndefined, Template

from pocketcontextagent.comparisons.registry import METHODS, configure_method, method_provenance
from pocketcontextagent.config import builtin_config_dir
from pocketcontextagent.environments.browsecomp_plus import BrowseCompPlusConfig, BrowseCompPlusEnvironment
from pocketcontextagent.run.benchmarks.benchmark_controller import submission_plan
from pocketcontextagent.run.benchmarks.benchmark_matrix import load_instances, pin_instances, run_manifest, run_matrix
from pocketcontextagent.run.benchmarks.browsecomp_plus_daytona import matrix_query_bytes
from pocketcontextagent.run.benchmarks.utils import controller_runtime
from pocketcontextagent.run.benchmarks.utils import swebench_verified as swe

PRIVATE = "SYNTHETIC_PRIVATE_GRADING_76391"
TOKENIZER_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
PROFILES = {
    "browsecomp-plus": ["browsecomp_plus_deepseek", "context_full_browsecomp"],
    "swebench-verified": ["context_benchmark_deepseek", "context_full_swebench"],
}
OFFLINE_CLI = """
import runpy
import sys

forbidden = []
def audit(event, args):
    if event == 'socket.bind':
        raise RuntimeError('Socket binding forbidden, including dependency capability probes')
    if event in {
        'socket.connect', 'socket.connect_ex', 'socket.getaddrinfo', 'socket.gethostbyname',
        'socket.gethostbyaddr', 'socket.sendto', 'socket.sendmsg',
        'subprocess.Popen', 'os.system', 'os.exec', 'os.posix_spawn',
    }:
        forbidden.append(event)
        raise AssertionError('External operation forbidden: ' + event)
    if event == 'import' and args[0].split('.')[0] in {'daytona', 'litellm', 'openai', 'anthropic', 'transformers'}:
        forbidden.append(args[0])
        raise AssertionError('API/model import forbidden: ' + args[0])

def profile(frame, event, arg):
    module = frame.f_globals.get('__name__', '')
    if event == 'call' and module.startswith('pocketcontextagent.') and frame.f_code.co_name in {
        'get_model', 'client', '_load', '_credentials', 'provider_environment', 'from_pretrained'
    }:
        forbidden.append(module + '.' + frame.f_code.co_name)
        raise AssertionError('API/model access forbidden: ' + forbidden[-1])

sys.addaudithook(audit)
sys.setprofile(profile)
sys.argv = ['mini-extra', 'benchmark-controller', *sys.argv[1:]]
try:
    runpy.run_module('pocketcontextagent.run.utilities.mini_extra', run_name='__main__')
finally:
    sys.setprofile(None)
    assert not forbidden, forbidden
"""


@pytest.fixture
def prepared_domains(tmp_path: Path) -> dict[str, Path]:
    paths = {benchmark: tmp_path / "data" / benchmark for benchmark in PROFILES}
    for path in paths.values():
        path.mkdir(parents=True)
    ids = json.loads((builtin_config_dir / "benchmarks/browsecomp_plus_eval150.json").read_text())["query_ids"][:2]
    (paths["browsecomp-plus"] / "queries.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "query_id": str(iid),
                    "query": f"Synthetic public fixture question {index}?",
                    "answer": PRIVATE,
                    "evidence": [PRIVATE],
                    "unexpected_private_field": PRIVATE,
                }
            )
            + "\n"
            for index, iid in enumerate([*ids, "synthetic_development_query"])
        )
    )
    rows = [
        {
            "instance_id": iid,
            "repo": repo,
            "base_commit": "a" * 40,
            "version": "1",
            "problem_statement": f"Synthetic public fixture issue {index}.",
            "patch": PRIVATE,
            "test_patch": PRIVATE,
            "FAIL_TO_PASS": [PRIVATE],
            "PASS_TO_PASS": [PRIVATE],
            "hints_text": PRIVATE,
            "image": "untrusted.invalid/private:latest",
        }
        for index, (iid, repo) in enumerate(
            [
                ("astropy__astropy-12907", "astropy/astropy"),
                ("django__django-11099", "django/django"),
            ]
        )
    ]
    data = paths["swebench-verified"]
    for name in ("queries.jsonl", "grading_dataset.jsonl"):
        (data / name).write_text("".join(json.dumps(row) + "\n" for row in rows))
    (data / "manifest.json").write_text(
        json.dumps(
            {
                "dataset": swe.DATASET_NAME,
                "dataset_revision": swe.DATASET_REVISION,
                "harness_revision": swe.HARNESS_REVISION,
                "split": "test",
                **{
                    f"{name}_sha256": hashlib.sha256((data / f"{name}.jsonl").read_bytes()).hexdigest()
                    for name in ("queries", "grading_dataset")
                },
            }
        )
    )
    return paths


def snapshot(root: Path) -> dict[str, bytes | None]:
    return {str(path.relative_to(root)): path.read_bytes() if path.is_file() else None for path in root.rglob("*")}


def offline_cli(root: Path, arguments: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-B", "-c", OFFLINE_CLI, *arguments],
        cwd=root,
        env={key: value for key, value in os.environ.items() if not key.endswith(("_API_KEY", "_TOKEN"))}
        | {
            "MSWEA_SILENT_STARTUP": "1",
            "LITELLM_LOCAL_MODEL_COST_MAP": "True",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "NO_COLOR": "1",
            "TERM": "dumb",
            "COLUMNS": "240",
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )


def test_default_swe_submission_stays_compatible_and_public(prepared_domains: dict[str, Path], tmp_path: Path) -> None:
    before = snapshot(tmp_path)
    data = prepared_domains["swebench-verified"]
    implicit = submission_plan(data, [], 0, 42, 1, [])
    assert implicit == submission_plan(data, [], 0, 42, 1, [], benchmark="swebench-verified", split="test")
    config, plan = implicit
    assert config["benchmark"] == plan["benchmark"] == "swebench-verified"
    assert config["model"]["model_name"] == "deepseek/deepseek-flash"
    assert plan["selected_query_ids"] == random.Random(42).sample(["astropy__astropy-12907", "django__django-11099"], 2)
    assert plan["planned_attempts"] == plan["maximum_active_coding_sandboxes"] * 2 == 8
    instances = load_instances(data, query_ids=[], limit=0, seed=42)
    assert all(set(row) == set(swe.PUBLIC_FIELDS) for row in instances)
    assert all(row["image"] == swe.instance_image(row["query_id"]) for row in instances)
    assert PRIVATE not in json.dumps([config, plan, instances])
    assert snapshot(tmp_path) == before


def test_browsecomp_selection_has_exact_public_rows_and_worker_independent_cohort(
    prepared_domains: dict[str, Path],
    tmp_path: Path,
) -> None:
    before = snapshot(tmp_path)
    data = prepared_domains["browsecomp-plus"]
    config, plan = submission_plan(data, PROFILES["browsecomp-plus"], 0, 42, 1, [], benchmark="browsecomp-plus")
    wide_config, wide = submission_plan(
        data,
        PROFILES["browsecomp-plus"],
        0,
        42,
        30,
        [],
        benchmark="browsecomp-plus",
        split="eval150",
    )
    raw = [json.loads(line) for line in (data / "queries.jsonl").read_text().splitlines()]
    expected = random.Random(42).sample([{key: row[key] for key in ("query_id", "query")} for row in raw[:2]], 2)
    assert config == wide_config and plan["benchmark"] == "browsecomp-plus" and plan["split"] == "eval150"
    assert plan["instances"] == wide["instances"] == expected
    assert plan["selected_query_ids"] == wide["selected_query_ids"] == [row["query_id"] for row in expected]
    assert plan["queries_sha256"] == hashlib.sha256(matrix_query_bytes(expected)).hexdigest()
    assert plan["queries_sha256"] != hashlib.sha256((data / "queries.jsonl").read_bytes()).hexdigest()
    assert plan["maximum_active_sandboxes"] == 4 and wide["maximum_active_sandboxes"] == 8
    assert plan["planned_attempts"] == wide["planned_attempts"] == 8
    assert PRIVATE not in json.dumps([config, plan, wide])
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize(("benchmark", "wall_time"), [("browsecomp-plus", 1800), ("swebench-verified", 7200)])
def test_full_profiles_share_safe_budgets_policy_and_provenance_across_all_methods(
    prepared_domains: dict[str, Path],
    benchmark: str,
    wall_time: int,
) -> None:
    config, selection = submission_plan(
        prepared_domains[benchmark],
        PROFILES[benchmark],
        0,
        42,
        2,
        [],
        benchmark=benchmark,
    )
    wide_config, wide_selection = submission_plan(
        prepared_domains[benchmark],
        PROFILES[benchmark],
        0,
        42,
        30,
        [],
        benchmark=benchmark,
    )
    base_config, _ = submission_plan(
        prepared_domains[benchmark],
        PROFILES[benchmark][:1],
        0,
        42,
        2,
        [],
        benchmark=benchmark,
    )
    assert config == wide_config and selection["selected_query_ids"] == wide_selection["selected_query_ids"]
    original = copy.deepcopy(config)
    assert set(METHODS) == {"pocket_context", "react", "resum", "acm"}
    for method in METHODS:
        configured = configure_method(config, method)
        agent = configured["agent"]
        assert agent["step_limit"] == 0
        assert agent["cost_limit"] == 1.0 > 0
        assert agent["wall_time_limit_seconds"] == wall_time > 0
        assert agent["context_token_limit"] == 128000
        assert configured["model"]["model_name"] == config["model"]["model_name"] == "deepseek/deepseek-flash"
        assert configured["model"]["model_kwargs"] == config["model"]["model_kwargs"]
        assert configured["model"]["cost_tracking"] == "default"
        provenance = method_provenance(configured, method)
        assert provenance["source"] and provenance["revision"]
        assert provenance == method_provenance(configure_method(base_config, method), method)
        assert len(provenance["prompts_sha256"]) == 64
        assert provenance["auxiliary_model"] == "same model and endpoint as main policy"
        assert "not reproduction of published scores" in provenance["evaluation"]
        assert "api_key" not in configured["model"].get("model_kwargs", {})
        if benchmark == "browsecomp-plus":
            assert configured["environment"]["tokenizer_name"] == "Qwen/Qwen3.5-9B"
            assert configured["environment"]["tokenizer_revision"] == TOKENIZER_REVISION
        else:
            instances = load_instances(
                prepared_domains[benchmark], query_ids=selection["selected_query_ids"], limit=0, seed=42
            )
            manifest = run_manifest(config, method, instances, 2, prepared_domains[benchmark] / "queries.jsonl", "test")
            assert manifest["dataset_revision"] == swe.DATASET_REVISION
            assert manifest["harness_revision"] == swe.HARNESS_REVISION
            assert manifest["method_provenance"] == provenance
            assert manifest["selected_query_ids"] == [row["query_id"] for row in instances]
            assert PRIVATE not in json.dumps(manifest)
    assert config == original


@pytest.mark.parametrize(("benchmark",), [("browsecomp-plus",), ("swebench-verified",)])  # noqa: PT006
def test_unlimited_observations_do_not_invent_negative_remaining_calls(
    prepared_domains: dict[str, Path],
    benchmark: str,
) -> None:
    config, _ = submission_plan(prepared_domains[benchmark], PROFILES[benchmark], 0, 42, 1, [], benchmark=benchmark)
    template = Template(config["model"]["observation_template"], undefined=StrictUndefined)
    for method in METHODS:
        assert (
            configure_method(config, method)["model"]["observation_template"] == config["model"]["observation_template"]
        )
    output = {"output": "Synthetic command output", "returncode": 0}
    unlimited = template.render(output=output, step_limit=0, n_model_calls=7)
    assert output["output"] in unlimited
    assert "-7" not in unlimited and "Model calls remaining" not in unlimited
    assert "Model calls remaining: 2" in template.render(output=output, step_limit=9, n_model_calls=7)


def test_tokenizer_revision_constructor_and_runtime_schema_are_pinned_without_loading_models() -> None:
    assert "tokenizer_revision" in {field.name for field in fields(BrowseCompPlusConfig)}
    assert BrowseCompPlusConfig(tokenizer_revision=TOKENIZER_REVISION).tokenizer_revision == TOKENIZER_REVISION
    assert inspect.signature(BrowseCompPlusEnvironment).parameters["tokenizer_revision"].default is None
    constructor = ast.parse(textwrap.dedent(inspect.getsource(BrowseCompPlusEnvironment.__init__)))
    tokenizer_calls = [
        node
        for node in ast.walk(constructor)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "AutoTokenizer"
        and node.func.attr == "from_pretrained"
    ]
    assert len(tokenizer_calls) == 1
    assert any(
        keyword.arg == "revision" and isinstance(keyword.value, ast.Name) and keyword.value.id == "tokenizer_revision"
        for keyword in tokenizer_calls[0].keywords
    )
    assert {"transformers==5.20.2", "tokenizers==0.23.3", "huggingface-hub==1.33.0"} <= set(
        controller_runtime.RUNTIME_REQUIREMENTS
    )
    assert ("transformers", "AutoTokenizer") in controller_runtime._RUNTIME_IMPORTS


def test_image_pins_are_canonical_copied_and_allow_superset_maps(prepared_domains: dict[str, Path]) -> None:
    instances = load_instances(prepared_domains["swebench-verified"], query_ids=[], limit=0, seed=42)
    original = copy.deepcopy(instances)
    images = {row["query_id"]: row["image"].removesuffix(":latest") + "@sha256:" + "a" * 64 for row in instances}
    images["unselected_task"] = "unused values must not constrain this cohort"
    image_copy = images.copy()
    pinned = pin_instances(instances, images)
    assert pinned is not instances and all(new is not old for new, old in zip(pinned, instances))
    assert pinned == [row | {"image": images[row["query_id"]]} for row in instances]
    assert instances == original and images == image_copy
    assert pin_instances(instances, None) == original


@pytest.mark.parametrize(
    ("suffix",),  # noqa: PT006
    [
        (":latest",),
        ("@sha256:" + "a" * 63,),
        ("@sha256:" + "a" * 65,),
        ("@sha256:" + "g" * 64,),
        ("@sha256:" + "A" * 64,),
        ("@sha256:" + "a" * 64 + "\n",),
    ],
)
def test_invalid_digest_pins_fail_without_mutating_instances(prepared_domains: dict[str, Path], suffix: str) -> None:
    instances = load_instances(prepared_domains["swebench-verified"], query_ids=[], limit=1, seed=42)
    original = copy.deepcopy(instances)
    with pytest.raises(ValueError):
        pin_instances(instances, {instances[0]["query_id"]: instances[0]["image"].removesuffix(":latest") + suffix})
    assert instances == original


@pytest.mark.parametrize(
    ("invalid",),  # noqa: PT006
    [({},), ([],), ("invalid",), (7,), (False,), ({"value": None},), ({"value": 12},)],
)
def test_missing_or_mistyped_image_maps_raise_value_error(prepared_domains: dict[str, Path], invalid: object) -> None:
    instances = load_instances(prepared_domains["swebench-verified"], query_ids=[], limit=1, seed=42)
    original = copy.deepcopy(instances)
    images = (
        {instances[0]["query_id"]: invalid["value"]} if isinstance(invalid, dict) and "value" in invalid else invalid
    )
    with pytest.raises(ValueError):
        pin_instances(instances, images)
    assert instances == original


def test_wrong_official_repository_and_partial_image_maps_are_rejected(prepared_domains: dict[str, Path]) -> None:
    instances = load_instances(prepared_domains["swebench-verified"], query_ids=[], limit=0, seed=42)
    original = copy.deepcopy(instances)
    first, second = instances
    first_pin = first["image"].removesuffix(":latest") + "@sha256:" + "b" * 64
    with pytest.raises(ValueError):
        pin_instances(instances, {first["query_id"]: first_pin})
    with pytest.raises(ValueError, match="canonical|repository|match"):
        pin_instances(instances, {first["query_id"]: second["image"].removesuffix(":latest") + "@sha256:" + "b" * 64})
    assert instances == original


def test_controller_and_local_matrix_consume_the_same_pinned_image_profile(
    prepared_domains: dict[str, Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data = prepared_domains["swebench-verified"]
    before = snapshot(data)
    instances = load_instances(data, query_ids=[], limit=0, seed=42)
    images = {row["query_id"]: row["image"].removesuffix(":latest") + "@sha256:" + "c" * 64 for row in instances}
    profile = tmp_path / "synthetic-image-pins.yaml"
    profile.write_text(yaml.safe_dump({"task_images": images}))
    specs = [*PROFILES["swebench-verified"], str(profile)]
    config, planned = submission_plan(data, specs, 0, 42, 2, [])
    assert config["task_images"] == images
    arguments = {
        "data_dir": data,
        "split": "test",
        "query_ids": [],
        "limit": 0,
        "seed": 42,
        "config_spec": specs,
        "output": tmp_path / "must-not-exist",
        "dry_run": True,
        "workers_per_method": 2,
    }
    run_matrix(**arguments)
    local = yaml.safe_load(capsys.readouterr().out)
    assert local["selected_query_ids"] == planned["selected_query_ids"]
    for method in METHODS:
        assert (
            run_manifest(config, method, pin_instances(instances, images), 2, data / "queries.jsonl", "test")[
                "task_images"
            ]
            == images
        )
    profile.write_text(yaml.safe_dump({"task_images": {instances[0]["query_id"]: images[instances[1]["query_id"]]}}))
    with pytest.raises(ValueError):
        submission_plan(data, specs, 0, 42, 2, [])
    with pytest.raises(ValueError):
        run_matrix(**arguments)
    assert snapshot(data) == before and not arguments["output"].exists()


@pytest.mark.parametrize(
    ("benchmark", "preflight"),
    [
        ("swebench-verified", False),
        ("swebench-verified", True),
        ("browsecomp-plus", False),
        ("browsecomp-plus", True),
    ],
)
def test_submit_dry_run_uses_dynamic_data_dir_with_no_network_api_models_or_writes(
    prepared_domains: dict[str, Path],
    tmp_path: Path,
    benchmark: str,
    preflight: bool,
) -> None:
    before = snapshot(tmp_path)
    arguments = [
        "submit",
        "--benchmark",
        benchmark,
        "--limit",
        "0",
        "--workers-per-method",
        "30",
        "--manifest",
        str(tmp_path / "controller-must-not-exist.json"),
        "--dry-run",
        *[argument for profile in PROFILES[benchmark] for argument in ("-c", profile)],
        *(["--preflight"] if preflight else []),
    ]
    result = offline_cli(tmp_path, arguments)
    assert result.returncode == 0, result.stdout + result.stderr
    plan = yaml.safe_load(result.stdout)
    assert plan["benchmark"] == benchmark and len(plan["selected_query_ids"]) == 2
    assert plan["planned_attempts"] == 8 and plan["attempts"] == (0 if preflight else 8)
    assert plan["preflight_only"] is preflight and plan["workers_per_method"] == 30
    assert (
        plan["selected_query_ids"]
        == submission_plan(
            prepared_domains[benchmark],
            PROFILES[benchmark],
            0,
            42,
            30,
            [],
            benchmark=benchmark,
        )[1]["selected_query_ids"]
    )
    assert PRIVATE not in result.stdout + result.stderr
    assert snapshot(tmp_path) == before
    assert not (tmp_path / "controller-must-not-exist.json").exists()


def test_submit_help_exposes_benchmark_selection_without_free_cloud_claims(tmp_path: Path) -> None:
    result = offline_cli(tmp_path, ["submit", "--help"])
    assert result.returncode == 0, result.stdout + result.stderr
    assert all(
        option in result.stdout for option in ("--benchmark", "--split", "--corpus-snapshot", "--data-dir", "--dry-run")
    )
    assert "browsecomp-plus" in result.stdout and "swebench-verified" in result.stdout
    assert not any(
        claim in result.stdout.lower() for claim in ("free compute", "free cloud", "zero cost", "zero-cost", "$0")
    )
    assert not list(tmp_path.iterdir())
