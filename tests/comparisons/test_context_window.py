import copy

import pytest
from jinja2 import StrictUndefined, Template

from pocketcontextagent.agents import get_agent
from pocketcontextagent.agents.default import AgentConfig
from pocketcontextagent.agents.utils.context_budget import estimate_tokens
from pocketcontextagent.comparisons.methods.base import ComparisonConfig
from pocketcontextagent.comparisons.methods.resum import ReSumConfig
from pocketcontextagent.comparisons.registry import METHODS, configure_method, method_provenance
from pocketcontextagent.environments.local import LocalEnvironment
from pocketcontextagent.models import get_model
from pocketcontextagent.run.benchmarks.benchmark_matrix import benchmark_config
from pocketcontextagent.run.benchmarks.browsecomp_plus_daytona import matrix_config


@pytest.mark.parametrize(("builder",), [(benchmark_config,), (matrix_config,)])  # noqa: PT006
@pytest.mark.parametrize(("window",), [(32000,), (64000,), (128000,)])  # noqa: PT006
def test_context_window_override_reaches_all_method_prompts_and_runtime(builder, window):
    default = builder([])
    shared = builder([], context_window=window)
    original = copy.deepcopy(shared)
    assert shared["model"] == default["model"]
    assert default["agent"]["context_token_limit"] == 128000
    for method in METHODS:
        config = configure_method(shared, method)
        config_class = {"pocket_context": AgentConfig, "resum": ReSumConfig}.get(method, ComparisonConfig)
        runtime = config_class(**config["agent"])
        variables = config["environment"] | config["model"] | runtime.model_dump()
        assert f"{window} tokens" in Template(runtime.system_template, undefined=StrictUndefined).render(**variables)
        assert (
            runtime.context_token_limit
            == method_provenance(config, method)["settings"]["context_token_limit"]
            == window
        )
        assert config["model"]["context_tools"] == METHODS[method].context_tools
        if method == "resum":
            assert runtime.resum_trigger_ratio == 0.9
            assert runtime.resum_headroom_tokens == 1000
            assert runtime.resum_summary_token_limit == 4096
    assert shared == original
    assert "agent_class" not in shared["agent"]


@pytest.mark.parametrize(
    ("builder", "profile"),
    [
        (benchmark_config, "context_benchmark_deepseek"),
        (matrix_config, "browsecomp_plus_deepseek"),
    ],
)
def test_context_window_flag_overrides_config_without_changing_provider(builder, profile):
    specs = [profile, "agent.context_token_limit=96000"]
    assert builder(specs)["agent"]["context_token_limit"] == 96000
    assert builder(specs, context_window=64000)["agent"]["context_token_limit"] == 64000
    assert specs == [profile, "agent.context_token_limit=96000"]
    for invalid in (0, -1, True, "64k"):
        with pytest.raises(ValueError, match="positive integer"):
            builder(specs, context_window=invalid)


@pytest.mark.parametrize(("method",), [("react",), ("resum",), ("acm",)])  # noqa: PT006
def test_pocket_pressure_does_not_change_baseline_accounting(tmp_path, method):
    config = configure_method(benchmark_config([], context_window=64000), method)
    model = get_model(config=config["model"])
    agent = get_agent(model, LocalEnvironment(), config["agent"] | {"context_db_path": tmp_path / "context.sqlite"})
    assert "compact_pressure_fraction" not in config["agent"]
    for characters in (100, 120000):
        messages = [model.format_message(role="user", content="x" * characters)]
        assert agent.context_tokens(messages) == estimate_tokens(
            agent.context._prepare(model, messages), tools=model.get_tools()
        )
        assert "one substantial range" not in agent._render_template(agent.config.system_template)
