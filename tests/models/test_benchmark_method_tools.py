import copy

import pytest
from jinja2 import Template

from pocketcontextagent.comparisons.registry import METHODS, configure_method, method_provenance
from pocketcontextagent.models.litellm_model import LitellmModel
from pocketcontextagent.models.litellm_response_model import LitellmResponseModel
from pocketcontextagent.models.openrouter_model import OpenRouterModel
from pocketcontextagent.models.openrouter_response_model import OpenRouterResponseModel
from pocketcontextagent.models.utils.compact_tool import parse_native_calls


@pytest.mark.parametrize(("method",), [(name,) for name in METHODS])  # noqa: PT006
@pytest.mark.parametrize(
    ("model_class",),  # noqa: PT006
    [(LitellmModel,), (LitellmResponseModel,), (OpenRouterModel,), (OpenRouterResponseModel,)],
)
def test_coding_benchmark_has_shell_tools_and_method_memory_tools(method, model_class):
    original = {
        "benchmark": "swebench-verified",
        "model": {"model_name": "deepseek/deepseek-flash"},
        "agent": {"context_token_limit": 128000},
    }
    before = copy.deepcopy(original)
    config = configure_method(original, method)
    model = model_class(**config["model"])
    names = {tool.get("function", tool)["name"] for tool in model.get_tools()}
    memory = {"pocket_context": {"compact_block", "expand_block"}, "acm": {"manage_context", "query_memory"}}.get(
        method, set()
    )
    assert names == {"bash"} | memory
    assert original == before and configure_method(config, method) == config
    prompt = Template(config["agent"]["system_template"]).render()
    assert "live web" not in prompt and "/testbed" in prompt
    assert ("compact_block" in prompt) == (method == "pocket_context")
    assert ("manage_context" in prompt) == (method == "acm")
    assert "domain-adapted" in method_provenance(config, method)["evaluation"]
    assert "131072" not in prompt and "128000" in prompt


def test_native_shell_actions_can_share_a_batch_with_acm_memory():
    assert parse_native_calls(
        [
            {"id": "shell", "name": "bash", "arguments": '{"command":"git status"}'},
            {"id": "memory", "name": "manage_context", "arguments": "{}"},
        ]
    ) == [{"command": "git status", "tool_call_id": "shell"}, {"type": "manage_context", "tool_call_id": "memory"}]
