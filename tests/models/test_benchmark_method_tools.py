import copy
import json

import pytest
from jinja2 import Template

from pocketcontextagent.comparisons.registry import METHODS, configure_method, method_provenance
from pocketcontextagent.models.litellm_model import LitellmModel
from pocketcontextagent.models.litellm_response_model import LitellmResponseModel
from pocketcontextagent.models.openrouter_model import OpenRouterModel
from pocketcontextagent.models.openrouter_response_model import OpenRouterResponseModel
from pocketcontextagent.models.utils.compact_tool import parse_native_calls


@pytest.mark.parametrize(
    ("benchmark", "research"), [("deepsearchqa", {"search", "browse"}), ("swebench-verified", {"bash"})]
)
@pytest.mark.parametrize(("method",), [(name,) for name in METHODS])  # noqa: PT006
@pytest.mark.parametrize(
    ("model_class",),  # noqa: PT006
    [(LitellmModel,), (LitellmResponseModel,), (OpenRouterModel,), (OpenRouterResponseModel,)],
)
def test_each_benchmark_has_domain_tools_and_method_memory_tools(benchmark, research, method, model_class):
    original = {
        "benchmark": benchmark,
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
    assert names == research | memory
    assert original == before and configure_method(config, method) == config
    prompt = Template(config["agent"]["system_template"]).render()
    assert ("live web" in prompt) == (benchmark == "deepsearchqa")
    assert ("/testbed" in prompt) == (benchmark == "swebench-verified")
    assert ("compact_block" in prompt) == (method == "pocket_context")
    assert ("manage_context" in prompt) == (method == "acm")
    assert "domain-adapted" in method_provenance(config, method)["evaluation"]
    assert "131072" not in prompt and "128000" in prompt


def test_native_browse_preserves_urls_and_rejects_invalid_arguments():
    url = "https://example.com/article?a=1&b=two"
    assert parse_native_calls([{"id": "read", "name": "browse", "arguments": json.dumps({"url": url})}]) == [
        {"type": "browse", "tool_call_id": "read", "command": f"browse {url}"}
    ]
    for arguments in ({}, {"url": ""}, {"url": " "}, {"url": url, "secret": "not allowed"}):
        assert parse_native_calls([{"id": "bad", "name": "browse", "arguments": json.dumps(arguments)}])[0]["error"]
    assert parse_native_calls(
        [
            {"id": "shell", "name": "bash", "arguments": '{"command":"git status"}'},
            {"id": "memory", "name": "manage_context", "arguments": "{}"},
        ]
    ) == [{"command": "git status", "tool_call_id": "shell"}, {"type": "manage_context", "tool_call_id": "memory"}]
