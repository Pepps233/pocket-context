import copy

import litellm
import pytest
from litellm.llms.deepseek.chat.transformation import DeepSeekChatConfig
from litellm.utils import get_optional_params

from pocketcontextagent.comparisons.registry import configure_method
from pocketcontextagent.config import get_config_from_spec
from pocketcontextagent.models.litellm_model import LitellmModel
from pocketcontextagent.models.utils.compact_tool import native_request_kwargs
from pocketcontextagent.utils.serialize import recursive_merge


def profile(method: str) -> dict:
    return configure_method(
        recursive_merge(get_config_from_spec("browsecomp_plus"), get_config_from_spec("browsecomp_plus_deepseek")),
        method,
    )


def request_body(model: LitellmModel, messages: list[dict], **overrides) -> dict:
    options = native_request_kwargs(
        tools=model.get_tools(), model_kwargs=model.config.model_kwargs, overrides=overrides
    )
    options.pop("api_base")
    provider = DeepSeekChatConfig()
    optional_params = get_optional_params(model="deepseek-flash", custom_llm_provider="deepseek", **options)
    extra_body = optional_params.pop("extra_body", {})
    body = provider.transform_request(
        model="deepseek-flash",
        messages=model._prepare_messages_for_api(messages),
        optional_params=optional_params,
        litellm_params={},
        headers={},
    )
    return body | provider.transform_extra_body(
        extra_body=extra_body, request=body, model="deepseek-flash", litellm_params={}
    )


@pytest.mark.parametrize(
    ("method", "memory_tools"),
    [
        ("pocket_context", {"compact_block", "expand_block"}),
        ("react", set()),
        ("resum", set()),
        ("acm", {"manage_context", "query_memory"}),
    ],
)
def test_direct_deepseek_profile_preserves_each_methods_tools_and_reasoning(method: str, memory_tools: set[str]):
    config = profile(method)
    model = LitellmModel(**config["model"])
    assert model.config.model_name == "deepseek/deepseek-flash"
    assert model.config.cost_tracking == "default"
    assert model.config.model_kwargs["api_base"] == "https://api.deepseek.com"
    assert "api_key" not in model.config.model_kwargs and "provider" not in model.config.model_kwargs
    assert config["agent"]["context_token_limit"] == 128000
    assert config["environment"]["k"] == 10 and config["environment"]["document_tokens"] == 8192
    messages = [
        {"role": "system", "content": "Research the corpus."},
        {"role": "user", "content": "Which city?"},
        {
            "role": "assistant",
            "content": None,
            "reasoning_content": "Search the festival evidence.",
            "tool_calls": [
                {"id": "search", "type": "function", "function": {"name": "search", "arguments": '{"query":"jazz"}'}}
            ],
            "extra": {"private_metadata": "not sent"},
        },
        {"role": "tool", "tool_call_id": "search", "content": "Montréal [city]"},
    ]
    before = copy.deepcopy(messages)
    body = request_body(model, messages)
    assert body["model"] == "deepseek-flash"
    assert body["thinking"] == {"type": "enabled"}
    assert body["reasoning_effort"] == "high"
    assert body["max_tokens"] == 16384
    assert {tool["function"]["name"] for tool in body["tools"]} == {"search", "get_document"} | memory_tools
    assert body["messages"][2]["reasoning_content"] == "Search the festival evidence."
    assert "extra" not in body["messages"][2] and messages == before
    assert (
        DeepSeekChatConfig().get_complete_url(
            api_base=model.config.model_kwargs["api_base"],
            api_key=None,
            model="deepseek-flash",
            optional_params={},
            litellm_params={},
        )
        == "https://api.deepseek.com/chat/completions"
    )


def test_deepseek_auxiliary_calls_are_tool_free_without_mutating_main_configuration():
    model = LitellmModel(**profile("acm")["model"])
    before = copy.deepcopy(model.config.model_kwargs)
    body = request_body(
        model, [{"role": "user", "content": "Summarize evidence."}], tools=[], tool_choice="none", max_tokens=4096
    )
    assert not body.get("tools") and body.get("tool_choice") in (None, "none")
    assert body["max_tokens"] == 4096 and body["thinking"] == {"type": "enabled"}
    assert model.config.model_kwargs == before
    assert {tool["function"]["name"] for tool in model.get_tools()} == {
        "search",
        "get_document",
        "manage_context",
        "query_memory",
    }


def test_deepseek_profile_has_registered_pricing_for_episode_cost_accounting():
    model = LitellmModel(**profile("react")["model"])
    response = litellm.ModelResponse(
        model="deepseek-flash",
        choices=[{"message": {"role": "assistant", "content": "Exact Answer: Montréal"}}],
        usage={"prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1100},
    )
    cost = model._calculate_cost(response)
    assert cost["cost_available"] is True and cost["cost"] > 0
