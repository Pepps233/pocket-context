import copy

import litellm
import pytest
from litellm.types.llms.openai import ResponsesAPIResponse
from litellm.types.router import GenericLiteLLMParams
from litellm.utils import ProviderConfigManager

from pocketcontextagent.comparisons.registry import configure_method
from pocketcontextagent.config import get_config_from_spec
from pocketcontextagent.models.litellm_response_model import LitellmResponseModel
from pocketcontextagent.models.utils.compact_tool import native_request_kwargs
from pocketcontextagent.utils.serialize import recursive_merge

MODEL = "meta/muse-spark-1.3-contributor"
API_MODEL = "muse-spark-1.3-contributor"


def profile(method: str) -> dict:
    return configure_method(
        recursive_merge(
            get_config_from_spec("browsecomp_plus"),
            get_config_from_spec("browsecomp_plus_muse_spark_1_3_contributor"),
        ),
        method,
    )


def provider():
    return ProviderConfigManager.get_provider_responses_api_config(
        provider=litellm.LlmProviders.META, model=API_MODEL, api_base="https://api.meta.ai/v1"
    )


def request_body(model: LitellmResponseModel, history: list[dict], **overrides) -> dict:
    options = native_request_kwargs(
        tools=model.get_tools(), model_kwargs=model.config.model_kwargs, overrides=overrides, responses_api=True
    )
    options.pop("api_base")
    return provider().transform_responses_api_request(
        model=API_MODEL,
        input=model._prepare_messages_for_api(history),
        response_api_optional_request_params=options,
        litellm_params=GenericLiteLLMParams(custom_llm_provider="meta"),
        headers={},
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
def test_muse_profile_selects_exact_13_contributor_and_preserves_native_reasoning(method: str, memory_tools: set[str]):
    config = profile(method)
    assert config["model"]["model_class"] == "litellm_response"
    assert config["model"]["model_name"] == MODEL
    model = LitellmResponseModel(**config["model"])
    assert model.config.cost_tracking == "default"
    assert model.config.model_kwargs["api_base"] == "https://api.meta.ai/v1"
    assert all(key not in model.config.model_kwargs for key in ("api_key", "provider", "thinking", "fallbacks"))
    assert config["agent"]["context_token_limit"] == 128000
    assert config["environment"]["k"] == 10 and config["environment"]["document_tokens"] == 8192
    encrypted = {
        "id": "reasoning-1",
        "type": "reasoning",
        "encrypted_content": "opaque-replay-payload",
        "summary": [{"type": "summary_text", "text": "Look for the festival evidence."}],
    }
    call = {"type": "function_call", "call_id": "search-1", "name": "search", "arguments": '{"query":"jazz"}'}
    history = [
        model.format_message(role="system", content="Research the corpus."),
        model.format_message(role="user", content="Which city?"),
        {"object": "response", "output": [encrypted, call], "extra": {"private_metadata": "not sent"}},
        {"type": "function_call_output", "call_id": "search-1", "output": "Montréal [city]"},
    ]
    before = copy.deepcopy(history)
    body = request_body(model, history)
    assert body["model"] == API_MODEL
    assert body["reasoning"] == {"effort": "high", "summary": "auto"}
    assert body["include"] == ["reasoning.encrypted_content"] and body["store"] is False
    assert body["max_output_tokens"] == 16384
    assert {tool["name"] for tool in body["tools"]} == {"search", "get_document"} | memory_tools
    assert encrypted in body["input"] and call in body["input"]
    assert history == before and all("extra" not in item for item in body["input"])
    assert (
        provider().get_complete_url(api_base=model.config.model_kwargs["api_base"], litellm_params={})
        == "https://api.meta.ai/v1/responses"
    )


def test_muse_contributor_auxiliary_calls_keep_exact_model_and_use_no_tools():
    model = LitellmResponseModel(**profile("acm")["model"])
    before = copy.deepcopy(model.config.model_kwargs)
    body = request_body(
        model,
        [model.format_message(role="user", content="Summarize evidence.")],
        tools=[],
        tool_choice="none",
        max_output_tokens=4096,
    )
    assert body["model"] == API_MODEL and not body.get("tools")
    assert body["max_output_tokens"] == 4096
    assert body["reasoning"]["effort"] == "high" and body["store"] is False
    assert model.config.model_kwargs == before
    assert {tool["name"] for tool in model.get_tools()} == {"search", "get_document", "manage_context", "query_memory"}


def test_muse_contributor_native_responses_have_registered_cost_accounting():
    model = LitellmResponseModel(**profile("react")["model"])
    response = ResponsesAPIResponse(
        id="resp-offline",
        object="response",
        created_at=0,
        status="completed",
        model=API_MODEL,
        output=[],
        parallel_tool_calls=False,
        usage={"input_tokens": 1000, "output_tokens": 100, "total_tokens": 1100},
    )
    cost = model._calculate_cost(response)
    assert cost["cost_available"] is True and cost["cost"] > 0
