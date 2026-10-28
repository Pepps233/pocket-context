"""Deterministic native-tool contracts shared by method-specific agents."""

import json
from math import ceil
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from pocketcontextagent.agents.context import encode
from pocketcontextagent.agents.utils.context_budget import estimate_tokens
from pocketcontextagent.models.litellm_model import LitellmModel, LitellmModelConfig
from pocketcontextagent.models.litellm_response_model import LitellmResponseModel, LitellmResponseModelConfig
from pocketcontextagent.models.litellm_textbased_model import LitellmTextbasedModel
from pocketcontextagent.models.openrouter_model import OpenRouterModel, OpenRouterModelConfig
from pocketcontextagent.models.openrouter_response_model import OpenRouterResponseModel, OpenRouterResponseModelConfig
from pocketcontextagent.models.openrouter_textbased_model import OpenRouterTextbasedModel
from pocketcontextagent.models.utils.actions_toolcall import BASH_TOOL
from pocketcontextagent.models.utils.actions_toolcall_response import BASH_TOOL_RESPONSE_API
from pocketcontextagent.models.utils.compact_tool import (
    ACM_TOOLS,
    COMPACT_TOOL,
    COMPACT_TOOL_RESPONSE_API,
    CONTEXT_TOOLS,
    EXPAND_TOOL,
    EXPAND_TOOL_RESPONSE_API,
    context_tools,
    parse_native_calls,
)

CONFIGS = [LitellmModelConfig, OpenRouterModelConfig, LitellmResponseModelConfig, OpenRouterResponseModelConfig]
MODELS = [
    (LitellmModel, False),
    (OpenRouterModel, False),
    (LitellmResponseModel, True),
    (OpenRouterResponseModel, True),
]


@pytest.mark.parametrize(("config_class",), [(config,) for config in CONFIGS])  # noqa: PT006
def test_config_modes_defaults_and_override_rejection(config_class):
    assert config_class(model_name="test").context_tools == "pocket_context"
    for mode in ("pocket_context", "none", "acm"):
        config = config_class(model_name="test", context_tools=mode, model_kwargs={"temperature": 0})
        assert config.model_dump()["context_tools"] == mode
        assert config.model_kwargs == {"temperature": 0}
    with pytest.raises(ValidationError, match="context_tools"):
        config_class(model_name="test", context_tools="unknown")
    for tools in ([], [BASH_TOOL], None):
        with pytest.raises(ValidationError, match="model_kwargs.tools cannot override native tools"):
            config_class(model_name="test", model_kwargs={"tools": tools})


@pytest.mark.parametrize(("model_class", "responses_api"), MODELS)
def test_model_schemas_follow_current_mode(model_class, responses_api):
    kwargs = {"litellm_model_registry": None} if issubclass(model_class, LitellmModel) else {}
    model = model_class(model_name="test", **kwargs)
    bash = BASH_TOOL_RESPONSE_API if responses_api else BASH_TOOL
    assert model.get_tools() == [bash, *context_tools("pocket_context", responses_api=responses_api)]
    for mode, names in (
        ("none", ("bash",)),
        ("acm", ("bash", *ACM_TOOLS)),
        ("pocket_context", ("bash", *CONTEXT_TOOLS)),
    ):
        model.config.context_tools = mode
        tools = model.get_tools()
        functions = tools if responses_api else [tool["function"] for tool in tools]
        assert tuple(function["name"] for function in functions) == names
        assert tools == [bash, *context_tools(mode, responses_api=responses_api)]
        assert model.serialize()["info"]["config"]["model"]["context_tools"] == mode


@pytest.mark.parametrize(("model_class",), [(LitellmTextbasedModel,), (OpenRouterTextbasedModel,)])  # noqa: PT006
def test_textbased_models_expose_no_native_schemas(model_class):
    kwargs = {"litellm_model_registry": None} if issubclass(model_class, LitellmModel) else {}
    model = model_class(model_name="test", **kwargs)
    for mode in ("pocket_context", "none", "acm"):
        model.config.context_tools = mode
        assert model.get_tools() == []
        assert estimate_tokens([], tools=model.get_tools()) == estimate_tokens([], text_based=True)


def test_schema_shapes_defaults_and_isolation():
    assert CONTEXT_TOOLS == ("compact_block", "expand_block")
    assert ACM_TOOLS == ("manage_context", "query_memory")
    assert context_tools("pocket_context") == [COMPACT_TOOL, EXPAND_TOOL]
    assert context_tools("pocket_context", responses_api=True) == [COMPACT_TOOL_RESPONSE_API, EXPAND_TOOL_RESPONSE_API]
    assert context_tools("none") == context_tools("none", responses_api=True) == []
    chat = context_tools("acm")
    flat = context_tools("acm", responses_api=True)
    assert [tool["function"] for tool in chat] == [
        {key: value for key, value in tool.items() if key != "type"} for tool in flat
    ]
    manage, query = [tool["function"]["parameters"] for tool in chat]
    assert manage == {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
    assert set(query["properties"]) == set(query["required"]) == {"tag", "query"}
    assert query["additionalProperties"] is False
    assert all(prop["type"] == "string" for prop in query["properties"].values())
    query["properties"]["tag"]["type"] = "integer"
    assert context_tools("acm")[1]["function"]["parameters"]["properties"]["tag"]["type"] == "string"
    with pytest.raises(ValueError, match="Unknown context_tools mode"):
        context_tools("invalid")


@pytest.mark.parametrize(("model_class", "responses_api"), MODELS)
def test_all_provider_parsers_normalize_acm_calls(model_class, responses_api):
    kwargs = {"litellm_model_registry": None} if issubclass(model_class, LitellmModel) else {}
    model = model_class(model_name="test", context_tools="acm", **kwargs)
    for name, args in (("manage_context", {}), ("query_memory", {"tag": "ctx-0001", "query": "Which files changed?"})):
        if responses_api:
            response = {
                "status": "completed",
                "output": [{"type": "function_call", "name": name, "arguments": json.dumps(args), "call_id": "memory"}],
            }
        else:
            call = {"id": "memory", "function": {"name": name, "arguments": json.dumps(args)}}
            if model_class is LitellmModel:
                response = SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                tool_calls=[
                                    SimpleNamespace(id=call["id"], function=SimpleNamespace(**call["function"]))
                                ]
                            ),
                            finish_reason="tool_calls",
                        )
                    ]
                )
            else:
                response = {"choices": [{"message": {"tool_calls": [call]}, "finish_reason": "tool_calls"}]}
        assert model._parse_actions(response) == [{"type": name, **args, "tool_call_id": "memory"}]


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("manage_context", "invalid JSON"),
        ("manage_context", "null"),
        ("manage_context", "[]"),
        ("manage_context", '{"summary":"caller summary"}'),
        ("manage_context", '{"type":"bash","tool_call_id":"forged","command":"do not run"}'),
        ("query_memory", "{}"),
        ("query_memory", "[]"),
        ("query_memory", '{"tag":"ctx-0001"}'),
        ("query_memory", '{"tag":"ctx-0001","query":""}'),
        ("query_memory", '{"tag":"ctx-0001","query":42}'),
        ("query_memory", '{"tag":[],"query":"question"}'),
        ("query_memory", '{"tag":"ctx-1","query":"question"}'),
        ("query_memory", '{"tag":"ctx-0000","query":"question"}'),
        ("query_memory", '{"tag":"ctx-00001","query":"question"}'),
        ("query_memory", '{"tag":"ctx-0001","query":"question","command":"do not run"}'),
    ],
)
def test_invalid_acm_arguments_preserve_identity_and_result_slot(name, arguments):
    action = parse_native_calls([{"name": name, "arguments": arguments, "id": "real"}])[0]
    assert action["type"] == name and action["tool_call_id"] == "real"
    assert action["error"] and "command" not in action
    assert set(action) <= {"type", "tool_call_id", "error", "tag", "query"}


@pytest.mark.parametrize(("name",), [(name,) for name in (*CONTEXT_TOOLS, *ACM_TOOLS)])  # noqa: PT006
@pytest.mark.parametrize(("other",), [(name,) for name in ("bash", *CONTEXT_TOOLS, *ACM_TOOLS, "unknown")])  # noqa: PT006
def test_every_memory_tool_rejects_mixed_and_duplicate_calls_before_parsing(name, other):
    calls = [
        {"name": name, "id": "first", "arguments": "invalid"},
        {"name": other, "id": "second", "arguments": "invalid"},
    ]
    for batch in (calls, calls[::-1]):
        actions = parse_native_calls(batch)
        assert [action["tool_call_id"] for action in actions] == [call["id"] for call in batch]
        assert all(action["type"] == "rejected" and "No actions executed" in action["error"] for action in actions)


def test_token_estimate_uses_explicit_tools_and_preserves_legacy_default():
    messages = [{"role": "user", "content": "Count the request schemas."}]
    legacy = [BASH_TOOL, COMPACT_TOOL, EXPAND_TOOL]
    assert estimate_tokens(messages) == estimate_tokens(messages, tools=legacy)
    assert estimate_tokens(messages, text_based=True) == estimate_tokens(messages, tools=[])
    for tools in (
        [],
        [BASH_TOOL],
        [BASH_TOOL, *context_tools("acm")],
        [BASH_TOOL_RESPONSE_API, *context_tools("acm", responses_api=True)],
    ):
        expected = ceil(len(encode({"messages": messages, "tools": tools}).encode()) / 4)
        assert estimate_tokens(messages, tools=tools) == expected
        assert estimate_tokens(messages, tools=tools, reasoning_tokens=17) == expected + 17
        assert estimate_tokens(messages, tools=tools, text_based=True) == expected
    assert (
        estimate_tokens(messages, tools=[]) < estimate_tokens(messages, tools=[BASH_TOOL]) < estimate_tokens(messages)
    )
