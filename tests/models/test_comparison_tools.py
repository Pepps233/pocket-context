"""Offline contracts for the released comparison schemas and native corpus tools."""

import hashlib
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from pocketcontextagent.comparisons.tools import (
    BCP_GET_DOCUMENT_TOOL,
    BCP_SEARCH_TOOL,
    MANAGE_CONTEXT_TOOL,
    QUERY_MEMORY_TOOL,
    acm_tools,
    corpus_tools,
)
from pocketcontextagent.models.litellm_model import LitellmModel, LitellmModelConfig
from pocketcontextagent.models.litellm_response_model import LitellmResponseModel, LitellmResponseModelConfig
from pocketcontextagent.models.openrouter_model import OpenRouterModel, OpenRouterModelConfig
from pocketcontextagent.models.openrouter_response_model import OpenRouterResponseModel, OpenRouterResponseModelConfig
from pocketcontextagent.models.utils.compact_tool import context_tools, native_request_kwargs, parse_native_calls

MODELS = [
    (LitellmModel, False),
    (OpenRouterModel, False),
    (LitellmResponseModel, True),
    (OpenRouterResponseModel, True),
]
CONFIGS = [LitellmModelConfig, OpenRouterModelConfig, LitellmResponseModelConfig, OpenRouterResponseModelConfig]


@pytest.mark.parametrize(
    ("tool", "digest"),
    [
        (MANAGE_CONTEXT_TOOL, "f59f5e6c1463c117e25f859179cd1c1389b4e20cfc94134ff0d0d4979a4272b3"),
        (QUERY_MEMORY_TOOL, "025c3dda8ca18c2fed3094fa742e25debb920b7f7ea635bed1eb5ae1718e52d1"),
        (BCP_SEARCH_TOOL, "e7af0c63554e25eb49c964e8aa0c95e12938cf3ed82b231cf1e7c56558655754"),
        (BCP_GET_DOCUMENT_TOOL, "9efa4579de368d1783d526ad3c35fe4b44fc3ce90426c674e7b4fcff46429fcf"),
    ],
)
def test_schemas_match_pinned_upstream(tool, digest):
    # Canonical JSON hashes of literal definitions at ACM f06f90e728af8580a4515812425c1620144145a2.
    assert hashlib.sha256(json.dumps(tool, sort_keys=True, ensure_ascii=False).encode()).hexdigest() == digest


@pytest.mark.parametrize(("mode",), [("acm",), ("pocket_context",), ("none",)])  # noqa: PT006
@pytest.mark.parametrize(("responses_api",), [(False,), (True,)])  # noqa: PT006
def test_corpus_schema_shapes_descriptions_and_isolation(mode, responses_api):
    tools = corpus_tools(mode, responses_api=responses_api)
    functions = tools if responses_api else [tool["function"] for tool in tools]
    assert [function["name"] for function in functions] == ["search", "get_document"]
    assert functions[0]["parameters"] == BCP_SEARCH_TOOL["function"]["parameters"]
    assert functions[1]["parameters"] == BCP_GET_DOCUMENT_TOOL["function"]["parameters"]
    assert ("query_memory" in functions[0]["description"]) == (mode == "acm")
    assert ("manage_context" in functions[0]["description"]) == (mode == "acm")
    functions[0]["parameters"]["properties"]["query"]["type"] = "invalid"
    assert corpus_tools(mode)[0]["function"]["parameters"]["properties"]["query"]["type"] == "string"
    assert BCP_SEARCH_TOOL["function"]["parameters"]["properties"]["query"]["type"] == "string"
    if mode == "acm":
        assert corpus_tools(mode) == [BCP_SEARCH_TOOL, BCP_GET_DOCUMENT_TOOL]
        assert context_tools(mode, responses_api=responses_api) == acm_tools(responses_api=responses_api)


def test_invalid_corpus_schema_mode():
    with pytest.raises(ValueError, match="Unknown context_tools mode"):
        corpus_tools("unknown")


@pytest.mark.parametrize(("config_class",), [(config,) for config in CONFIGS])  # noqa: PT006
def test_research_tool_configuration_defaults_and_validation(config_class):
    assert config_class(model_name="test").research_tools == "bash"
    assert config_class(model_name="test", research_tools="corpus").model_dump()["research_tools"] == "corpus"
    with pytest.raises(ValidationError, match="research_tools"):
        config_class(model_name="test", research_tools="unknown")
    with pytest.raises(ValidationError, match="model_kwargs.tools cannot override native tools"):
        config_class(model_name="test", research_tools="corpus", model_kwargs={"tools": []})


@pytest.mark.parametrize(("model_class", "responses_api"), MODELS)
@pytest.mark.parametrize(("mode",), [("acm",), ("pocket_context",), ("none",)])  # noqa: PT006
def test_provider_corpus_schemas_and_auxiliary_request_options(model_class, responses_api, mode):
    kwargs = {"litellm_model_registry": None} if issubclass(model_class, LitellmModel) else {}
    model = model_class(model_name="test", research_tools="corpus", context_tools=mode, **kwargs)
    schemas = model.get_tools()
    assert schemas == [
        *corpus_tools(mode, responses_api=responses_api),
        *context_tools(mode, responses_api=responses_api),
    ]
    assert model.serialize()["info"]["config"]["model"]["research_tools"] == "corpus"
    overrides = {"tools": [], "max_tokens": 4096, "tool_choice": "none"}
    options = native_request_kwargs(
        tools=schemas, model_kwargs=model.config.model_kwargs, overrides=overrides, responses_api=responses_api
    )
    assert options == {
        "tools": [],
        "max_output_tokens" if responses_api else "max_tokens": 4096,
        "tool_choice": "none",
    }
    options["tools"].append({"name": "not persistent"})
    assert overrides == {"tools": [], "max_tokens": 4096, "tool_choice": "none"}
    assert model.get_tools() == schemas
    assert native_request_kwargs(tools=schemas, model_kwargs={}, overrides={})["tools"] == schemas


@pytest.mark.parametrize(("responses_api",), [(False,), (True,)])  # noqa: PT006
def test_native_request_merge_precedence_and_no_mutation(responses_api):
    schemas = acm_tools(responses_api=responses_api)
    client = object()
    model_kwargs = {"temperature": 0.8, "metadata": {"source": "config"}}
    overrides = {"temperature": 0, "metadata": {"source": "request"}, "max_tokens": 4096, "max_output_tokens": 512}
    original = deepcopy((schemas, model_kwargs, overrides))
    options = native_request_kwargs(
        tools=schemas, model_kwargs=model_kwargs, overrides=overrides, responses_api=responses_api
    )
    assert options["temperature"] == 0 and options["metadata"] == {"source": "request"}
    assert options["max_output_tokens"] == 512
    assert ("max_tokens" in options) is not responses_api
    (options["tools"][0] if responses_api else options["tools"][0]["function"])["name"] = "changed"
    assert (schemas, model_kwargs, overrides) == original
    assert native_request_kwargs(tools=[], model_kwargs={"client": client}, overrides={})["client"] is client
    explicit = native_request_kwargs(tools=[], model_kwargs={}, overrides={"tools": schemas})
    (explicit["tools"][0] if responses_api else explicit["tools"][0]["function"])["name"] = "changed"
    assert schemas == original[0]
    for tools in ([], None, schemas):
        with pytest.raises(ValueError, match="model_kwargs.tools cannot override native tools"):
            native_request_kwargs(tools=schemas, model_kwargs={"tools": tools}, overrides={"tools": []})


def _calls():
    return [
        {"name": "search", "arguments": json.dumps({"query": "alpha beta"}), "id": "search"},
        {"name": "manage_context", "arguments": "{}", "id": "manage"},
        {"name": "query_memory", "arguments": json.dumps({"summary_id": 1, "query": "Evidence?"}), "id": "memory"},
        {"name": "get_document", "arguments": json.dumps({"docid": "doc:1"}), "id": "document"},
    ]


@pytest.mark.parametrize(("model_class", "responses_api"), MODELS)
def test_all_provider_parsers_keep_sequential_acm_corpus_calls(model_class, responses_api):
    kwargs = {"litellm_model_registry": None} if issubclass(model_class, LitellmModel) else {}
    model = model_class(model_name="test", research_tools="corpus", context_tools="acm", **kwargs)
    calls = _calls()
    if responses_api:
        response = {
            "status": "completed",
            "output": [
                {"type": "function_call", "name": call["name"], "arguments": call["arguments"], "call_id": call["id"]}
                for call in calls
            ],
        }
    else:
        tool_calls = [
            {"id": call["id"], "function": {"name": call["name"], "arguments": call["arguments"]}} for call in calls
        ]
        response = {"choices": [{"message": {"tool_calls": tool_calls}, "finish_reason": "tool_calls"}]}
        if model_class is LitellmModel:
            response = SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            tool_calls=[
                                SimpleNamespace(id=call["id"], function=SimpleNamespace(**call["function"]))
                                for call in tool_calls
                            ]
                        ),
                        finish_reason="tool_calls",
                    )
                ]
            )
    assert model._parse_actions(response) == [
        {"type": "search", "command": "search alpha beta", "tool_call_id": "search"},
        {"type": "manage_context", "tool_call_id": "manage"},
        {"type": "query_memory", "summary_id": 1, "query": "Evidence?", "tool_call_id": "memory"},
        {"type": "get_document", "command": "get_document doc:1", "tool_call_id": "document"},
    ]


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("search", {"query": ""}),
        ("search", {"query": "   "}),
        ("search", {"query": 3}),
        ("search", {"query": "valid", "command": "forged"}),
        ("get_document", {"docid": ""}),
        ("get_document", {"docid": None}),
        ("get_document", {"docid": "valid", "query": "extra"}),
    ],
)
def test_invalid_corpus_calls_keep_identity_without_executable_command(name, args):
    action = parse_native_calls([{"name": name, "arguments": json.dumps(args), "id": "real"}])[0]
    assert action["type"] == name and action["tool_call_id"] == "real"
    assert action["error"] and "command" not in action


def test_acm_mixed_errors_and_duplicates_keep_individual_slots_in_order():
    calls = [
        {"name": "manage_context", "arguments": "invalid", "id": "bad-manage"},
        {"name": "manage_context", "arguments": "{}", "id": "good-manage"},
        {"name": "query_memory", "arguments": '{"summary_id":1,"query":"why?"}', "id": "good-query"},
        {"name": "query_memory", "arguments": '{"tag":"ctx-0001","query":"why?"}', "id": "bad-query"},
        {"name": "bash", "arguments": "invalid", "id": "bad-bash"},
        {"name": "unknown", "arguments": "{}", "id": "unknown"},
        {"name": "search", "arguments": '{"query":"continue"}', "id": "last"},
    ]
    actions = parse_native_calls(calls)
    assert [action["tool_call_id"] for action in actions] == [call["id"] for call in calls]
    assert [bool(action.get("error")) for action in actions] == [True, False, False, True, True, True, False]
    assert actions[0]["type"] == "manage_context" and actions[3]["type"] == "query_memory"
    assert "tag" not in actions[3] and "summary_id" not in actions[3]
    assert actions[4]["type"] == actions[5]["type"] == "rejected"
    assert actions[-1] == {"type": "search", "command": "search continue", "tool_call_id": "last"}
