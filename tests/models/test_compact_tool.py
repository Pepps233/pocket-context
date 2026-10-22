import json
from types import SimpleNamespace

import pytest

from pocketcontextagent.models.utils.actions_toolcall import parse_toolcall_actions
from pocketcontextagent.models.utils.actions_toolcall_response import parse_toolcall_actions_response
from pocketcontextagent.models.utils.compact_tool import COMPACT_TOOL, COMPACT_TOOL_RESPONSE_API


@pytest.mark.parametrize(("response_api",), [(False,), (True,)])  # noqa: PT006
def test_native_compaction_parsing_and_batch_rejection(response_api):
    args = {"start_quote": "first", "end_quote": "last"}

    def parse(calls):
        if response_api:
            return parse_toolcall_actions_response(
                [
                    {"type": "function_call", "name": name, "arguments": arguments, "call_id": call_id}
                    for name, arguments, call_id in calls
                ],
                format_error_template="{{error}}",
            )
        return parse_toolcall_actions(
            [
                SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=arguments))
                for name, arguments, call_id in calls
            ],
            format_error_template="{{error}}",
        )

    compact = ("compact_block", json.dumps(args), "memory")
    assert parse([compact]) == [{"type": "compact_block", **args, "tool_call_id": "memory"}]
    for calls in ([compact, ("bash", "invalid JSON", "shell")], [compact, compact]):
        rejected = parse(calls)
        assert len(rejected) == 2
        assert all(action["type"] == "rejected" and "No actions executed" in action["error"] for action in rejected)
    assert "error" in parse([("compact_block", "{}", "bad")])[0]
    assert (
        "error"
        in parse([("compact_block", json.dumps(args | {"summary": "Do not accept a caller summary"}), "old")])[0]
    )
    assert set(COMPACT_TOOL["function"]["parameters"]["properties"]) == {"start_quote", "end_quote"}
    assert parse([("bash", '{"command":"echo ok"}', "shell")]) == [{"command": "echo ok", "tool_call_id": "shell"}]
    assert COMPACT_TOOL["function"] == {key: value for key, value in COMPACT_TOOL_RESPONSE_API.items() if key != "type"}


def test_compaction_arguments_cannot_override_dispatch_identity():
    from pocketcontextagent.models.utils.compact_tool import parse_native_calls

    action = parse_native_calls(
        [
            {
                "name": "compact_block",
                "id": "real-call",
                "arguments": json.dumps(
                    {
                        "start_quote": "first",
                        "end_quote": "last",
                        "summary": "Keep results",
                        "type": "bash",
                        "tool_call_id": "forged-call",
                        "command": "touch forbidden",
                    }
                ),
            }
        ]
    )[0]
    assert action["type"] == "compact_block" and action["tool_call_id"] == "real-call"
    assert "error" in action and "command" not in action
    assert parse_native_calls([{"name": "bash", "id": "empty", "arguments": '{"command":""}'}]) == [
        {"command": "", "tool_call_id": "empty"}
    ]


@pytest.mark.parametrize(("tag",), [("ctx-0001",), ("ctx-10000",)])  # noqa: PT006
def test_expansion_schema_and_dedicated_turn(tag):
    from pocketcontextagent.models.utils.compact_tool import (
        EXPAND_TOOL,
        EXPAND_TOOL_RESPONSE_API,
        parse_native_calls,
    )

    call = {"name": "expand_block", "id": "expand", "arguments": json.dumps({"tag": tag})}
    assert parse_native_calls([call]) == [{"type": "expand_block", "tag": tag, "tool_call_id": "expand"}]
    for other in (call, {"name": "bash", "id": "shell", "arguments": "invalid"}):
        assert all(action["type"] == "rejected" for action in parse_native_calls([call, other]))
    assert EXPAND_TOOL["function"] == {key: value for key, value in EXPAND_TOOL_RESPONSE_API.items() if key != "type"}


@pytest.mark.parametrize(
    ("args",),  # noqa: PT006
    [
        ({},),
        ({"tag": "ctx-1"},),
        ({"tag": "ctx-0000"},),
        ({"tag": "ctx-00001"},),
        ({"tag": "ctx-0001", "command": "echo forbidden"},),
        ({"tag": 1},),
    ],
)  # noqa: PT006
def test_invalid_expansion_arguments_remain_tool_errors(args):
    from pocketcontextagent.models.utils.compact_tool import parse_native_calls

    action = parse_native_calls([{"name": "expand_block", "id": "real", "arguments": json.dumps(args)}])[0]
    assert action["type"] == "expand_block" and action["tool_call_id"] == "real"
    assert "error" in action and "command" not in action
