import json

import pytest

from pocketcontextagent.agents.context import encode, visible_text
from pocketcontextagent.models.utils.content_string import get_content_string
from tests.agents.test_compact_loop import api as api
from tests.agents.test_compact_loop import make_agent, native_batch, native_turn, read_blocks, start_history
from tests.agents.test_expand_loop import next_turn


def test_ids_resolve_all_results_and_survive_retries(tmp_path, api):
    agent = make_agent(tmp_path, api)
    agent.model.config.outputs[0] = native_batch(
        [
            ("bash", {"command": "echo repeated"}, "one"),
            ("bash", {"command": "printf 'repeated'; printf '%03000d' 0"}, "two"),
        ],
        "repeated",
        api,
    )
    start_history(agent)
    blocks = agent.context.eligible_blocks()
    assert blocks == {"b001": ["m000003", "m000004", "m000005"]}
    before = list(agent.context.active)
    for number in range(2):
        result = next_turn(agent, "compact_block", {"start_id": "b999", "end_id": "b001"}, f"bad-{number}", api)
        assert "Eligible IDs: b001" in get_content_string(result[0])
        assert agent.context.active[: len(before)] == before
        assert agent.context.eligible_blocks()["b001"] == blocks["b001"]
        assert not agent.compactions
    assert "b001: 3 messages" in encode(agent.model.requests[-1])
    assert "Compacted" in get_content_string(
        next_turn(
            agent,
            "compact_block",
            {"start_id": "b001", "end_id": "b001"},
            "valid",
            api,
        )[0]
    )
    assert read_blocks(agent)[0]["selection_order"] == blocks["b001"]
    assert "stale" in get_content_string(
        next_turn(
            agent,
            "compact_block",
            {"start_id": "b001", "end_id": "b001"},
            "stale",
            api,
        )[0]
    )
    assert len(agent.compactions) == 1
    assert "continued" in get_content_string(
        next_turn(agent, "bash", {"command": "echo continued"}, "continue", api)[0]
    )


def test_id_range_cannot_cross_protected_request(tmp_path, api):
    agent = make_agent(tmp_path, api)
    agent.add_messages(agent.model.format_message(role="user", content="Old task " + "x" * 3000))
    start_history(agent)
    blocks = agent.context.eligible_blocks()
    first, last = list(blocks)[0], list(blocks)[-1]
    result = next_turn(agent, "compact_block", {"start_id": first, "end_id": last}, "cross", api)
    assert "Protected" in get_content_string(result[0]) and "Eligible IDs" in get_content_string(result[0])
    assert not agent.compactions and not read_blocks(agent)


def test_expansions_get_distinct_complete_block_ids(tmp_path, api):
    agent = make_agent(tmp_path, api)
    start_history(agent)
    next_turn(agent, "compact_block", {"start_id": "b001", "end_id": "b001"}, "compact", api)
    for call in ("expand-one", "expand-two"):
        next_turn(agent, "expand_block", {"tag": "ctx-0001"}, call, api)
    blocks = agent.context.eligible_blocks()
    expanded = [key for key, group in blocks.items() if agent.context.records[group[0]].provenance == "expansion"]
    assert len(expanded) == 2 and all(len(blocks[key]) == 2 for key in expanded)
    assert "b001" not in blocks
    assert "Compacted" in get_content_string(
        next_turn(
            agent,
            "compact_block",
            {"start_id": expanded[0], "end_id": expanded[0]},
            "recompact",
            api,
        )[0]
    )
    assert expanded[0] not in agent.context.eligible_blocks() and expanded[1] in agent.context.eligible_blocks()
    assert read_blocks(agent)[1]["children"] == ["ctx-0001"]


def test_legacy_decoded_commands_ignore_failed_control_arguments(tmp_path, api):
    command = "printf '%s' 'escaped \"command\" \\path'; printf '%03000d' 0"
    agent = make_agent(tmp_path, api)
    agent.model.config.outputs[0] = native_turn("bash", {"command": command}, "a", "Archive", api)
    start_history(agent)
    for number in range(2):
        assert "matches 0" in get_content_string(
            next_turn(
                agent,
                "compact_block",
                {"start_quote": command, "end_quote": "absent"},
                f"bad-{number}",
                api,
            )[0]
        )
    result = next_turn(
        agent,
        "compact_block",
        {
            "start_quote": command,
            "end_quote": 'escaped "command" \\path0000',
        },
        "good",
        api,
    )
    assert "Compacted" in get_content_string(result[0])
    assert len(read_blocks(agent)[0]["records"]) == 2


@pytest.mark.parametrize(("kind",), [("chat",), ("responses",), ("anthropic",)])  # noqa: PT006
def test_anchor_values_are_decoded_without_control_arguments(kind):
    args = {"command": 'search "father" "police"\nnext'}
    if kind == "chat":
        message = {
            "tool_calls": [
                {"function": {"name": "bash", "arguments": json.dumps(args)}},
                {"function": {"name": "compact_block", "arguments": json.dumps({"start_quote": "poison"})}},
            ]
        }
    elif kind == "responses":
        message = {
            "object": "response",
            "output": [
                {"type": "function_call", "name": "bash", "arguments": json.dumps(args)},
                {"type": "function_call", "name": "expand_block", "arguments": '{"tag":"poison"}'},
            ],
        }
    else:
        message = {
            "content": [
                {"type": "tool_use", "name": "bash", "input": args},
                {"type": "tool_use", "name": "compact_block", "input": {"start_quote": "poison"}},
            ]
        }
    assert args["command"] in visible_text(message, anchors=True)
    assert "poison" not in encode(visible_text(message, anchors=True))
    assert "poison" in encode(visible_text(message))
