import copy
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from pocketcontextagent.agents.context import ContextAssemblyError, ContextManager, encode
from pocketcontextagent.agents.default import DefaultAgent
from pocketcontextagent.agents.interactive import InteractiveAgent
from pocketcontextagent.environments.local import LocalEnvironment
from pocketcontextagent.models.litellm_model import LitellmModel
from pocketcontextagent.models.litellm_response_model import LitellmResponseModel
from pocketcontextagent.models.test_models import DeterministicResponseAPIToolcallModel, DeterministicToolcallModel
from pocketcontextagent.models.utils.actions_toolcall import parse_toolcall_actions
from pocketcontextagent.models.utils.actions_toolcall_response import parse_toolcall_actions_response
from pocketcontextagent.models.utils.content_string import get_content_string
from pocketcontextagent.run.utilities.inspector import _messages_to_steps


def native_turn(name, arguments, call_id, content, response_api=False):
    return native_batch([(name, arguments, call_id)], content, response_api)


def native_batch(calls, content, response_api=False):
    if response_api:
        output = [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": content}]}]
        output.extend(
            {"type": "function_call", "name": name, "arguments": json.dumps(args), "call_id": call_id}
            for name, args, call_id in calls
        )
        return {
            "object": "response",
            "output": output,
            "extra": {
                "actions": parse_toolcall_actions_response(output, format_error_template="{{error}}"),
                "cost": 0,
            },
        }
    tool_calls = [
        {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
        for name, args, call_id in calls
    ]
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": tool_calls,
        "extra": {
            "actions": parse_toolcall_actions(
                [SimpleNamespace(id=call["id"], function=SimpleNamespace(**call["function"])) for call in tool_calls],
                format_error_template="{{error}}",
            ),
            "cost": 0,
        },
    }


class RecordingMixin:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.requests = []

    def query(self, messages, **kwargs):
        self.requests.append(copy.deepcopy(messages))
        return super().query(messages, **kwargs)


class RecordingModel(RecordingMixin, DeterministicToolcallModel):
    def _prepare_messages_for_api(self, messages):
        return LitellmModel(model_name="test", set_cache_control="default_end")._prepare_messages_for_api(messages)


class RecordingResponseModel(RecordingMixin, DeterministicResponseAPIToolcallModel):
    def _prepare_messages_for_api(self, messages):
        return LitellmResponseModel(model_name="test")._prepare_messages_for_api(messages)


@pytest.fixture(params=[False, True], ids=["chat", "responses"])
def api(request):
    return request.param


def compaction_args(**kwargs):
    return {
        "start_quote": "Inspect archive-start",
        "end_quote": "archive-output0000",
        "summary": "Inspection printed archive-output followed by zeros; command succeeded.",
        **kwargs,
    }


def make_agent(tmp_path, api, *, args=None, agent_class=DefaultAgent, outputs=None, **kwargs):
    model = (RecordingResponseModel if api else RecordingModel)(
        outputs=outputs
        or [
            native_turn(
                "bash", {"command": "printf 'archive-output'; printf '%03000d' 0"}, "a", "Inspect archive-start", api
            ),
            native_turn("bash", {"command": "echo retained-output"}, "c", "Retained latest output", api),
            native_turn("compact_block", args or compaction_args(), "compact", "Store inspection history", api),
            native_turn("bash", {"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"}, "done", "Finish", api),
        ]
    )
    return agent_class(
        model,
        LocalEnvironment(),
        system_template="System",
        instance_template="{{task}}",
        cost_limit=0,
        context_db_path=tmp_path / "context.sqlite3",
        **kwargs,
    )


def start_history(agent):
    agent.add_messages(
        agent.model.format_message(role="system", content="System"),
        agent.model.format_message(role="user", content="Keep working"),
    )
    agent.step()
    agent.step()


def read_blocks(agent):
    with sqlite3.connect(agent.context.db_path) as db:
        return [
            json.loads(row[0])
            for row in db.execute("SELECT record FROM blocks WHERE run_id = ? ORDER BY tag", (agent.context.run_id,))
        ]


def test_compaction_finishes_before_next_work_decision(tmp_path, api):
    agent = make_agent(tmp_path, api)
    assert agent.run("Keep working")["exit_status"] == "Submitted"
    assert len(agent.model.requests) == 4 and len(agent.messages) == 10
    assert "[ctx-0001]:" in get_content_string(agent.model.requests[3][2])
    assert "Assistant-generated memory" in get_content_string(agent.model.requests[3][2])
    assert "Retained latest output" in get_content_string(agent.model.requests[3][3])
    assert "0" * 1000 not in encode(agent.model.requests[3])
    assert "Inspection printed" not in get_content_string(agent.model.requests[3][-1])
    assert read_blocks(agent)[0]["messages"] == agent.messages[2:4]
    assert read_blocks(agent)[0]["relationships"] == {"m000003": [["a"], []], "m000004": [[], ["a"]]}
    serialized = agent.save(tmp_path / "trajectory.json")
    assert serialized["context"]["revision"] == 1
    assert json.loads((tmp_path / "trajectory.json").read_text())["messages"] == agent.messages
    assert _messages_to_steps(serialized["messages"])[1][0] == agent.messages[2]
    before, after = agent.context.manifests[2:4]
    event = agent.context.events[0]
    assert [key for key in before["ids"] if key not in event["selected_ids"]] == [
        key for key in after["ids"] if key in before["ids"]
    ]
    assert event["summary_id"] in after["ids"]
    assert before["characters"] - after["characters"] == read_blocks(agent)[0]["reduction"]
    payload = agent.model._prepare_messages_for_api(agent.model.requests[3])
    for record in agent.context.records.values():
        assert record.id not in encode(payload)
        assert all(item_id not in encode(payload) for item_id in record.item_ids.values())
    assert all(entry["item_ids"] for entry in after["entries"])
    with sqlite3.connect(agent.context.db_path) as db:
        assert db.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == 4


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"end_quote": "Inspect archive-start"}, "splits a tool exchange"),
        ({"start_quote": "absent"}, "matches 0"),
        ({"start_quote": "output"}, "matches"),
        ({"start_quote": "Retained latest output", "end_quote": "Inspect archive-start"}, "Reversed"),
        ({"start_quote": "System", "end_quote": "archive-output0000"}, "Protected"),
        ({"start_quote": "Keep working"}, "Protected"),
        ({"start_quote": "Retained latest output", "end_quote": "Retained latest output"}, "Protected"),
        ({"start_quote": "Store inspection history"}, "matches 0"),
        ({"summary": "x" * 3000}, "Net request reduction"),
    ],
)  # noqa: PT006
def test_selection_errors_leave_original_context(tmp_path, api, overrides, error):
    agent = make_agent(tmp_path, api, args=compaction_args(**overrides))
    start_history(agent)
    before = list(agent.context.active)
    results = agent.step()
    assert error in get_content_string(results[0])
    assert agent.context.active[: len(before)] == before
    assert agent.context.revision == 0 and not read_blocks(agent)
    assert (results[0].get("call_id") or results[0].get("tool_call_id")) == "compact"
    assert results[0]["extra"]["returncode"] == -1


def test_minimum_range_and_net_reduction_include_call(tmp_path, api):
    agent = make_agent(tmp_path, api, compact_min_chars=4000)
    start_history(agent)
    assert "select at least 4000" in get_content_string(agent.step()[0])
    assert agent.context.revision == 0
    agent = make_agent(tmp_path / "overhead", api, compact_min_reduction=2700, args=compaction_args(summary="S" * 200))
    start_history(agent)
    assert "Net request reduction" in get_content_string(agent.step()[0])
    assert not read_blocks(agent)


def test_multiple_calls_are_rejected_without_shell_effects(tmp_path, api):
    marker = tmp_path / "should-not-exist"
    for calls in (
        [
            ("bash", {"command": f"touch {marker}"}, "shell"),
            ("compact_block", compaction_args(), "compact"),
        ],
        [("compact_block", compaction_args(), "one"), ("compact_block", compaction_args(), "two")],
    ):
        turn = native_batch(calls, "Reject batch", api)
        agent = make_agent(tmp_path, api, outputs=[turn])
        agent.add_messages(agent.model.format_message(role="user", content="Task"))
        assert len(agent.step()) == 2
        assert all("No actions executed" in get_content_string(message) for message in agent.messages[-2:])
        assert not marker.exists() and agent.context.revision == 0
        assert [(message.get("call_id") or message.get("tool_call_id")) for message in agent.messages[-2:]] == [
            call[2] for call in calls
        ]


def test_unresolved_calls_and_multiple_results(tmp_path, api):
    agent = make_agent(tmp_path, api)
    agent.add_messages(agent.model.format_message(role="user", content="Task"))
    agent.model.config.outputs[2] = native_turn(
        "compact_block", compaction_args(end_quote="Inspect archive-start"), "compact", "Store inspection history", api
    )
    agent.query()
    agent.step()
    assert "unresolved" in get_content_string(agent.step()[0])
    assert not read_blocks(agent)
    batch = native_batch(
        [("bash", {"command": "echo alpha"}, "a1"), ("bash", {"command": "printf 'omega'; printf '%03000d' 0"}, "a2")],
        "Batch start",
        api,
    )
    for end, succeeds in [("alpha", False), ("omega0000", True)]:
        agent = make_agent(
            tmp_path,
            api,
            outputs=[
                batch,
                native_turn("bash", {"command": "echo retained"}, "c", "Retain", api),
                native_turn(
                    "compact_block", compaction_args(start_quote="Batch start", end_quote=end), "memory", "Compact", api
                ),
            ],
        )
        start_history(agent)
        assert ("Compacted" in get_content_string(agent.step()[0])) == succeeds
        assert agent.context.revision == int(succeeds)


def test_stale_response_and_repeated_dispatch(tmp_path, api):
    agent = make_agent(tmp_path, api)
    start_history(agent)
    response = agent.query()
    first = agent.execute_actions(response)
    assert agent.execute_actions(response) == first
    assert len(read_blocks(agent)) == 1 and agent.context.revision == 1
    action = {**response["extra"]["actions"][0], "summary": "conflicting summary"}
    from pocketcontextagent.agents.context import SelectionError

    with pytest.raises(SelectionError, match="Conflicting reuse"):
        agent.context.compact(action, response, agent.model, None, min_chars=0, min_reduction=0)
    agent = make_agent(tmp_path / "stale", api)
    start_history(agent)
    old = agent.query()
    agent.query()
    assert "Stale selection" in get_content_string(agent.execute_actions(old)[0])
    assert not read_blocks(agent)


def test_database_failure_preserves_projection(tmp_path, api):
    agent = make_agent(tmp_path, api)
    start_history(agent)
    response = agent.query()
    before = list(agent.context.active)
    with sqlite3.connect(agent.context.db_path) as locked:
        locked.execute("BEGIN IMMEDIATE")
        assert "locked" in get_content_string(agent.execute_actions(response)[0])
    assert agent.context.active[: len(before)] == before
    assert agent.context.revision == 0 and not read_blocks(agent)
    agent = make_agent(tmp_path / "write-failure", api)
    start_history(agent)
    response = agent.query()
    with sqlite3.connect(agent.context.db_path) as db:
        db.execute("CREATE TRIGGER fail_event BEFORE INSERT ON events BEGIN SELECT RAISE(ABORT, 'disk failure'); END")
    assert "disk failure" in get_content_string(agent.execute_actions(response)[0])
    assert not read_blocks(agent) and agent.context.revision == 0


def test_recompaction_keeps_archives_and_lineage(tmp_path, api):
    agent = make_agent(
        tmp_path,
        api,
        outputs=[
            native_turn(
                "bash", {"command": "printf 'archive-output'; printf '%09000d' 0"}, "a", "Inspect archive-start", api
            ),
            native_turn("bash", {"command": "echo retained"}, "c", "Retained", api),
            native_turn(
                "compact_block",
                compaction_args(summary="Detailed observation " + "x" * 2100),
                "first",
                "Compact first",
                api,
            ),
            native_turn("bash", {"command": "echo continued"}, "d", "Continue", api),
            native_turn(
                "compact_block",
                compaction_args(start_quote="[ctx-0001]:", end_quote="[ctx-0001]:"),
                "second",
                "Compact again",
                api,
            ),
        ],
    )
    start_history(agent)
    agent.step()
    agent.step()
    assert "[ctx-0002]" in get_content_string(agent.step()[0])
    blocks = read_blocks(agent)
    assert len(blocks) == 2 and blocks[1]["children"] == ["ctx-0001"]
    assert blocks[0]["messages"] == agent.messages[2:4]
    assert blocks[1]["version"] == 2 and blocks[1]["messages"] == blocks[1]["records"] == []
    assert blocks[1]["references"][0]["source_record"] == blocks[0]["summary_record"]["id"]
    assert "[ctx-0002 refs=ctx-0001]:" in encode(agent.context.projection())
    assert agent.serialize()["context"]["blocks"] == {"ctx-0001": [], "ctx-0002": ["ctx-0001"]}


def test_reasoning_and_multimodal_payloads_survive(tmp_path, api):
    agent = make_agent(tmp_path, api)
    if api:
        reasoning = {"type": "reasoning", "id": "reasoning", "summary": [], "encrypted_content": "opaque-signed-data"}
        agent.model.config.outputs[0]["output"].insert(0, reasoning)
        agent.model.config.outputs[1]["output"].insert(0, copy.deepcopy(reasoning))
    else:
        reasoning = {"type": "thinking", "thinking": "Private reasoning", "signature": "signed-data"}
        for index in (0, 1):
            agent.model.config.outputs[index]["content"] = [
                {"type": "text", "text": agent.model.config.outputs[index]["content"]},
                copy.deepcopy(reasoning),
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,original-bytes"}},
            ]
    start_history(agent)
    originals = copy.deepcopy(agent.messages[2:6])
    assert "Compacted" in get_content_string(agent.step()[0])
    assert read_blocks(agent)[0]["messages"] == originals[:2]
    active = agent.context.assemble(agent.model)
    assert agent.messages[2:6] == originals
    assert reasoning in (active[3]["output"] if api else active[3]["content"])
    if not api:
        assert active[3]["content"][-1]["image_url"]["url"].endswith("original-bytes")
    assert agent.context.records["m000003"].item_ids


def test_hidden_metadata_is_not_a_quote_and_assembly_errors_stop(tmp_path, api):
    agent = make_agent(tmp_path, api, args=compaction_args(start_quote="secret-metadata"))
    agent.model.config.outputs[0]["extra"]["provider_metadata"] = "secret-metadata"
    start_history(agent)
    assert "matches 0" in get_content_string(agent.step()[0])
    agent.context.active.pop(2)
    with pytest.raises(ContextAssemblyError, match="committed replacement"):
        agent.query()


def test_interactive_compaction_precedes_confirmation(tmp_path, api):
    agent = make_agent(tmp_path, api, agent_class=InteractiveAgent, mode="confirm")
    agent.add_messages(
        agent.model.format_message(role="user", content="Old task " + "x" * 3000),
        agent.model.format_message(role="user", content="New task"),
    )
    agent.model.current_index = 1
    agent.model.config.outputs[2] = native_turn(
        "compact_block", compaction_args(start_quote="Old task", end_quote="Old task"), "compact", "Store memory", api
    )
    assert "Compacted" in get_content_string(agent.step()[0])
    assert agent.context.revision == 1


def test_concurrent_runs_share_database_without_sharing_context(tmp_path):
    def run(index):
        agent = make_agent(tmp_path, bool(index % 2))
        assert agent.run(f"Task {index}")["exit_status"] == "Submitted"
        return agent

    with ThreadPoolExecutor(max_workers=2) as pool:
        agents = list(pool.map(run, range(2)))
    assert agents[0].context.run_id != agents[1].context.run_id
    assert all(len(read_blocks(agent)) == 1 for agent in agents)
    with sqlite3.connect(agents[0].context.db_path) as db:
        assert db.execute("SELECT COUNT(DISTINCT run_id) FROM blocks").fetchone()[0] == 2
    assert ContextManager().db_path.parent.name == "contexts"


def test_native_prompt_consistency_and_text_mode_unchanged():
    for name in ("mini", "benchmarks/swebench", "benchmarks/programbench", "benchmarks/browsecomp_plus"):
        config = yaml.safe_load(Path(f"pocketcontextagent/config/{name}.yaml").read_text())
        prompt = config["agent"]["system_template"] + config["agent"]["instance_template"]
        assert "exactly one compact_block" in prompt
        assert "expand_block" in prompt
        assert "compact_block" in config["model"]["format_error_template"]
        assert "expand_block" in config["model"]["format_error_template"]
        assert "AT LEAST ONE bash" not in encode(config)
    assert "compact_block" not in Path("pocketcontextagent/config/default.yaml").read_text()
    assert "expand_block" not in Path("pocketcontextagent/config/default.yaml").read_text()


def test_repeated_call_id_is_idempotent_and_conflicts_fail(tmp_path, api):
    agent = make_agent(tmp_path, api)
    start_history(agent)
    agent.step()
    agent.model.config.outputs.extend(
        [
            native_turn("compact_block", compaction_args(), "compact", "Repeat committed call", api),
            native_turn(
                "compact_block", compaction_args(summary="Different summary"), "compact", "Conflicting call", api
            ),
        ]
    )
    agent.model.current_index = 3
    assert "Compacted [ctx-0001]" in get_content_string(agent.step()[0])
    assert "Conflicting reuse" in get_content_string(agent.step()[0])
    assert len(read_blocks(agent)) == 1 and agent.context.revision == 1


def test_synthetic_feedback_does_not_replace_latest_user_protection(tmp_path, api):
    agent = make_agent(tmp_path, api, args=compaction_args(start_quote="Keep working"))
    start_history(agent)
    agent.add_messages(
        agent.model.format_message(role="user", content="Generated correction", extra={"interrupt_type": "FormatError"})
    )
    assert "Protected" in get_content_string(agent.step()[0])
    assert agent.context.records["m000007"].provenance == "feedback"


def test_unknown_provider_tools_are_rejected_safely(tmp_path, api):
    agent = make_agent(tmp_path, api)
    if api:
        agent.model.config.outputs[0]["output"].append({"type": "web_search_call", "id": "server-call"})
    else:
        agent.model.config.outputs[0]["content"] = [
            {"type": "text", "text": "Inspect archive-start"},
            {"type": "server_tool_use", "id": "server-call", "name": "web_search", "input": {}},
        ]
    start_history(agent)
    assert "Cannot safely group" in get_content_string(agent.step()[0])
    assert not read_blocks(agent)


def test_plaintext_reasoning_quotes_exclude_signatures(tmp_path, api):
    agent = make_agent(tmp_path, api, args=compaction_args(start_quote="Reasoned archive-start"))
    if api:
        agent.model.config.outputs[0]["output"].insert(
            0,
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "Reasoned archive-start"}],
                "encrypted_content": "hidden-signature",
            },
        )
    else:
        agent.model.config.outputs[0]["content"] = [
            {"type": "text", "text": "Inspect archive-start"},
            {"type": "thinking", "thinking": "Reasoned archive-start", "signature": "hidden-signature"},
        ]
    start_history(agent)
    assert "Compacted" in get_content_string(agent.step()[0])
    assert "hidden-signature" in encode(read_blocks(agent)[0]["messages"])
    from pocketcontextagent.agents.context import visible_text

    assert "hidden-signature" not in encode(visible_text(agent.messages[2]))
