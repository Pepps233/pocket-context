"""Deterministic replay integration tests for the local baseline adaptations."""

import copy
import json
import sqlite3
from types import SimpleNamespace

import pytest

from pocketcontextagent.agents.baselines import ACMAgent, ReActAgent, ReSumAgent
from pocketcontextagent.agents.context import ContextAssemblyError, encode, relationships
from pocketcontextagent.environments.local import LocalEnvironment
from pocketcontextagent.exceptions import FormatError
from pocketcontextagent.models.test_models import (
    DeterministicModel,
    DeterministicResponseAPIToolcallModel,
    DeterministicToolcallModel,
    make_output,
    make_response_api_output,
)
from pocketcontextagent.models.utils.compact_tool import context_tools, parse_native_calls


class ReplayMixin:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.requests = []
        self.options = []

    def query(self, messages, **kwargs):
        self.requests.append(copy.deepcopy(messages))
        self.options.append(kwargs)
        return super().query(messages, **kwargs)


class ChatReplay(ReplayMixin, DeterministicToolcallModel):
    pass


class ResponsesReplay(ReplayMixin, DeterministicResponseAPIToolcallModel):
    pass


@pytest.fixture(params=[False, True], ids=["chat", "responses"])
def api(request):
    return request.param


def turn(api, name, args, call_id, *, text="work", cost=0.1):
    return batch(api, [(name, args, call_id)], text=text, cost=cost)


def batch(api, calls, *, text="work", cost=0.1):
    native = [{"name": name, "arguments": json.dumps(args), "id": call_id} for name, args, call_id in calls]
    extra = {"actions": parse_native_calls(native), "cost": cost}
    if api:
        return {
            "object": "response",
            "output": [
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]},
                *[
                    {
                        "type": "function_call",
                        "call_id": call["id"],
                        "name": call["name"],
                        "arguments": call["arguments"],
                    }
                    for call in native
                ],
            ],
            "extra": extra,
        }
    return {
        "role": "assistant",
        "content": text,
        "tool_calls": [
            {"id": call["id"], "type": "function", "function": {"name": call["name"], "arguments": call["arguments"]}}
            for call in native
        ],
        "extra": extra,
    }


def plain(api, text, *, cost=0.25):
    response = make_response_api_output(text, []) if api else make_output(text, [])
    response["extra"]["cost"] = cost
    return response


def make_agent(tmp_path, api, cls, outputs, **kwargs):
    return cls(
        (ResponsesReplay if api else ChatReplay)(outputs=outputs),
        LocalEnvironment(),
        system_template="Keep the system instructions",
        instance_template="{{task}}",
        cost_limit=0,
        context_db_path=tmp_path / "context.sqlite3",
        **kwargs,
    )


def start(agent):
    agent.add_messages(
        agent.model.format_message(role="system", content="Keep the system instructions"),
        agent.model.format_message(role="user", content="Keep the original task"),
    )


def blocks(agent):
    with sqlite3.connect(agent.context.db_path) as db:
        return [json.loads(row[0]) for row in db.execute("SELECT record FROM blocks ORDER BY tag")]


@pytest.mark.parametrize(("cls",), [(ReActAgent,), (ReSumAgent,)])  # noqa: PT006
def test_no_memory_schema_catalog_or_side_effects(tmp_path, api, cls):
    marker = tmp_path / "must-not-exist"
    agent = make_agent(
        tmp_path,
        api,
        cls,
        [
            turn(api, "bash", {"command": "echo evidence"}, "a"),
            turn(api, "bash", {"command": "echo latest"}, "b"),
            batch(api, [("bash", {"command": f"touch {marker}"}, "shell"), ("manage_context", {}, "memory")]),
            turn(api, "expand_block", {"tag": "ctx-0001"}, "expand"),
            turn(api, "query_memory", {"tag": "ctx-0001", "query": "secret"}, "recall"),
        ],
    )
    start(agent)
    agent.step()
    agent.step()
    assert len(agent.step()) == 2
    agent.step()
    agent.step()
    assert not marker.exists() and agent.context.revision == 0
    assert agent.model.config.context_tools == "none" and not context_tools(agent.model.config.context_tools)
    assert all("Eligible context blocks" not in encode(request) for request in agent.model.requests)
    assert "evidence" in encode(agent.model.requests[-1])
    assert not agent.compactions and not agent.memory_queries and not blocks(agent)
    assert all(message["extra"]["returncode"] == -1 for message in agent.messages[-4:] if relationships(message)[1])


def test_resum_rolls_summaries_preserves_latest_pairs_and_audit(tmp_path, api):
    agent = make_agent(
        tmp_path,
        api,
        ReSumAgent,
        [
            turn(api, "bash", {"command": "printf 'RAW-FIRST-%0800d' 0"}, "a", text="first evidence"),
            turn(api, "bash", {"command": "echo RAW-SECOND"}, "b", text="second evidence"),
            plain(api, "FIRST-SUMMARY"),
            turn(api, "bash", {"command": "echo RAW-THIRD"}, "c", text="third evidence"),
            plain(api, "ROLLING-SUMMARY"),
            turn(api, "bash", {"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"}, "done"),
        ],
        resum_token_threshold=1,
    )
    assert agent.run("Keep the original task")["exit_status"] == "Submitted"
    assert agent.context.revision == 2 and len(agent.compactions) == 2 and not blocks(agent)
    first_main, second_main = agent.model.requests[3], agent.model.requests[5]
    assert "FIRST-SUMMARY" in encode(first_main) and "RAW-FIRST" not in encode(first_main)
    assert "RAW-SECOND" in encode(first_main) and "RAW-THIRD" in encode(second_main)
    assert "ROLLING-SUMMARY" in encode(second_main) and "FIRST-SUMMARY" not in encode(second_main)
    assert "FIRST-SUMMARY" in encode(agent.compactions[1]["messages"])
    assert all(
        "Keep the system instructions" in encode(request) and "Keep the original task" in encode(request)
        for request in [first_main, second_main]
    )
    assert "RAW-FIRST" in encode(agent.messages) and "RAW-SECOND" in encode(agent.messages)
    for request in (first_main, second_main):
        calls = [call for message in request for call in relationships(message)[0]]
        results = [result for message in request for result in relationships(message)[1]]
        assert sorted(calls) == sorted(results)
        assert "Eligible context blocks" not in encode(request)
    assert all(options.get("final_answer") for options in (agent.model.options[2], agent.model.options[4]))
    assert agent.cost == pytest.approx(0.9)
    assert agent.serialize()["info"]["model_stats"]["api_calls"] == 6
    agent.context.active.pop(2)
    with pytest.raises(ContextAssemblyError, match="committed replacement"):
        agent.query()


def test_acm_incremental_archives_and_model_mediated_recall(tmp_path, api):
    agent = make_agent(
        tmp_path,
        api,
        ACMAgent,
        [
            turn(api, "bash", {"command": "echo RAW-FIRST-EVIDENCE"}, "a"),
            turn(api, "manage_context", {}, "compact1"),
            plain(api, "FIRST-MEMORY"),
            turn(api, "bash", {"command": "echo RAW-SECOND-EVIDENCE"}, "b"),
            turn(api, "manage_context", {}, "compact2"),
            plain(api, "SECOND-MEMORY"),
            turn(api, "query_memory", {"tag": "ctx-0001", "query": "What did first evidence show?"}, "recall"),
            plain(api, "Focused keypoint only"),
            turn(api, "expand_block", {"tag": "ctx-0001"}, "forbidden"),
            turn(api, "bash", {"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"}, "done"),
        ],
    )
    assert agent.run("Keep the original task")["exit_status"] == "Submitted"
    stored = blocks(agent)
    assert len(stored) == 2 and agent.context.revision == 2
    assert stored[0]["messages"] == agent.messages[2:4]
    assert "RAW-FIRST-EVIDENCE" not in encode(stored[1]["messages"])
    assert "RAW-SECOND-EVIDENCE" in encode(stored[1]["messages"])
    assert "FIRST-MEMORY" not in encode(agent.compactions[1]["messages"])
    assert "RAW-FIRST-EVIDENCE" not in encode(agent.compactions[1]["messages"])
    assert "Keep the original task" in encode(agent.compactions[1]["messages"])
    query_request = encode(agent.memory_queries[0]["messages"])
    assert "What did first evidence show?" in query_request and "RAW-FIRST-EVIDENCE" in query_request
    assert "RAW-SECOND-EVIDENCE" not in query_request
    assert "Focused keypoint only" in encode(agent.model.requests[8])
    assert "RAW-FIRST-EVIDENCE" not in encode(agent.model.requests[8])
    assert not agent.context.expansions
    assert agent.model.config.context_tools == "acm"
    assert {tool["function"]["name"] for tool in context_tools("acm")} == {"manage_context", "query_memory"}
    assert all("Eligible context blocks" not in encode(request) for request in agent.model.requests)
    assert agent.cost == pytest.approx(1.45)
    stats = agent.serialize()["info"]["model_stats"]
    assert stats["api_calls"] == 10 and stats["compaction_api_calls"] == 2 and stats["memory_query_api_calls"] == 1
    assert len(agent.messages) == 16 and len(agent.compactions) == 2


@pytest.mark.parametrize(("kind",), [("summary",), ("query",)])  # noqa: PT006
@pytest.mark.parametrize(("failure",), [("empty",), ("tools",), ("truncated",), ("format",)])  # noqa: PT006
def test_malformed_auxiliary_calls_are_charged_and_never_execute(tmp_path, api, kind, failure):
    marker = tmp_path / "must-not-exist"
    response = plain(api, "" if failure == "empty" else "malformed", cost=0.75)
    if failure == "tools":
        response = turn(api, "bash", {"command": f"touch {marker}"}, "child", cost=0.75)
    if failure == "truncated":
        response["extra"]["response"] = {"status": "incomplete"}
    if failure == "format":
        response = make_output("", [{"raise": FormatError(response)}])
    outputs = [turn(api, "bash", {"command": "echo evidence"}, "a"), turn(api, "manage_context", {}, "compact")]
    if kind == "query":
        outputs += [
            plain(api, "summary"),
            turn(api, "query_memory", {"tag": "ctx-0001", "query": "evidence?"}, "query"),
        ]
    outputs.append(response)
    agent = make_agent(tmp_path, api, ACMAgent, outputs)
    start(agent)
    agent.step()
    if kind == "query":
        agent.step()
    before = copy.deepcopy(agent.context.projection())
    assert agent.step()[0]["extra"]["returncode"] == -1
    assert not marker.exists()
    calls = agent.compactions if kind == "summary" else agent.memory_queries
    assert len(calls) == 1 and calls[0]["response"]["extra"]["cost"] == 0.75
    assert agent.cost == pytest.approx(0.95 if kind == "summary" else 1.30)
    assert agent.context.projection()[: len(before)] == before
    assert agent.context.revision == (kind == "query")
    assert len(blocks(agent)) == (kind == "query")


def test_acm_rejects_ranges_mixed_batches_and_stale_dispatch_without_calls(tmp_path, api):
    marker = tmp_path / "must-not-exist"
    agent = make_agent(
        tmp_path,
        api,
        ACMAgent,
        [
            turn(api, "bash", {"command": "echo evidence"}, "a"),
            turn(api, "manage_context", {"start_id": "b001"}, "range"),
            batch(api, [("manage_context", {}, "manage"), ("bash", {"command": f"touch {marker}"}, "shell")]),
            turn(api, "manage_context", {}, "stale"),
            turn(api, "bash", {"command": "echo newer"}, "newer"),
        ],
    )
    start(agent)
    agent.step()
    assert agent.step()[0]["extra"]["returncode"] == -1
    assert len(agent.step()) == 2
    stale = agent.query()
    agent.query()
    assert "Stale selection" in encode(agent.execute_actions(stale))
    assert not marker.exists() and not agent.compactions and not agent.memory_queries and not blocks(agent)


def test_acm_dispatch_is_idempotent_and_sqlite_failure_preserves_history(tmp_path, api):
    agent = make_agent(
        tmp_path,
        api,
        ACMAgent,
        [
            turn(api, "bash", {"command": "echo evidence"}, "a"),
            turn(api, "manage_context", {}, "compact"),
            plain(api, "memory"),
        ],
    )
    start(agent)
    agent.step()
    message = agent.query()
    with sqlite3.connect(agent.context.db_path) as db:
        db.execute("CREATE TRIGGER fail BEFORE INSERT ON events BEGIN SELECT RAISE(ABORT, 'disk failure'); END")
    before = copy.deepcopy(agent.context.projection())
    first = agent.execute_actions(message)
    assert "disk failure" in encode(first)
    assert agent.execute_actions(message) == first
    assert agent.context.revision == 0 and not blocks(agent)
    assert agent.context.projection()[: len(before)] == before and len(agent.compactions) == 1


def test_resum_storage_failure_preserves_history_and_continues(tmp_path, api):
    agent = make_agent(
        tmp_path,
        api,
        ReSumAgent,
        [
            turn(api, "bash", {"command": "echo original evidence"}, "a"),
            turn(api, "bash", {"command": "echo latest evidence"}, "b"),
            plain(api, "summary"),
            turn(api, "bash", {"command": "echo continued"}, "c"),
        ],
        resum_token_threshold=1,
    )
    start(agent)
    agent.step()
    agent.step()
    before = copy.deepcopy(agent.context.projection())
    with sqlite3.connect(agent.context.db_path) as db:
        db.execute("CREATE TRIGGER fail BEFORE INSERT ON events BEGIN SELECT RAISE(ABORT, 'disk failure'); END")
    assert "continued" in encode(agent.step())
    assert agent.context.projection()[: len(before)] == before
    assert agent.context.revision == 0 and not blocks(agent)
    assert len(agent.compactions) == 1 and agent.cost == pytest.approx(0.55)


@pytest.mark.parametrize(("cls",), [(ReActAgent,), (ReSumAgent,), (ACMAgent,)])  # noqa: PT006
def test_unsupported_model_interfaces_fail_closed(tmp_path, cls):
    with pytest.raises(TypeError, match="native-tool"):
        cls(DeterministicModel(outputs=[]), LocalEnvironment(), system_template="s", instance_template="u")
    model = SimpleNamespace(config=SimpleNamespace(action_regex="bash", context_tools="pocket_context"))
    with pytest.raises(TypeError, match="native-tool"):
        cls(model, LocalEnvironment(), system_template="s", instance_template="u")
    assert model.config.context_tools == "pocket_context"
    with pytest.raises(TypeError, match="native-tool"):
        cls(SimpleNamespace(config=SimpleNamespace()), LocalEnvironment(), system_template="s", instance_template="u")
