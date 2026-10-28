"""Pinned ACM prompt/parser contracts and native replay over real SQLite archives."""

import copy
import hashlib
import json
import sqlite3
from types import SimpleNamespace

import pytest

from pocketcontextagent.agents.context import ContextAssemblyError, SelectionError, encode, relationships
from pocketcontextagent.comparisons.methods.acm import ACMAgent, _extract_memory, _strip_thinking
from pocketcontextagent.comparisons.methods.acm_prompts import (
    _SUMMARY_INSTRUCTION,
    BCP_STOP_PROMPT,
    QUERY_MEMORY_PROMPT,
    build_system_prompt,
)
from pocketcontextagent.comparisons.methods.base import conversation_text, public_messages
from pocketcontextagent.environments.browsecomp_plus import BrowseCompPlusEnvironment
from pocketcontextagent.exceptions import FormatError
from pocketcontextagent.models.test_models import (
    DeterministicResponseAPIToolcallModel,
    DeterministicToolcallModel,
    make_output,
    make_response_api_output,
)
from pocketcontextagent.models.utils.actions_toolcall import parse_toolcall_actions
from pocketcontextagent.models.utils.actions_toolcall_response import parse_toolcall_actions_response

MEMORY = "## Knowledge state\nMontréal hosts the jazz festival [city].\n\n## Thoughts\nFetch [city] to verify the year."
QUESTION = "Which city hosts the jazz festival?"
FINAL = "Explanation: The evidence names the city [city].\nExact Answer: Montréal\nConfidence: 100%"


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


class SQLiteCorpus:
    """Small actual SQL corpus; retrieval and archive storage do not require network mocks."""

    def __init__(self, path):
        self.path = path
        with sqlite3.connect(path) as db:
            db.execute("CREATE TABLE documents (docid TEXT PRIMARY KEY, text TEXT)")
            db.executemany(
                "INSERT INTO documents VALUES (?, ?)",
                [
                    ("city", "RAW-FIRST Montréal jazz festival 1980"),
                    ("other", "RAW-SECOND Bananas grow in tropical climates"),
                ],
            )

    def search(self, query, k):
        with sqlite3.connect(self.path) as db:
            return [
                {"docid": docid, "text": text, "score": 1.0}
                for docid, text in db.execute(
                    "SELECT docid, text FROM documents WHERE instr(text, ?) > 0 LIMIT ?", (query, k)
                )
            ]

    def get_document(self, docid):
        with sqlite3.connect(self.path) as db:
            row = db.execute("SELECT text FROM documents WHERE docid = ?", (docid,)).fetchone()
        return {"docid": docid, "text": row[0]} if row else None


@pytest.fixture(params=[False, True], ids=["chat", "responses"])
def api(request):
    return request.param


def batch(api, calls, *, text="research reasoning", cost=0.1):
    native = [
        {"name": name, "arguments": args if isinstance(args, str) else json.dumps(args), "id": call_id}
        for name, args, call_id in calls
    ]
    if api:
        output = [
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]},
            *[
                {"type": "function_call", "name": call["name"], "arguments": call["arguments"], "call_id": call["id"]}
                for call in native
            ],
        ]
        return {
            "object": "response",
            "output": output,
            "extra": {
                "actions": parse_toolcall_actions_response(output, format_error_template="{{error}}"),
                "cost": cost,
            },
        }
    tool_calls = [
        {"id": call["id"], "type": "function", "function": {"name": call["name"], "arguments": call["arguments"]}}
        for call in native
    ]
    return {
        "role": "assistant",
        "content": text,
        "tool_calls": tool_calls,
        "extra": {
            "actions": parse_toolcall_actions(
                [SimpleNamespace(id=call["id"], function=SimpleNamespace(**call["function"])) for call in tool_calls],
                format_error_template="{{error}}",
            ),
            "cost": cost,
        },
    }


def turn(api, name, args, call_id, **kwargs):
    return batch(api, [(name, args, call_id)], **kwargs)


def plain(api, text, *, cost=0.25):
    message = make_response_api_output(text, []) if api else make_output(text, [])
    message["extra"]["cost"] = cost
    return message


def make_agent(tmp_path, api, outputs):
    return ACMAgent(
        (ResponsesReplay if api else ChatReplay)(outputs=outputs, observation_template="{{output.output}}"),
        BrowseCompPlusEnvironment(SQLiteCorpus(tmp_path / "corpus.sqlite3"), context_tools="acm"),
        system_template="{% raw %}" + build_system_prompt() + "{% endraw %}",
        instance_template="{{task}}",
        context_db_path=tmp_path / "context.sqlite3",
        cost_limit=0,
    )


def start(agent):
    agent.extra_template_vars["task"] = QUESTION
    agent.add_messages(
        agent.model.format_message(role="system", content=build_system_prompt()),
        agent.model.format_message(role="user", content=QUESTION),
    )


def assert_native_pairs(messages):
    calls = [call for message in messages for call in relationships(message)[0]]
    results = [result for message in messages for result in relationships(message)[1]]
    assert sorted(calls) == sorted(results)
    assert len(calls) == len(set(calls))


def test_prompts_are_verbatim_pinned_release():
    # SHA-256 values computed from f06f90e's original src/prompts.py, not this port.
    for prompt, digest in [
        (_SUMMARY_INSTRUCTION, "9ddff032f0afb6a718c9cd2818abe21790b2e1174eca83a092930411b8d40670"),
        (QUERY_MEMORY_PROMPT, "18ea58cdb9b14fc1b160202979a8f916a3d6fddb26dd7f0fbf181bd966a633aa"),
        (BCP_STOP_PROMPT, "6af63de00bebb071f01f84051b8a7a803e6a86e6099c3958eb267d72e282a2cc"),
        (build_system_prompt(32768), "976bf5cb4ae56c77f388f6df63348f6414276927f78100296d421cb6071e6ad4"),
        (
            build_system_prompt(32768, use_memory_tools=False),
            "09599fdfa6a367fc04a7066d1800015589058259a15fce1db654ef8ec2ce71ae",
        ),
    ]:
        assert hashlib.sha256(prompt.encode()).hexdigest() == digest


@pytest.mark.parametrize(
    ("text", "expected", "parse_ok"),
    [
        (f"<think>secret reasoning</think><memory>{MEMORY}</memory>", MEMORY, True),
        ("<memory>" + "a" * 49 + "</memory>", "<memory>" + "a" * 49 + "</memory>", False),
        ("<memory>" + "a" * 50 + "</memory>", "a" * 50, True),
        ("<memory>restart<memory>" + MEMORY + "</memory>", "<memory>restart<memory>" + MEMORY + "</memory>", False),
        ("<memory>" + MEMORY + "</think></memory>", "<memory>" + MEMORY + "</think></memory>", False),
        ("unstructured fallback", "unstructured fallback", False),
        ("", "", False),
    ],
)
def test_exact_released_parser_and_thinking_rule(text, expected, parse_ok):
    assert _extract_memory(_strip_thinking(text)) == (expected, parse_ok)


def test_two_segments_are_tool_results_and_recall_only_uses_selected_raw_archive(tmp_path, api):
    agent = make_agent(
        tmp_path,
        api,
        [
            turn(api, "search", {"query": "jazz"}, "first"),
            turn(api, "manage_context", {}, "compact1"),
            plain(api, f"<think>private summary planning</think><memory>{MEMORY}</memory>"),
            turn(api, "get_document", {"docid": "other"}, "second"),
            turn(api, "manage_context", {}, "compact2"),
            plain(api, f"<memory>{MEMORY.replace('Montréal', 'SECOND-MEMORY')}</memory>"),
            turn(api, "query_memory", {"summary_id": 1, "query": "What year did the first evidence name?"}, "recall"),
            plain(api, "<think>private recall planning</think>- **Relevant findings:** 1980 [city]."),
            plain(api, FINAL, cost=0.1),
        ],
    )
    assert agent.run(QUESTION)["exit_status"] == "Submitted"
    assert agent.context.archive(1) == public_messages(agent.messages[2:4])
    assert agent.context.archive(2) == public_messages(agent.messages[6:8])
    assert "RAW-FIRST" in encode(agent.context.archive(1)) and "RAW-SECOND" in encode(agent.context.archive(2))
    assert "compact1" not in encode(agent.context.archive(2)) and "summary_id" not in encode(agent.context.archive(2))
    assert "RAW-FIRST" not in encode(agent.context.archive(2))
    assert agent.compactions[0]["messages"][0] == agent.model.format_message(
        role="user",
        content=_SUMMARY_INSTRUCTION.format(question=QUESTION, conversation=conversation_text(agent.messages[2:4])),
    )
    assert "RAW-FIRST" not in encode(agent.compactions[1]["messages"])
    assert MEMORY not in encode(agent.compactions[1]["messages"])
    recall = encode(agent.memory_queries[0]["messages"])
    assert "RAW-FIRST" in recall and "RAW-SECOND" not in recall and "What year" in recall
    final_request = agent.model.requests[-1]
    assert "1980 [city]" in encode(final_request) and "private recall planning" not in encode(final_request)
    assert "RAW-FIRST" not in encode(final_request) and "RAW-SECOND" not in encode(final_request)
    summaries = [message for message in final_request if "## Knowledge state" in conversation_text([message])]
    assert len(summaries) == 2
    for message in summaries:
        assert relationships(message)[1] in (["compact1"], ["compact2"])
        assert "Assistant-generated memory" not in encode(message)
    assert_native_pairs(final_request)
    assert agent._boundary == 6 and agent._summary_id == 2
    assert all(call["parse_ok"] and call["committed"] for call in agent.compactions)
    assert all("Eligible context blocks" not in encode(request) for request in agent.model.requests)
    assert agent.cost == pytest.approx(1.35)
    stats = agent.serialize()["info"]["model_stats"]
    assert stats["api_calls"] == 9 and stats["compaction_api_calls"] == 2 and stats["memory_query_api_calls"] == 1
    assert all(agent.model.options[index]["tools"] == [] for index in (2, 5, 7))
    assert all(agent.model.options[index]["max_tokens"] == 4096 for index in (2, 5))
    assert agent.model.options[7]["max_tokens"] == agent.config.response_token_limit
    assert "drop_call_message=True" in agent.serialize()["acm_provenance"]["boundary_deviation"]
    with sqlite3.connect(agent.context.db_path) as db:
        assert db.execute("SELECT summary_id FROM comparison_archives ORDER BY summary_id").fetchall() == [(1,), (2,)]
        assert db.execute("SELECT count(*) FROM events").fetchone()[0] == 2
        assert db.execute("SELECT count(*) FROM blocks").fetchone()[0] == 0
    with pytest.raises(SelectionError, match="not found"):
        agent.context.archive(3)


@pytest.mark.parametrize(
    ("attempts", "expected", "parse_ok"),
    [
        (["not tagged", f"<memory>{MEMORY}</memory>"], MEMORY, True),
        (
            ["", "<think>hidden</think>best-effort unstructured final attempt"],
            "best-effort unstructured final attempt",
            False,
        ),
        (["<memory>...</memory>", "<memory>still too short</memory>"], "<memory>still too short</memory>", False),
        (["", ""], "", False),
    ],
)
def test_summary_retries_once_and_keeps_original_fallback_with_full_costs(tmp_path, api, attempts, expected, parse_ok):
    agent = make_agent(
        tmp_path,
        api,
        [
            turn(api, "search", {"query": "jazz"}, "evidence"),
            turn(api, "manage_context", {}, "compact"),
            *[plain(api, text) for text in attempts],
        ],
    )
    start(agent)
    agent.step()
    result = agent.step()
    assert result[0]["extra"]["raw_output"] == f"[summary_id: 1] {expected}"
    assert result[0]["extra"]["parse_ok"] is parse_ok
    assert len(agent.compactions) == 2 and agent.compactions[0]["messages"] == agent.compactions[1]["messages"]
    assert agent.compactions[-1]["parse_ok"] is parse_ok and agent.compactions[-1]["committed"]
    assert agent.cost == pytest.approx(0.7)
    assert agent.serialize()["info"]["model_stats"]["api_calls"] == 4
    assert agent.context.revision == 1 and len(agent.context.archive(1)) >= 2


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("manage_context", {"start_id": "b001", "end_id": "b002"}),
        ("manage_context", "malformed JSON"),
        ("query_memory", {"summary_id": "1", "query": "year?"}),
        ("query_memory", {"summary_id": True, "query": "year?"}),
        ("query_memory", {"summary_id": 0, "query": "year?"}),
        ("query_memory", {"tag": "ctx-0001", "query": "year?"}),
        ("query_memory", {"summary_id": 1, "query": ""}),
        ("query_memory", {"summary_id": 5, "query": "year?"}),
    ],
)
def test_invalid_or_nonexistent_memory_has_native_error_slot_without_auxiliary_calls(tmp_path, api, name, args):
    agent = make_agent(tmp_path, api, [turn(api, name, args, "original-id")])
    start(agent)
    results = agent.step()
    assert len(results) == 1 and results[0]["extra"]["returncode"] == -1
    assert relationships(results[0])[1] == ["original-id"]
    assert not agent.compactions and not agent.memory_queries and agent.context.revision == 0
    assert agent._boundary == 2 and agent._summary_id == 0


@pytest.mark.parametrize(("memory_first",), [(True,), (False,)])  # noqa: PT006
def test_mixed_native_calls_keep_full_sibling_exchange_out_of_next_archive(tmp_path, api, memory_first):
    siblings = [("manage_context", {}, "compact1"), ("search", {"query": "Bananas"}, "sibling")]
    agent = make_agent(
        tmp_path,
        api,
        [
            turn(api, "search", {"query": "jazz"}, "first"),
            batch(api, siblings if memory_first else siblings[::-1]),
            plain(api, f"<memory>{MEMORY}</memory>"),
            turn(api, "get_document", {"docid": "city"}, "second"),
            turn(api, "manage_context", {}, "compact2"),
            plain(api, f"<memory>{MEMORY}</memory>"),
        ],
    )
    start(agent)
    agent.step()
    assert len(agent.step()) == 2
    boundary = agent._boundary
    assert boundary == len(agent.context.active) == 5
    agent.step()
    agent.step()
    assert "sibling" not in encode(agent.context.archive(2))
    assert "RAW-SECOND" not in encode(agent.context.archive(2))
    assert "RAW-FIRST" in encode(agent.context.archive(2))
    assert agent.env.tool_call_counts == {"search": 2, "get_document": 1}
    assert_native_pairs(agent.context.projection())


def test_storage_failure_rolls_back_archive_deletion_and_boundary_and_dispatch_is_idempotent(tmp_path, api):
    agent = make_agent(
        tmp_path,
        api,
        [
            turn(api, "search", {"query": "jazz"}, "first"),
            turn(api, "manage_context", {}, "compact"),
            plain(api, f"<memory>{MEMORY}</memory>"),
        ],
    )
    start(agent)
    agent.step()
    message = agent.query()
    with sqlite3.connect(agent.context.db_path) as db:
        db.execute("CREATE TRIGGER fail BEFORE INSERT ON events BEGIN SELECT RAISE(ABORT, 'disk failure'); END")
    before = copy.deepcopy(agent.context.projection())
    results = agent.execute_actions(message)
    assert "disk failure" in encode(results) and results[0]["extra"]["returncode"] == -1
    assert agent.execute_actions(message) == results and len(agent.compactions) == 1
    assert agent._boundary == 2 and agent._summary_id == 0 and agent.context.revision == 0
    assert agent.context.projection()[: len(before)] == before
    assert not agent.compactions[-1]["committed"] and agent.cost == pytest.approx(0.45)
    with sqlite3.connect(agent.context.db_path) as db:
        assert db.execute("SELECT count(*) FROM comparison_archives").fetchone()[0] == 0
        assert db.execute("SELECT count(*) FROM events").fetchone()[0] == 0


def test_pc_batch_rejected_before_research_and_stale_memory_rejected_before_auxiliary(tmp_path, api):
    agent = make_agent(
        tmp_path,
        api,
        [
            batch(api, [("search", {"query": "jazz"}, "search"), ("expand_block", {"tag": "ctx-0001"}, "pc")]),
            turn(api, "manage_context", {}, "stale"),
            turn(api, "search", {"query": "Bananas"}, "later"),
        ],
    )
    start(agent)
    assert len(agent.step()) == 2
    assert agent.env.tool_call_counts == {"search": 0, "get_document": 0}
    stale = agent.query()
    agent.query()
    assert "Stale selection" in encode(agent.execute_actions(stale))
    assert not agent.compactions and not agent.memory_queries and agent.context.revision == 0
    agent.context.active.pop(2)
    with pytest.raises(ContextAssemblyError, match="committed replacement"):
        agent.query()


@pytest.mark.parametrize(("kind",), [("compaction",), ("query",)])  # noqa: PT006
def test_billed_auxiliary_format_failure_preserves_context_boundary_and_complete_ledger(tmp_path, api, kind):
    billed = plain(api, "Malformed billed provider response", cost=0.75)
    billed["extra"].update(cost_available=False, response={"usage": {"input_tokens": 123, "output_tokens": 17}})
    outputs = [
        turn(api, "search", {"query": "jazz"}, "evidence"),
        turn(api, "manage_context", {}, "compact"),
    ]
    if kind == "query":
        outputs.extend(
            [
                plain(api, f"<memory>{MEMORY}</memory>"),
                turn(api, "query_memory", {"summary_id": 1, "query": "Which year?"}, "recall"),
            ]
        )
    else:
        outputs.append(plain(api, "Malformed first attempt: no memory tags"))
    outputs.append(make_output("", [{"raise": FormatError(billed)}]))
    agent = make_agent(tmp_path, api, outputs)
    start(agent)
    agent.step()
    if kind == "query":
        agent.step()
    before = copy.deepcopy(agent.context.projection())
    boundary, summary_id = agent._boundary, agent._summary_id
    results = agent.step()
    assert results[0]["extra"]["returncode"] == -1 and "FormatError" in encode(results)
    assert agent.context.projection()[: len(before)] == before
    assert agent._boundary == boundary and agent._summary_id == summary_id
    ledger = agent.compactions if kind == "compaction" else agent.memory_queries
    assert ledger[-1]["response"] == billed and "FormatError" in ledger[-1]["error"]
    assert ledger[-1]["response"]["extra"]["response"]["usage"]["input_tokens"] == 123
    assert ledger[-1]["response"]["extra"]["cost_available"] is False
    assert len(ledger) == (2 if kind == "compaction" else 1)
    assert agent.cost == pytest.approx(1.2 if kind == "compaction" else 1.3)
    assert agent.context.revision == (0 if kind == "compaction" else 1)
    assert agent.env.tool_call_counts == {"search": 1, "get_document": 0}


def test_rerun_resets_boundaries_integer_ids_costs_and_ledgers_without_reusing_old_archives(tmp_path, api):
    episode = [
        turn(api, "search", {"query": "jazz"}, "evidence"),
        turn(api, "manage_context", {}, "compact"),
        plain(api, f"<memory>{MEMORY}</memory>"),
        plain(api, FINAL, cost=0.1),
    ]
    agent = make_agent(tmp_path, api, episode + copy.deepcopy(episode))
    assert agent.run(QUESTION)["exit_status"] == "Submitted"
    first_run = agent.context.run_id
    first_archive = agent.context.archive(1)
    assert agent.run(QUESTION)["exit_status"] == "Submitted"
    assert agent.context.run_id != first_run
    assert agent._boundary == 4 and agent._summary_id == 1
    assert agent.context.archive(1) == public_messages(agent.messages[2:4])
    assert "RAW-FIRST" in encode(agent.context.archive(1))
    assert len(agent.compactions) == 1 and not agent.memory_queries
    assert agent.n_calls == 3 and agent.cost == pytest.approx(0.55)
    assert agent.serialize()["info"]["model_stats"]["api_calls"] == 4
    assert agent.context.revision == 1 and len(agent.context.events) == 1
    with sqlite3.connect(agent.context.db_path) as db:
        assert db.execute("SELECT summary_id FROM comparison_archives ORDER BY run_id").fetchall() == [(1,), (1,)]
        assert (
            json.loads(
                db.execute("SELECT record FROM comparison_archives WHERE run_id = ?", (first_run,)).fetchone()[0]
            )
            == first_archive
        )
    assert_native_pairs(agent.context.projection())
