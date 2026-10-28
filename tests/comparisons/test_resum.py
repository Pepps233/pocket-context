"""ReSum parity and real-environment replay tests; no network or patched components."""

import copy
import hashlib
import json
import math

import pytest
from pydantic import ValidationError

from pocketcontextagent.comparisons.methods.resum import (
    UPSTREAM_COMMIT,
    ReSumAgent,
    ReSumConfig,
    normalize_summary,
    summary_prompt,
)
from pocketcontextagent.comparisons.methods.resum_prompts import (
    CONTINUATION_TEXT,
    QUERY_SUMMARY_PROMPT,
    QUERY_SUMMARY_PROMPT_LAST,
)
from pocketcontextagent.environments.local import LocalEnvironment
from pocketcontextagent.exceptions import FormatError, LimitsExceeded
from pocketcontextagent.models.test_models import (
    DeterministicResponseAPIToolcallModel,
    DeterministicToolcallModel,
    make_response_api_output,
    make_toolcall_output,
)

QUESTION = "Question sentinel"
SYSTEM = "Shared corpus instructions."
EVIDENCE_COMMAND = "printf 'fresh-evidence-marker'; printf '%026000d' 0"


@pytest.fixture(params=["chat", "responses"])
def api(request):
    return request.param


def output(api: str, content: str = "", command: str | None = None, call_id: str = "call") -> dict:
    actions = [{"command": command, "tool_call_id": call_id}] if command else []
    if api == "responses":
        return make_response_api_output(content, actions)
    calls = (
        [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "bash", "arguments": json.dumps({"command": command})},
            }
        ]
        if command
        else []
    )
    return make_toolcall_output(content, calls, actions)


def make_agent(tmp_path, api: str, outputs: list[dict], **kwargs) -> ReSumAgent:
    model_type = DeterministicToolcallModel if api == "chat" else DeterministicResponseAPIToolcallModel
    agent = ReSumAgent(
        model_type(outputs=outputs),
        LocalEnvironment(cwd=str(tmp_path)),
        system_template=SYSTEM,
        instance_template="{{task}}",
        context_db_path=tmp_path / "context.sqlite3",
        cost_limit=0,
        response_token_limit=128,
        **kwargs,
    )
    agent.extra_template_vars["task"] = QUESTION
    agent.add_messages(
        agent.model.format_message(role="system", content=SYSTEM),
        agent.model.format_message(role="user", content=QUESTION),
    )
    return agent


def text(message: dict) -> str:
    content = message.get("output") if message.get("type") == "function_call_output" else message.get("content", "")
    return content if isinstance(content, str) else "\n".join(block.get("text", "") for block in content or [])


def test_pinned_prompt_and_continuation_parity():
    # SHA256 of literal values in pinned prompt.py and new_observation in react_agent.py.
    assert hashlib.sha256(QUERY_SUMMARY_PROMPT.encode()).hexdigest() == (
        "7cabd8e4d6283c8142e4833efc29b8c7f1a97f53a40cd06d4beef084d7311ab1"
    )
    assert hashlib.sha256(QUERY_SUMMARY_PROMPT_LAST.encode()).hexdigest() == (
        "57c6316e7bf5378ce9eec7a76a4541e859bf330ce2531a2a2f5d50dbdbf700c0"
    )
    continuation = "Question: " + QUESTION + CONTINUATION_TEXT + "Summary: <summary>Evidence</summary>\n"
    assert hashlib.sha256(continuation.encode()).hexdigest() == (
        "7bb4bededb92e7714b3ca17b0ed8490601c838ea59d28de0c6079e08b2cbbb04"
    )
    history = [{"role": "assistant", "content": "A literal 'quote'"}, {"role": "user", "content": "Evidence"}]
    assert summary_prompt(QUESTION, history, None) == QUERY_SUMMARY_PROMPT.replace("{{{question}}}", QUESTION).replace(
        "{{{recent_history_messages}}}", "\n".join(str(message) for message in history)
    )
    assert summary_prompt(QUESTION, history, "<summary>Earlier</summary>") == QUERY_SUMMARY_PROMPT_LAST.replace(
        "{{{question}}}", QUESTION
    ).replace("{{{recent_history_messages}}}", "\n".join(str(message) for message in history)).replace(
        "{{{last_summary}}}", "<summary>Earlier</summary>"
    )


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("", ""),
        ("<think>hidden\nreasoning</think>\n<summary>Known fact</summary>", "<summary>Known fact</summary>"),
        ("  untagged evidence  ", "<summary>untagged evidence</summary>"),
        ("preface <summary> keep whitespace </summary> ignored", "<summary> keep whitespace </summary>"),
        ("<summary>missing closing tag", "<summary>missing closing tag</summary>"),
        ("<summary>first</summary><summary>second</summary>", "<summary>first</summary>"),
        ("<think>reasoning only</think>", "<summary></summary>"),
        ("   ", "<summary></summary>"),
        ("<THINK>case sensitive</THINK>evidence", "<summary><THINK>case sensitive</THINK>evidence</summary>"),
    ],
)
def test_original_summary_normalization(content, expected):
    assert normalize_summary(content) == expected


@pytest.mark.parametrize(("window",), [(8192,), (65536,), (131072,)])  # noqa: PT006
def test_configured_window_scaling_and_output_cap(tmp_path, window):
    agent = make_agent(tmp_path, "chat", [], context_token_limit=window)
    assert isinstance(agent.config, ReSumConfig)
    assert agent.summary_threshold == 0.9 * (window - 1000)
    assert agent.config.resum_summary_token_limit == 4096
    agent.config.resum_trigger_ratio = 0.8
    agent.config.resum_headroom_tokens = 250
    assert agent.summary_threshold == 0.8 * (window - 250)
    with pytest.raises(ValidationError):
        ReSumConfig(system_template=SYSTEM, instance_template="{{task}}", resum_summary_token_limit=4097)


@pytest.mark.parametrize(("cross_threshold",), [(False,), (True,)])  # noqa: PT006
def test_trigger_counts_complete_history_after_results_before_next_query(tmp_path, api, cross_threshold):
    agent = make_agent(
        tmp_path,
        api,
        [
            output(api, "Find evidence", EVIDENCE_COMMAND),
            *([output(api, "<summary>Evidence</summary>")] if cross_threshold else []),
            output(api, "Next reasoning"),
        ],
    )
    agent.step()
    tokens = agent.context_tokens(agent.context.projection())
    agent.config.context_token_limit = 1000 + math.ceil((tokens + (1 if not cross_threshold else -1)) / 0.9)
    before = list(agent.context.active)
    agent.query()
    assert len(agent.compactions) == int(cross_threshold)
    assert agent.n_calls == 2
    assert agent.cost == 2 + int(cross_threshold)
    if cross_threshold:
        assert agent.context.events[0]["selected_ids"] == before[1:]
        assert agent.context.events[0]["archived"] is False
        assert agent.context.active[0] == before[0]
        assert not set(before[1:]) & set(agent.context.active)
        assert "fresh-evidence-marker" in text(agent.compactions[0]["messages"][0])
        assert SYSTEM not in text(agent.compactions[0]["messages"][0])
    else:
        assert agent.context.active[: len(before)] == before
        assert agent.last_summary is None


def test_repeated_full_resets_roll_summary_without_retaining_latest_exchange(tmp_path, api):
    agent = make_agent(
        tmp_path,
        api,
        [
            output(api, "First search", EVIDENCE_COMMAND, "first"),
            output(api, "<think>hidden</think><summary>First facts</summary>"),
            output(api, "Second search", EVIDENCE_COMMAND, "second"),
            output(api, "<summary>First and second facts</summary>"),
            output(api, "Conclude", "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT; echo done", "final"),
        ],
        context_token_limit=8192,
    )
    assert agent.run(QUESTION)["submission"].strip() == "done"
    assert agent.n_calls == 3 and agent.cost == 5
    assert agent.last_summary == "<summary>First and second facts</summary>"
    assert len(agent.compactions) == len(agent.context.events) == 2
    first, second = agent.resum_summaries
    assert first["last_summary"] is None
    assert second["last_summary"] == first["summary"] == "<summary>First facts</summary>"
    assert first["continuation_id"] in agent.context.events[1]["selected_ids"]
    assert first["continuation_id"] not in second["input_ids"]
    prompt = text(agent.compactions[1]["messages"][0])
    assert "Last Summary\n<summary>First facts</summary>" in prompt
    assert "Second search" in prompt and "First search" not in prompt
    assert "Below is a summary" not in prompt
    assert all(len(manifest["ids"]) == 2 for manifest in agent.context.manifests)
    assert all(not event["archived"] for event in agent.context.events)
    assert not agent.memory_queries and not agent.context.expansions
    serialized = agent.serialize()
    assert serialized["comparison"]["method"] == "resum"
    assert serialized["resum"]["upstream_commit"] == UPSTREAM_COMMIT
    assert serialized["resum"]["training_free"] is True
    assert serialized["resum"]["config"]["context_token_limit"] == 8192
    assert serialized["info"]["model_stats"]["api_calls"] == 5
    assert serialized["info"]["model_stats"]["compaction_api_calls"] == 2
    assert serialized["messages"][1] == agent.model.format_message(role="user", content=QUESTION)
    assert len(serialized["messages"]) > len(agent.context.active)
    assert agent.model.config.context_tools == "none"
    assert agent.model.config.research_tools == "corpus"
    agent.model.config.outputs.append(
        output(api, "Fresh run", "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT; echo fresh", "fresh")
    )
    assert agent.run("A fresh task")["submission"].strip() == "fresh"
    assert agent.last_summary is None and not agent.resum_summaries and not agent.compactions
    assert agent.n_calls == agent.cost == 1
    assert text(agent.messages[1]) == "A fresh task"


@pytest.mark.parametrize(("failure", "charge"), [("empty", 0.4), ("provider", 0.0), ("format", 0.37)])
def test_failed_summary_preserves_live_history_and_previous_summary_with_accounting(tmp_path, api, failure, charge):
    failed = output(api, "")
    failed["extra"]["cost"] = 0.4
    if failure != "empty":
        exc = (
            RuntimeError("Unavailable")
            if failure == "provider"
            else FormatError({"role": "assistant", "content": "bad response", "extra": {"cost": charge}})
        )
        failed["extra"]["actions"] = [{"raise": exc}]
    agent = make_agent(
        tmp_path,
        api,
        [
            output(api, "First", "echo first", "first"),
            output(api, "<summary>Previous facts</summary>"),
            output(api, "Latest", "echo latest", "latest"),
            failed,
        ],
    )
    agent.step()
    assert agent.summarize_history(agent.context.projection())
    agent.step()
    before = copy.deepcopy(agent.context.projection())
    ids = list(agent.context.active)
    trajectory = copy.deepcopy(agent.messages)
    assert not agent.summarize_history(before)
    assert agent.context.projection() == before and agent.context.active == ids
    assert agent.messages == trajectory
    assert agent.last_summary == "<summary>Previous facts</summary>"
    assert len(agent.context.events) == 1
    assert len(agent.compactions) == 2 and not agent.resum_summaries[-1]["success"]
    assert agent.cost == pytest.approx(3 + charge)
    assert agent.n_calls == 2
    if failure != "empty":
        assert agent.compactions[-1]["error"]


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("<think>only reasoning</think>", "<summary></summary>"),
        ("   ", "<summary></summary>"),
        ("<THINK>not stripped upstream</THINK>fact", "<summary><THINK>not stripped upstream</THINK>fact</summary>"),
    ],
)
def test_agent_uses_original_case_sensitive_raw_content_parser(tmp_path, api, content, expected):
    agent = make_agent(
        tmp_path,
        api,
        [output(api, "Gather", "echo evidence"), output(api, content)],
    )
    agent.step()
    assert agent.summarize_history(agent.context.projection())
    assert agent.last_summary == expected
    assert "Summary: " + expected + "\n" in text(agent.context.projection()[1])
    assert agent.cost == 2 and len(agent.context.active) == 2


def test_storage_failure_does_not_reset_context_or_summary(tmp_path, api):
    agent = make_agent(
        tmp_path,
        api,
        [output(api, "Gather", "echo evidence"), output(api, "<summary>Facts</summary>")],
    )
    agent.step()
    before = copy.deepcopy(agent.context.projection())
    ids = list(agent.context.active)
    blocker = tmp_path / "regular-file"
    blocker.write_text("Not a directory")
    agent.context.db_path = blocker / "context.sqlite3"
    assert not agent.summarize_history(before)
    assert agent.context.active == ids and agent.context.projection() == before
    assert agent.last_summary is None and not agent.context.events
    assert agent.cost == 2 and len(agent.compactions) == 1


def test_no_auxiliary_call_after_step_budget_or_final_generation(tmp_path, api):
    agent = make_agent(
        tmp_path,
        api,
        [output(api, "Gather", "echo evidence"), output(api, "Final reasoning")],
    )
    agent.step()
    agent.config.context_token_limit = 8192
    agent.config.resum_headroom_tokens = 8192
    agent.config.step_limit = 1
    with pytest.raises(LimitsExceeded):
        agent.query()
    assert not agent.compactions and not agent.resum_summaries
    agent.config.step_limit = 0
    agent._final_generation = True
    agent.query()
    assert not agent.compactions and agent.n_calls == 2
