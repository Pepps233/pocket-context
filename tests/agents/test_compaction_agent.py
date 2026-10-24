from pathlib import Path

import pytest

from pocketcontextagent.agents.context import encode
from pocketcontextagent.agents.default import DefaultAgent
from pocketcontextagent.environments.local import LocalEnvironment
from pocketcontextagent.models.test_models import make_output, make_response_api_output
from tests.agents.test_compact_loop import (
    RecordingModel,
    RecordingResponseModel,
    compaction_args,
    make_agent,
    native_turn,
    read_blocks,
    start_history,
)
from tests.agents.test_compact_loop import api as api


def test_compaction_agent_sees_prefix_without_future(tmp_path: Path, api):
    summary = "B printed selected-B; C printed selected-C. Both commands succeeded."
    model = (RecordingResponseModel if api else RecordingModel)(
        outputs=[
            native_turn("bash", {"command": "echo earlier-A"}, "a", "Earlier A", api),
            native_turn("bash", {"command": "printf 'selected-B'; printf '%03000d' 0"}, "b", "BEGIN SELECTED B", api),
            native_turn("bash", {"command": "printf 'selected-C'; printf '%03000d' 1"}, "c", "Selected C", api),
            native_turn("bash", {"command": "echo FUTURE-D-SECRET"}, "d", "Future D", api),
            native_turn(
                "compact_block",
                {"start_quote": "BEGIN SELECTED B", "end_quote": "selected-C0000"},
                "compact",
                "FUTURE-COMPACTION-SECRET",
                api,
            ),
            make_response_api_output(summary, []) if api else make_output(summary, [], cost=0.25),
            native_turn("bash", {"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"}, "done", "Finish", api),
        ]
    )
    model.config.outputs[0]["extra"]["provider_metadata"] = "HIDDEN-FUTURE-SECRET"
    agent = DefaultAgent(
        model,
        LocalEnvironment(),
        system_template="Original system A",
        instance_template="{{task}}",
        cost_limit=0,
        context_db_path=tmp_path / "context.sqlite",
    )
    assert agent.run("Original question A")["exit_status"] == "Submitted"
    assert agent.context.revision == 1
    assert len(model.requests) == 7
    child_request = encode(model.requests[5])
    assert all(
        text in child_request
        for text in (
            "Original system A",
            "Original question A",
            "earlier-A",
            "BEGIN SELECTED B",
            "selected-B0000",
            "selected-C0000",
        )
    )
    assert "FUTURE-D-SECRET" not in child_request and "Future D" not in child_request
    assert "FUTURE-COMPACTION-SECRET" not in child_request
    assert "HIDDEN-FUTURE-SECRET" not in child_request
    assert agent.n_calls == 6
    assert agent.cost == (1 if api else 0.25)
    rebuilt = encode(model.requests[-1])
    assert summary in rebuilt and "FUTURE-D-SECRET" in rebuilt and "earlier-A" in rebuilt
    assert "0" * 2000 not in rebuilt
    assert read_blocks(agent)[0]["summary"] == summary
    assert len(agent.compactions) == 1
    assert agent.serialize()["info"]["model_stats"]["api_calls"] == 7


def test_recompaction_excludes_management_turns_but_keeps_evidence(tmp_path: Path, api):
    agent = make_agent(
        tmp_path,
        api,
        outputs=[
            native_turn(
                "bash", {"command": "printf 'archive-output'; printf '%06000d' 0"}, "a", "Inspect archive-start", api
            ),
            native_turn("bash", {"command": "echo retained-output"}, "b", "Keep working", api),
            native_turn(
                "compact_block",
                compaction_args(summary="Remember original evidence " + "x" * 2100),
                "compact",
                "Manage context",
                api,
            ),
            native_turn("expand_block", {"tag": "ctx-0001"}, "expand", "Retrieve archived evidence", api),
            native_turn("bash", {"command": "echo latest"}, "latest", "Continue research", api),
            native_turn(
                "compact_block",
                {"start_quote": "[ctx-0001]: Remember", "end_quote": "[ctx-0001]: (exp=1 part=2/2"},
                "recompact",
                "Combine evidence",
                api,
            ),
            native_turn("bash", {"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"}, "done", "Finish", api),
        ],
    )
    assert agent.run("Find evidence")["exit_status"] == "Submitted"
    assert agent.context.revision == 2 and len(agent.context.expansions) == 1
    request = encode(agent.compactions[1]["messages"][1:])
    assert "Remember original evidence" in request and "archive-output" + "0" * 6000 in request
    assert "retained-output" in request and "Continue research" not in request
    assert all(text not in request for text in ("Manage context", "Compacted [", "Expanded [", "Tool expand_block"))
    blocks = read_blocks(agent)
    assert blocks[1]["children"] == ["ctx-0001"]
    assert "Compacted [ctx-0001]" in encode(blocks[1]["messages"])
    assert "archive-output" + "0" * 6000 in encode(blocks[0]["messages"])


@pytest.mark.parametrize(("failure",), [("empty",), ("tools",), ("truncated",), ("oversized",)])  # noqa: PT006
def test_failed_summary_keeps_context_and_accounts_for_call(tmp_path: Path, api, failure: str):
    agent = make_agent(tmp_path, api)
    marker = tmp_path / "must-not-exist"
    if failure == "tools":
        response = native_turn("bash", {"command": f"touch {marker}"}, "child", "Do not execute this", api)
    else:
        text = "" if failure == "empty" else "x" * 5000 if failure == "oversized" else "Incomplete summary"
        response = make_response_api_output(text, []) if api else make_output(text, [], cost=0.25)
        if failure == "truncated":
            response["extra"]["response"] = {"choices": [{"finish_reason": "length"}]}
    agent.model.config.outputs[3] = response
    start_history(agent)
    before = list(agent.context.active)
    assert agent.step()[0]["extra"]["returncode"] == -1
    assert agent.context.active[: len(before)] == before
    assert agent.context.revision == 0 and not read_blocks(agent)
    assert len(agent.compactions) == 1 and agent.compactions[0]["response"] == response
    assert agent.cost == response["extra"]["cost"]
    assert agent.serialize()["info"]["model_stats"]["compaction_api_calls"] == 1
    assert not marker.exists()


def test_invalid_selection_and_committed_replay_do_not_summarize(tmp_path: Path, api):
    agent = make_agent(tmp_path, api, args=compaction_args(start_quote="not in history"))
    start_history(agent)
    assert agent.step()[0]["extra"]["returncode"] == -1
    assert not agent.compactions and len(agent.model.requests) == 3
    agent = make_agent(tmp_path / "valid", api)
    start_history(agent)
    response = agent.query()
    result = agent.execute_actions(response)
    assert agent.execute_actions(response) == result
    assert len(agent.compactions) == 1 and len(agent.model.requests) == 4
    assert read_blocks(agent)[0]["summary_input_ids"] == ["m000001", "m000002", "m000003", "m000004"]
    assert len(agent.serialize()["compactions"]) == 1


def test_compaction_call_respects_cost_budget_without_consuming_main_steps(tmp_path: Path, api):
    agent = make_agent(tmp_path, api, step_limit=3)
    start_history(agent)
    assert "Compacted" in encode(agent.step())
    assert agent.n_calls == 3 and len(agent.compactions) == 1
    agent = make_agent(tmp_path / "cost", api)
    for output in agent.model.config.outputs:
        output["extra"]["cost"] = 1
    agent.config.cost_limit = 3
    assert agent.run("Keep working")["exit_status"] == "LimitsExceeded"
    assert agent.n_calls == 3 and len(agent.model.requests) == 3
    assert not agent.compactions and agent.context.revision == 0


def test_internal_child_formatting_failure_does_not_request_boundary_repair(tmp_path, api):
    agent = make_agent(tmp_path, api)
    first = agent.model.config.outputs[0]["output"][0] if api else agent.model.config.outputs[0]
    first["content"] = [
        {"type": "output_text" if api else "text", "text": "Inspect archive-start"},
        {"type": "image", "source": {}},
    ]
    del agent.model.config.outputs[3]
    assert agent.run("Keep working")["exit_status"] == "Submitted"
    assert agent.context.revision == 0 and not read_blocks(agent)
    assert "KeyError" in agent.compactions[0]["error"] and not agent.compactions[0]["response"]
    assert "history preserved. Continue the task" in encode(agent.model.requests[-1])
    assert "repair" not in encode(agent.model.requests[-1])
    assert "archive-output" + "0" * 3000 in encode(agent.model.requests[-1])
    assert len(agent.model.requests) == 4
