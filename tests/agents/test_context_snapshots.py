import json
import sqlite3

from pocketcontextagent.models.utils.content_string import get_content_string
from tests.agents.test_compact_loop import api as api
from tests.agents.test_compact_loop import compaction_args, make_agent, native_turn, read_blocks
from tests.agents.test_expand_loop import compacted_agent, next_turn


def test_saved_events_match_committed_snapshots_after_continued_work(tmp_path, api):
    agent = make_agent(
        tmp_path,
        api,
        outputs=[
            native_turn(
                "bash", {"command": "printf 'archive-output'; printf '%09000d' 0"}, "a", "Inspect archive-start", api
            ),
            native_turn("bash", {"command": "echo retained"}, "b", "Retain", api),
            native_turn(
                "compact_block", compaction_args(summary="Detailed evidence " + "x" * 2100), "first", "Compact", api
            ),
            native_turn("bash", {"command": "echo continue"}, "c", "Continue", api),
            native_turn(
                "compact_block", {"start_quote": "[ctx-0001]:", "end_quote": "[ctx-0001]:"}, "second", "Recompact", api
            ),
            native_turn("expand_block", {"tag": "ctx-0001"}, "expand", "Retrieve original evidence", api),
            native_turn("bash", {"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"}, "done", "Finish", api),
        ],
        output_path=tmp_path / "trajectory.json",
    )
    assert agent.run("Keep working")["exit_status"] == "Submitted"
    with sqlite3.connect(agent.context.db_path) as db:
        events = [json.loads(row[0]) for row in db.execute("SELECT record FROM events ORDER BY revision")]
        manifests = [json.loads(row[0]) for row in db.execute("SELECT record FROM requests ORDER BY sequence")]
    assert len(events) == 2 and len(agent.context.expansions) == 1
    for event in events:
        start = event["active_before"].index(event["selected_ids"][0])
        assert event["active_after"] == (
            event["active_before"][:start]
            + [event["summary_id"]]
            + event["active_before"][start + len(event["selected_ids"]) :]
        )
    assert agent.context.events == events
    assert json.loads(agent.config.output_path.read_text())["context"]["events"] == events
    assert agent.context.serialize()["manifests"] == manifests
    assert agent.context.projection()


def test_compaction_replay_does_not_accumulate_token_markers(tmp_path, api):
    agent = compacted_agent(tmp_path, api, context_token_limit=100000, response_token_limit=100)
    for number in range(2):
        receipt = next_turn(agent, "compact_block", compaction_args(), "compact", api)[0]
        assert get_content_string(receipt).count("[CURRENT CONTEXT TOKEN:") == 1
        if number == 0:
            assert json.loads(json.dumps(agent.context._applied["compact"])) == read_blocks(agent)[0]
            agent.context._applied.clear()
    assert agent.context.revision == len(agent.compactions) == 1


def test_expansion_journal_is_not_changed_by_observation_annotations(tmp_path, api):
    agent = compacted_agent(tmp_path, api, context_token_limit=100000, response_token_limit=100)
    results = next_turn(agent, "expand_block", {"tag": "ctx-0001"}, "expand", api)
    assert get_content_string(results[0]).count("[CURRENT CONTEXT TOKEN:") == 1
    with sqlite3.connect(agent.context.db_path) as db:
        operation = json.loads(db.execute("SELECT record FROM operations").fetchone()[0])
    assert agent.context.expansions == [operation]
    assert len(results) == 3
    assert len(next_turn(agent, "expand_block", {"tag": "ctx-0001"}, "expand", api)) == 1
    assert agent.context.expansions == [operation]


def test_budgeted_compaction_reports_the_actual_rebuilt_request_reduction(tmp_path, api):
    agent = make_agent(tmp_path, api, context_token_limit=100000, response_token_limit=100)
    assert agent.run("Keep working")["exit_status"] == "Submitted"
    block = read_blocks(agent)[0]
    before, after = agent.context.manifests[block["request_sequence"] : block["request_sequence"] + 2]
    assert block["reduction"] == before["characters"] - after["characters"]
    assert json.loads(json.dumps(agent.context._applied["compact"])) == block
    rejected = make_agent(
        tmp_path / "threshold",
        api,
        context_token_limit=100000,
        response_token_limit=100,
        compact_min_reduction=block["reduction"] + 1,
    )
    assert rejected.run("Keep working")["exit_status"] == "Submitted"
    assert rejected.context.revision == 0 and not read_blocks(rejected)
    assert "Net request reduction" in get_content_string(rejected.messages[-3])


def test_saved_checkpoint_stays_frozen_while_agent_recompacts(tmp_path, api):
    agent = compacted_agent(
        tmp_path, api, archive_chars=9000, args=compaction_args(summary="Detailed evidence " + "x" * 2100)
    )
    checkpoint = agent.save(tmp_path / "checkpoint.json")
    saved = json.loads((tmp_path / "checkpoint.json").read_text())
    next_turn(agent, "bash", {"command": "echo continued"}, "continue", api)
    next_turn(agent, "compact_block", {"start_quote": "[ctx-0001]:", "end_quote": "[ctx-0001]:"}, "second", api)
    assert agent.context.revision == 2 and len(agent.compactions) == 2
    assert checkpoint == saved
    live = agent.serialize()
    expected = json.loads(json.dumps(live))
    live["messages"][2]["extra"]["actions"].clear()
    live["compactions"][0]["messages"].clear()
    live["context"]["blocks"]["ctx-0002"].clear()
    assert agent.serialize() == expected
    assert read_blocks(agent)[1]["children"] == ["ctx-0001"]


def test_context_export_does_not_expose_committed_lineage(tmp_path, api):
    agent = compacted_agent(tmp_path, api)
    snapshot = agent.context.serialize()
    expected = json.loads(json.dumps(snapshot))
    snapshot["blocks"]["ctx-0001"].append("ctx-9999")
    snapshot["events"][0]["active_after"].clear()
    snapshot["manifests"][0]["ids"].clear()
    assert agent.context.serialize() == expected
    assert json.loads(json.dumps(agent.context._applied["compact"])) == read_blocks(agent)[0]
