"""Exercise native memory tools through the production HTTP adapter."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from pocketcontextagent.agents.default import DefaultAgent
from pocketcontextagent.environments.local import LocalEnvironment
from pocketcontextagent.models.litellm_response_model import LitellmResponseModel
from tests.agents.test_compact_loop import native_turn


@pytest.fixture
def responses_server():
    requests, outputs = [], []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            output = outputs.pop(0)
            body = json.dumps(
                output
                if isinstance(output, dict)
                else {
                    "id": f"resp_{len(requests)}",
                    "object": "response",
                    "created_at": 1,
                    "model": "gpt-4.1",
                    "status": "completed",
                    "output": output,
                    "usage": {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20},
                }
            ).encode()
            self.send_response(401 if isinstance(output, dict) else 200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", requests, outputs
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize(("fail_summary",), [(False,), (True,)])  # noqa: PT006
def test_production_responses_compaction_and_expansion(tmp_path, responses_server, fail_summary):
    url, requests, outputs = responses_server
    for name, arguments in [
        ("bash", {"command": "printf 'archive-output'; printf '%03000d' 0"}),
        ("bash", {"command": "echo retained"}),
        ("compact_block", {"start_id": "b001", "end_id": "b001"}),
        (None, {}),
        ("expand_block", {"tag": "ctx-0001"}),
        ("bash", {"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"}),
    ]:
        outputs.append(
            native_turn(name, arguments, f"call_{len(outputs)}", "Inspect archive-start", True)["output"]
            if name
            else [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Stored evidence"}],
                }
            ]
        )
    # Only the first response should contain the starting anchor.
    for output in outputs[1:]:
        output[0]["content"][0]["text"] = (
            "Continue" if output[0]["content"][0]["text"] != "Stored evidence" else "Stored evidence"
        )
    if fail_summary:
        outputs[3] = {"error": {"message": "Child request denied", "type": "authentication_error"}}
        del outputs[4]
    agent = DefaultAgent(
        LitellmResponseModel(
            model_name="openai/gpt-4.1",
            model_kwargs={"api_base": url, "api_key": "local-test"},
            cost_tracking="ignore_errors",
        ),
        LocalEnvironment(),
        context_db_path=tmp_path / "context.sqlite",
        cost_limit=0,
        system_template="System",
        instance_template="{{task}}",
    )
    assert agent.run("Keep working")["exit_status"] == "Submitted"
    assert agent.context.revision == int(not fail_summary)
    assert len(agent.context.expansions) == int(not fail_summary)
    if fail_summary:
        assert "AuthenticationError" in agent.compactions[0]["error"]
        assert "history preserved. Continue the task" in json.dumps(requests[-1])
        assert "archive-output" + "0" * 3000 in json.dumps(requests[-1])
    assert requests[3]["tool_choice"] == "none"
    for request in requests:
        for item in request["input"]:
            if item.get("role") in ("user", "system") and isinstance(item.get("content"), list):
                assert all(block["type"] in ("input_text", "input_image") for block in item["content"]), item


def test_production_retry_sequence_uses_stable_ids(tmp_path, responses_server):
    url, requests, outputs = responses_server
    turns = [
        ("bash", {"command": "printf 'repeated'; printf '%03000d' 0"}),
        ("bash", {"command": "printf 'repeated'; printf '%03000d' 0"}),
        ("bash", {"command": "echo retained"}),
        ("compact_block", {"start_quote": "repeated0000", "end_quote": "repeated0000"}),
        ("compact_block", {"start_id": "b001", "end_id": "b002"}),
        (None, {}),
        ("compact_block", {"start_id": "b001", "end_id": "b002"}),
        (None, {}),
        ("expand_block", {"tag": "ctx-0001"}),
        ("bash", {"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"}),
    ]
    for index, (name, args) in enumerate(turns):
        outputs.append(native_turn(name, args, f"call_{index}", f"Turn {index}", True)["output"] if name else [])
    outputs[7] = [
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Both commands printed repeated followed by zeros."}],
        }
    ]
    agent = DefaultAgent(
        LitellmResponseModel(
            model_name="openai/gpt-4.1",
            model_kwargs={"api_base": url, "api_key": "local-test"},
            cost_tracking="ignore_errors",
        ),
        LocalEnvironment(),
        system_template="System",
        instance_template="{{task}}",
        cost_limit=0,
        context_db_path=tmp_path / "context.sqlite",
    )
    assert agent.run("Continue until done")["exit_status"] == "Submitted"
    assert len(requests) == 10 and len(agent.compactions) == 2 and agent.context.revision == 1
    assert "matches 2 messages. Use start_id/end_id. Eligible IDs: b001, b002" in json.dumps(requests[4])
    assert "history preserved. Continue the task" in json.dumps(requests[6])
    assert "b001: 2 messages" in json.dumps(requests[6])
    assert "b001: 2 messages" not in json.dumps(requests[8])
    for index in (5, 7):
        assert requests[index]["tool_choice"] == "none"
        assert "retained" not in json.dumps(requests[index]["input"])
        assert all(block["type"] == "input_text" for block in requests[index]["input"][1]["content"])
    assert len(agent.context.expansions) == 1
