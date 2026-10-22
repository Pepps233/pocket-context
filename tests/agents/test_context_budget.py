import copy
from pathlib import Path

import pytest

from pocketcontextagent.agents.default import DefaultAgent
from pocketcontextagent.agents.utils.context_budget import append_text, output_token_setting
from pocketcontextagent.environments.local import LocalEnvironment
from pocketcontextagent.models.test_models import (
    DeterministicModel,
    DeterministicResponseAPIToolcallModel,
    DeterministicToolcallModel,
    make_output,
    make_response_api_output,
    make_toolcall_output,
)


class RecordingModel(DeterministicModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.requests = []

    def query(self, messages, **kwargs):
        self.requests.append((copy.deepcopy(messages), kwargs))
        return super().query(messages, **kwargs)


@pytest.mark.parametrize(("api",), [("text",), ("chat",), ("responses",)])  # noqa: PT006
def test_mark_only_last_result_and_archive_it(tmp_path: Path, api: str):
    actions = [{"command": "printf first", "tool_call_id": "one"}, {"command": "printf second", "tool_call_id": "two"}]
    if api == "text":
        model = DeterministicModel(outputs=[make_output("Run commands", actions)])
    elif api == "chat":
        model = DeterministicToolcallModel(outputs=[make_toolcall_output("Run commands", [], actions)])
    else:
        model = DeterministicResponseAPIToolcallModel(outputs=[make_response_api_output("Run commands", actions)])
    agent = DefaultAgent(
        model,
        LocalEnvironment(),
        system_template="Work",
        instance_template="{{task}}",
        cost_limit=0,
        context_token_limit=128000,
        context_db_path=tmp_path / "context.sqlite",
    )
    agent.add_messages(model.format_message(role="system", content="Work"))
    agent.step()
    field = "output" if api == "responses" else "content"
    assert "CURRENT CONTEXT TOKEN" not in agent.messages[-2][field]
    assert agent.messages[-1][field].startswith("<returncode>0</returncode>\n<output>\nsecond</output>")
    assert agent.messages[-1][field].splitlines()[-1].startswith("[CURRENT CONTEXT TOKEN: ")
    assert agent.context.projection()[-1] == agent.messages[-1]
    assert agent.context.assemble(model)[-1] == agent.messages[-1]


def test_final_generation_reserves_space_and_never_executes_tools(tmp_path: Path):
    model = RecordingModel(
        outputs=[
            make_output("Read results", [{"command": "printf '%06000d' 0"}]),
            make_output("Best answer", [{"command": f"touch {tmp_path / 'unwanted'}"}]),
        ]
    )
    agent = DefaultAgent(
        model,
        LocalEnvironment(),
        system_template="Work",
        instance_template="{{task}}",
        cost_limit=0,
        context_token_limit=5000,
        response_token_limit=3000,
        context_db_path=tmp_path / "context.sqlite",
    )
    assert agent.run("Find evidence")["submission"] == "Best answer"
    assert len(model.requests) == 2
    assert model.requests[1][1]["final_answer"] is True
    assert 0 < model.requests[1][1]["max_tokens"] < 3000
    assert agent.context_tokens(model.requests[1][0]) + model.requests[1][1]["max_tokens"] == 5000
    assert "Stop searching" in model.requests[1][0][-1]["content"]
    assert not (tmp_path / "unwanted").exists()


def test_budget_disabled_and_multimodal_result_annotations(tmp_path: Path):
    model = RecordingModel(outputs=[make_output("Done", [{"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"}])])
    agent = DefaultAgent(
        model,
        LocalEnvironment(),
        system_template="Work",
        instance_template="{{task}}",
        cost_limit=0,
        context_db_path=tmp_path / "context.sqlite",
    )
    assert agent.run("Task")["exit_status"] == "Submitted"
    assert model.requests[0][1] == {}
    message = {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "content": [
                    {"type": "text", "text": "Evidence"},
                    {"type": "image", "source": {"type": "url", "url": "image"}},
                ],
            }
        ],
    }
    append_text(message, "[CURRENT CONTEXT TOKEN: 50]")
    assert message["content"][0]["content"][-1] == {"type": "text", "text": "[CURRENT CONTEXT TOKEN: 50]"}
    assert message["content"][0]["content"][1]["source"]["url"] == "image"
    assert output_token_setting(model, 16384) == ("max_tokens", 16384)
