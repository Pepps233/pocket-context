from pathlib import Path

import pytest

from pocketcontextagent.agents.default import DefaultAgent
from pocketcontextagent.environments.local import LocalEnvironment
from pocketcontextagent.models.litellm_model import LitellmModel
from pocketcontextagent.models.litellm_response_model import LitellmResponseModel
from pocketcontextagent.models.utils.content_string import get_content_string
from tests.agents.test_compact_loop import (
    RecordingModel,
    RecordingResponseModel,
    make_agent,
    native_turn,
    start_history,
)
from tests.agents.test_compact_loop import api as api
from tests.agents.test_expand_loop import next_turn


class PressureChatModel(RecordingModel):
    def get_tools(self) -> list[dict]:
        return LitellmModel(model_name="test").get_tools()


class PressureResponseModel(RecordingResponseModel):
    def get_tools(self) -> list[dict]:
        return LitellmResponseModel(model_name="test").get_tools()


def tool_names(options: dict) -> list[str]:
    return [tool.get("function", tool)["name"] for tool in options["tools"]]


def test_pressure_gates_tools_and_catalog_and_rearms_after_compaction(tmp_path: Path, api: bool):
    agent = make_agent(
        tmp_path, api, context_token_limit=32000, response_token_limit=4000, compact_pressure_fraction=0.5
    )
    agent.model = (PressureResponseModel if api else PressureChatModel)(**agent.model.config.model_dump())
    agent.model.config.outputs[0] = native_turn(
        "bash", {"command": "printf 'archive-output'; printf '%060000d' 0"}, "a", "Inspect archive-start", api
    )
    start_history(agent)
    agent.step()
    assert tool_names(agent.model.query_options[0]) == ["bash", "expand_block"]
    assert "Eligible context blocks" not in get_content_string(agent.model.requests[0][-1])
    assert "compact_block" in tool_names(agent.model.query_options[2])
    assert "Eligible context blocks" in get_content_string(agent.model.requests[2][-1])
    assert agent.context.revision == len(agent.compactions) == 1
    next_turn(agent, "bash", {"command": "echo keep solving"}, "continue", api)
    assert tool_names(agent.model.query_options[4]) == ["bash", "expand_block"]
    assert "Eligible context blocks" not in get_content_string(agent.model.requests[4][-1])
    assert get_content_string(agent.model.requests[4][-1]).endswith("[Available archives for expand_block]\nctx-0001")
    next_turn(agent, "compact_block", {"start_id": "b002", "end_id": "b002"}, "hidden", api)
    assert "Compaction unavailable" in get_content_string(agent.messages[-1])
    assert agent.context.revision == len(agent.compactions) == 1
    next_turn(agent, "expand_block", {"tag": "ctx-0001"}, "expand", api)
    assert len(agent.context.expansions) == 1
    next_turn(agent, "bash", {"command": "echo review evidence"}, "review", api)
    assert "compact_block" in tool_names(agent.model.query_options[-1])
    assert "Eligible context blocks" in get_content_string(agent.model.requests[-1][-1])
    assert agent.context.request_projection(agent.model) == agent.context.assemble(agent.model)
    assert [manifest["compaction_available"] for manifest in agent.context.manifests[:4]] == [False, True, True, False]


@pytest.mark.parametrize(("input_tokens", "available"), [(23807, False), (23808, True)])
def test_pressure_uses_half_the_input_budget_without_output_or_catalog(
    tmp_path: Path, api: bool, input_tokens, available
):
    model = (PressureResponseModel if api else PressureChatModel)(
        outputs=[native_turn("bash", {"command": "echo evidence"}, "one", "Investigate", api)]
    )
    agent = DefaultAgent(
        model,
        LocalEnvironment(),
        context_token_limit=64000,
        context_db_path=tmp_path / "context.sqlite",
        system_template="Work",
        instance_template="{{task}}",
        cost_limit=0,
    )
    initial_input_tokens = agent.context_tokens(
        [model.format_message(role="system", content="Work"), model.format_message(role="user", content="xxxx")],
        compact=False,
    )
    agent.add_messages(
        model.format_message(role="system", content="Work"),
        model.format_message(role="user", content="xxxx" + "x" * (4 * (input_tokens - initial_input_tokens))),
    )
    agent.step()
    assert ("compact_block" in tool_names(model.query_options[0])) == available
    assert agent.context.manifests[0]["compaction_available"] == available
