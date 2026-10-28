import copy

import pytest
from pydantic import ValidationError

from pocketcontextagent.agents.context import ContextOperationError, SelectionError
from pocketcontextagent.comparisons.methods.acm_prompts import build_system_prompt as published_prompt
from pocketcontextagent.comparisons.methods.base import ComparisonConfig, ComparisonContext, conversation_text
from pocketcontextagent.comparisons.methods.react import ReActAgent, build_system_prompt
from pocketcontextagent.environments.local import LocalEnvironment
from pocketcontextagent.models.test_models import DeterministicToolcallModel


def test_react_prompt_and_rendered_history_preserve_original_policy_and_native_calls():
    assert build_system_prompt(131072) == published_prompt(131072, use_memory_tools=False)
    messages = [
        {
            "role": "assistant",
            "content": None,
            "reasoning_content": "Find the festival",
            "tool_calls": [{"id": "search", "function": {"name": "search", "arguments": '{"query":"jazz"}'}}],
            "extra": {"private_key": "NOT_MODEL_VISIBLE"},
        },
        {"role": "tool", "tool_call_id": "search", "content": "Montréal [city]"},
    ]
    before = copy.deepcopy(messages)
    rendered = conversation_text(messages)
    assert all(text in rendered for text in ("search", "jazz", "Find the festival", "Montréal", "[city]"))
    assert "NOT_MODEL_VISIBLE" not in rendered and messages == before


def test_comparison_deletion_rejects_partial_exchanges_and_archives_are_run_scoped(tmp_path):
    context = ComparisonContext(tmp_path / "archive.sqlite")
    context.ingest({"role": "system", "content": "System"})
    context.ingest({"role": "user", "content": "Question"})
    call = context.ingest(
        {"role": "assistant", "tool_calls": [{"id": "call", "function": {"name": "search", "arguments": "{}"}}]}
    )
    result = context.ingest({"role": "tool", "tool_call_id": "call", "content": "Evidence"})
    before = context.projection()
    with pytest.raises(SelectionError, match="split"):
        context.remove([call], operation="manage_context", summary_id=1, archive=True)
    assert context.projection() == before and not context.events
    context.remove([call, result], operation="manage_context", summary_id=1, archive=True)
    assert context.archive(1) == before[2:]
    assert context.projection() == before[:2]
    with pytest.raises(SelectionError, match="not found"):
        ComparisonContext(context.db_path).archive(1)


def test_unknown_failed_auxiliary_cost_is_not_zero_and_pressure_is_not_tunable(tmp_path):
    agent = ReActAgent(
        DeterministicToolcallModel(outputs=[]),
        LocalEnvironment(),
        system_template="System",
        instance_template="{{task}}",
        cost_limit=0,
        context_db_path=tmp_path / "archive.sqlite",
    )
    with pytest.raises(ContextOperationError, match="Auxiliary call failed"):
        agent.call_auxiliary("Prompt")
    stats = agent.serialize()["info"]["model_stats"]
    assert stats["api_calls"] == 1 and stats["cost_available"] is False
    with pytest.raises(ValidationError):
        ComparisonConfig(system_template="System", instance_template="Question", context_pressure_fraction=0.1)
