import pytest

from pocketcontextagent.models.utils.response import has_tool_calls, response_text


@pytest.mark.parametrize(
    ("message", "tools"),
    [
        (
            {
                "role": "assistant",
                "content": "<think>Exact Answer: Wrong</think>\nExact Answer: Right",
                "reasoning_content": "Wrong",
            },
            False,
        ),
        (
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Wrong"},
                    {"type": "text", "text": "Exact Answer: Right"},
                    {"type": "tool_use", "input": {"command": "Wrong"}},
                ],
            },
            True,
        ),
        (
            {
                "object": "response",
                "output": [
                    {"type": "reasoning", "summary": [{"type": "summary_text", "text": "Wrong"}]},
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Exact Answer: Right"}],
                    },
                    {"type": "function_call", "arguments": "Wrong"},
                ],
            },
            True,
        ),
        (
            {
                "role": "assistant",
                "content": "Exact Answer: Right",
                "tool_calls": [{"function": {"arguments": "Wrong"}}],
            },
            True,
        ),
    ],
)
def test_extract_only_assistant_text(message: dict, tools: bool):
    assert response_text(message) == "Exact Answer: Right"
    assert has_tool_calls(message) == tools
