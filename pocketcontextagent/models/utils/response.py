"""Inspect assistant text without treating reasoning or tool arguments as an answer."""

import re
from typing import Any


def response_items(message: dict) -> list[dict]:
    return message.get("output", []) if message.get("object") == "response" else [message]


def has_tool_calls(response: Any, *, action_regex: str = "") -> bool:
    response = response if isinstance(response, dict) else response.model_dump()
    message = response["choices"][0]["message"] if "choices" in response else response
    return bool(action_regex and re.search(action_regex, response_text(message), re.DOTALL)) or any(
        item.get("tool_calls")
        or item.get("type") == "function_call"
        or any(block.get("type") == "tool_use" for block in item.get("content") or [] if isinstance(block, dict))
        for item in response_items(message)
    )


def response_text(message: dict) -> str:
    text = "\n".join(
        block.get("text", "")
        for item in response_items(message)
        if item.get("role") == "assistant" or item.get("type") == "message"
        for block in (
            [{"type": "text", "text": item["content"]}]
            if isinstance(item.get("content"), str)
            else item.get("content") or []
        )
        if block.get("type") in ("text", "output_text")
    )
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
