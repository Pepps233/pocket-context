"""Offline context estimates and provider-neutral tool-result annotations."""

from collections.abc import Callable, Iterator
from math import ceil

from pocketcontextagent.agents.context import encode
from pocketcontextagent.models.utils.actions_toolcall import BASH_TOOL
from pocketcontextagent.models.utils.compact_tool import COMPACT_TOOL, EXPAND_TOOL


def output_token_setting(model, default: int) -> tuple[str, int]:
    kwargs = getattr(model.config, "model_kwargs", {})
    for key in ("max_completion_tokens", "max_output_tokens", "max_tokens"):
        if kwargs.get(key) is not None:
            return key, int(kwargs[key])
    responses = any("response_model" in cls.__module__ for cls in type(model).__mro__)
    return "max_output_tokens" if responses else "max_tokens", default


def retained_reasoning_tokens(messages: list[dict]) -> int:
    tokens = 0
    for message in messages:
        if message.get("object") != "response" or not any(
            item.get("type") == "reasoning" and item.get("encrypted_content") for item in message.get("output", [])
        ):
            continue
        usage = message.get("usage") or {}
        reasoning = (usage.get("output_tokens_details") or {}).get("reasoning_tokens")
        tokens += (usage.get("output_tokens", 0) if reasoning is None else reasoning) or 0
    return tokens


def _without_encrypted_reasoning(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: _without_encrypted_reasoning(child)
            for key, child in value.items()
            if key != "encrypted_content" or value.get("type") != "reasoning"
        }
    if isinstance(value, list):
        return [_without_encrypted_reasoning(child) for child in value]
    return value


def estimate_tokens(
    messages: list[dict],
    tokenizer=None,
    *,
    text_based: bool = False,
    reasoning_tokens: int = 0,
    tools: list[dict] | None = None,
) -> int:
    """Count serialized request tokens, falling back to four UTF-8 bytes per token.

    The corpus tokenizer is used when configured; schemas and message framing are included.
    Pass tools to count the actual exposed schemas; None preserves the Pocket Context default.
    Encrypted reasoning is excluded from text counts; retained reasoning uses reported usage instead.
    This is an estimate rather than the provider's exact chat-template token count.
    """
    if tools is None:
        tools = [] if text_based else [BASH_TOOL, COMPACT_TOOL, EXPAND_TOOL]
    text = encode(
        {
            "messages": _without_encrypted_reasoning(messages),
            "tools": tools,
        }
    )
    return reasoning_tokens + (
        len(tokenizer.encode(text, add_special_tokens=False)) if tokenizer is not None else ceil(len(text.encode()) / 4)
    )


def append_text(message: dict, text: str) -> None:
    field = "output" if message.get("type") == "function_call_output" else "content"
    content = message.get(field, "")
    if isinstance(content, str):
        message[field] = content + "\n" + text
    elif isinstance(content, list):
        if content and content[-1].get("type") == "tool_result":
            append_text(content[-1], text)
        else:
            content.append(
                {
                    "type": "input_text" if field == "output" or message.get("type") == "message" else "text",
                    "text": text,
                }
            )


def text_fields(message: dict) -> Iterator[tuple[dict, str]]:
    field = "output" if message.get("type") == "function_call_output" else "content"
    content = message.get(field)
    if isinstance(content, str):
        yield message, field
    elif isinstance(content, list):
        for block in content:
            if block.get("type") in ("text", "input_text", "output_text"):
                yield block, "text"
            elif block.get("type") == "tool_result":
                yield from text_fields(block)


def fit_observations(messages: list[dict], budget: int, count_tokens: Callable[[list[dict]], int]) -> None:
    """Bound new visible results before ingestion, retaining full raw_output metadata."""
    suffix = "\n[Output truncated to leave room for a final answer.]"
    fields = [(message, container, key) for message in messages for container, key in text_fields(message)]
    for message, container, key in sorted(fields, key=lambda field: len(field[1][field[2]]), reverse=True):
        if count_tokens(messages) <= budget:
            return
        original = container[key]
        if len(original) <= len(suffix):
            continue
        low, high = 0, len(original)
        container[key] = suffix
        message.setdefault("extra", {})["context_truncated"] = True
        if count_tokens(messages) > budget:
            continue
        while low < high:
            middle = (low + high + 1) // 2
            container[key] = original[:middle] + suffix
            if count_tokens(messages) <= budget:
                low = middle
            else:
                high = middle - 1
        container[key] = original[:low] + suffix
