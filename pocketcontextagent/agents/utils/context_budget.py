"""Offline context estimates and provider-neutral tool-result annotations."""

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


def estimate_tokens(messages: list[dict], tokenizer=None, *, text_based: bool = False) -> int:
    """Count serialized request tokens, falling back to four UTF-8 bytes per token.

    The corpus tokenizer is used when configured; schemas and message framing are included.
    This is an estimate rather than the provider's exact chat-template token count.
    """
    text = encode({"messages": messages, "tools": [] if text_based else [BASH_TOOL, COMPACT_TOOL, EXPAND_TOOL]})
    return (
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
            content.append({"type": "input_text" if field == "output" else "text", "text": text})
