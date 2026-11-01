"""Provider-neutral input and adapters for a compression agent."""

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol

from pocket_context.prompts import COMPRESSION_SYSTEM_PROMPT


@dataclass(frozen=True)
class CompressionRequest:
    history: list[dict]
    start: int
    end: int
    system_prompt: str
    user_prompt: str

    @property
    def messages(self) -> list[dict]:
        """Return standard role/content messages for a model adapter."""
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_prompt},
        ]


class CompressionRunner(Protocol):
    def run(self, request: CompressionRequest) -> str | Awaitable[str]: ...


CompressionAgent = Callable[[CompressionRequest], str | Awaitable[str]] | CompressionRunner
PromptBuilder = Callable[[list[dict], int], str]


def default_user_prompt(history: list[dict], start: int) -> str:
    """Mark selected messages within the bounded history prefix."""
    parts = [
        f"Summarize inclusive history messages {start + 1} through {len(history)}. "
        "The history ends at the selected right bound."
    ]
    for index, message in enumerate(history):
        label = "selected" if index >= start else "earlier context"
        parts.append(f"History message {index + 1} ({label}):\n{json.dumps(message, ensure_ascii=False)}")
    return "\n\n".join(parts)


def compression_request(
    history: list[dict],
    start: int,
    *,
    system_prompt: str = COMPRESSION_SYSTEM_PROMPT,
    user_prompt_builder: PromptBuilder = default_user_prompt,
) -> CompressionRequest:
    """Build an agent-ready request with customizable prompts."""
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise ValueError("system_prompt must be a non-empty string")
    user_prompt = user_prompt_builder(history, start)
    if not isinstance(user_prompt, str) or not user_prompt.strip():
        raise ValueError("user_prompt_builder must return a non-empty string")
    return CompressionRequest(history, start, len(history) - 1, system_prompt, user_prompt)


def run_compressor(agent: CompressionAgent, request: CompressionRequest) -> str | Awaitable[str]:
    """Call either a function adapter or an object with run(request)."""
    if callable(run := getattr(agent, "run", None)):
        return run(request)
    if callable(agent):
        return agent(request)
    raise TypeError("agent must be callable or provide run(request)")
