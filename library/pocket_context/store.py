"""A small chronological message store with reversible compaction."""

import json
import sqlite3
from copy import deepcopy
from inspect import isawaitable, iscoroutine
from pathlib import Path
from uuid import uuid4

from pocket_context.compression import (
    CompressionAgent,
    CompressionRequest,
    PromptBuilder,
    compression_request,
    default_user_prompt,
    run_compressor,
)
from pocket_context.prompts import COMPRESSION_SYSTEM_PROMPT
from pocket_context.tools import COMPACT_TOOL, EXPAND_TOOL


class ContextStore:
    """Keep an active projection in memory and archived blocks in SQLite.

    Messages are provider-shaped dictionaries owned by the calling agent.
    Compaction calls a supplied agent adapter, so this module has no model
    or provider dependencies.
    """

    def __init__(self, db_path: str | Path = ":memory:", *, compact_at_percent: float = 50.0):
        if not 0 <= compact_at_percent <= 100:
            raise ValueError("compact_at_percent must be between 0 and 100")
        self.db_path = str(db_path)
        self.compact_at_percent = compact_at_percent
        self._db = sqlite3.connect(self.db_path)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS blocks (tag TEXT PRIMARY KEY, messages TEXT NOT NULL)"
        )
        self._db.commit()
        self._messages: dict[str, dict] = {}
        self._active: list[str] = []
        self._protected: set[str] = set()
        self._archive_tags: set[str] = set()
        self._next_id = 0
        self._next_tag = 0

    def record(self, message: dict, *, protected: bool = False) -> str:
        """Append a message and return its stable ID."""
        self._next_id += 1
        message_id = f"m{self._next_id:06d}"
        self._messages[message_id] = json.loads(json.dumps(message))
        self._active.append(message_id)
        if protected or message.get("role") in ("system", "developer"):
            self._protected.add(message_id)
        return message_id

    def _request(
        self,
        start: int,
        end: int,
        system_prompt: str,
        user_prompt_builder: PromptBuilder,
    ) -> CompressionRequest:
        """Prepare a bounded history prefix for the compression agent."""
        if not 0 <= start <= end < len(self._active):
            raise IndexError("compaction range must select existing active messages")
        if any(key in self._protected for key in self._active[start : end + 1]):
            raise ValueError("compaction range includes a protected message")
        return compression_request(
            self.assemble()[: end + 1],
            start,
            system_prompt=system_prompt,
            user_prompt_builder=user_prompt_builder,
        )

    def compact(
        self,
        start: int,
        end: int,
        agent: CompressionAgent,
        *,
        system_prompt: str = COMPRESSION_SYSTEM_PROMPT,
        user_prompt_builder: PromptBuilder = default_user_prompt,
    ) -> str:
        """Call a compression agent, then replace the inclusive active range.

        ``start`` and ``end`` are zero-based indexes into :meth:`assemble`.
        The original selected messages are retained in SQLite for expansion.
        """
        active = self._active.copy()
        request = self._request(start, end, system_prompt, user_prompt_builder)
        summary = run_compressor(agent, request)
        if isawaitable(summary):
            if iscoroutine(summary):
                summary.close()
            raise TypeError("async compression agent requires await pocket.acompact(...)")
        return self._save_compaction(start, end, summary, active)

    async def acompact(
        self,
        start: int,
        end: int,
        agent: CompressionAgent,
        *,
        system_prompt: str = COMPRESSION_SYSTEM_PROMPT,
        user_prompt_builder: PromptBuilder = default_user_prompt,
    ) -> str:
        """Call an asynchronous compression agent and archive its result."""
        active = self._active.copy()
        request = self._request(start, end, system_prompt, user_prompt_builder)
        summary = run_compressor(agent, request)
        return self._save_compaction(
            start, end, await summary if isawaitable(summary) else summary, active
        )

    def _save_compaction(self, start: int, end: int, summary: str, active: list[str]) -> str:
        if self._active != active:
            raise RuntimeError("context changed while the compression agent was running")
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError("compression agent must return a non-empty string")

        selected = self._active[start : end + 1]
        self._next_tag += 1
        tag = f"ctx-{self._next_tag:04d}-{uuid4().hex[:8]}"
        self._db.execute(
            "INSERT INTO blocks VALUES (?, ?)",
            (tag, json.dumps([self._messages[key] for key in selected], ensure_ascii=False)),
        )
        self._db.commit()
        self._archive_tags.add(tag)
        message_id = self.record(
            {"_context_block": tag, "role": "user", "content": f"[{tag}]: {summary}\n(Assistant-generated memory.)"}
        )
        self._active.remove(message_id)
        self._active[start : end + 1] = [message_id]
        return tag

    def expand(self, tag: str) -> list[dict]:
        """Retrieve archived messages without replacing their active summary."""
        if tag not in self._archive_tags:
            raise KeyError(f"unknown context block: {tag}")
        row = self._db.execute("SELECT messages FROM blocks WHERE tag = ?", (tag,)).fetchone()
        if row is None:
            raise KeyError(f"unknown context block: {tag}")
        return json.loads(row[0])

    def assemble(self) -> list[dict]:
        """Return a copy of the current message projection."""
        return [json.loads(json.dumps(self._messages[key])) for key in self._active]

    def tools(
        self, used_tokens: int, context_token_limit: int, *, response_token_limit: int = 0
    ) -> list[dict]:
        """Return tool definitions available at the current context pressure."""
        if used_tokens < 0 or context_token_limit <= 0 or not 0 <= response_token_limit <= context_token_limit:
            raise ValueError("token counts must be non-negative and context_token_limit must be positive")
        tools = []
        if (
            any(key not in self._protected for key in self._active)
            and used_tokens >= self.compact_at_percent / 100 * max(0, context_token_limit - response_token_limit)
        ):
            tools.append(deepcopy(COMPACT_TOOL))
        if self._archive_tags:
            tools.append(deepcopy(EXPAND_TOOL))
        return tools

    def close(self) -> None:
        """Close the backing SQLite connection."""
        self._db.close()

    def __enter__(self) -> "ContextStore":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
