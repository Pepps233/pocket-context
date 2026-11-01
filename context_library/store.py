"""A small chronological message store with reversible compaction."""

import json
import sqlite3
from pathlib import Path
from uuid import uuid4


class ContextStore:
    """Keep an active projection in memory and archived blocks in SQLite.

    Messages are provider-shaped dictionaries owned by the calling agent.
    Compaction receives a summary from the caller, so this module has no model
    or provider dependencies.
    """

    def __init__(self, db_path: str | Path = ":memory:"):
        self.db_path = str(db_path)
        self._db = sqlite3.connect(self.db_path)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS blocks (tag TEXT PRIMARY KEY, messages TEXT NOT NULL)"
        )
        self._db.commit()
        self._messages: dict[str, dict] = {}
        self._active: list[str] = []
        self._next_id = 0
        self._next_tag = 0

    def record(self, message: dict) -> str:
        """Append a message and return its stable ID."""
        self._next_id += 1
        message_id = f"m{self._next_id:06d}"
        self._messages[message_id] = json.loads(json.dumps(message))
        self._active.append(message_id)
        return message_id

    def compact(self, start: int, end: int, summary: str) -> str:
        """Replace the inclusive active-message range with a summary.

        ``start`` and ``end`` are zero-based indexes into :meth:`assemble`.
        The original selected messages are retained in SQLite for expansion.
        """
        if not 0 <= start <= end < len(self._active):
            raise IndexError("compaction range must select existing active messages")
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError("summary must be a non-empty string")

        selected = self._active[start : end + 1]
        self._next_tag += 1
        tag = f"ctx-{self._next_tag:04d}-{uuid4().hex[:8]}"
        self._db.execute(
            "INSERT INTO blocks VALUES (?, ?)",
            (tag, json.dumps([self._messages[key] for key in selected], ensure_ascii=False)),
        )
        self._db.commit()
        message_id = self.record({"_context_block": tag, "role": "system", "content": summary})
        self._active.remove(message_id)
        self._active[start : end + 1] = [message_id]
        return tag

    def expand(self, tag: str) -> list[dict]:
        """Restore an archived block in place and return the restored messages."""
        index = next(
            (i for i, key in enumerate(self._active) if self._messages[key].get("_context_block") == tag),
            None,
        )
        row = self._db.execute("SELECT messages FROM blocks WHERE tag = ?", (tag,)).fetchone()
        if row is None:
            raise KeyError(f"unknown context block: {tag}")
        if index is None:
            raise ValueError(f"context block is not currently active: {tag}")

        restored = json.loads(row[0])
        ids = []
        for message in restored:
            self._next_id += 1
            message_id = f"m{self._next_id:06d}"
            self._messages[message_id] = message
            ids.append(message_id)
        self._active[index : index + 1] = ids
        return restored

    def assemble(self) -> list[dict]:
        """Return a copy of the current message projection."""
        return [json.loads(json.dumps(self._messages[key])) for key in self._active]

    def close(self) -> None:
        """Close the backing SQLite connection."""
        self._db.close()

    def __enter__(self) -> "ContextStore":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
