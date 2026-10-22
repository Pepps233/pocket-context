"""Corpus-only commands for the existing bash-tool model interfaces."""

import json
from dataclasses import asdict, dataclass
from typing import Protocol

from pocketcontextagent.exceptions import Submitted


class Retriever(Protocol):
    def search(self, query: str, k: int) -> list[dict]: ...

    def get_document(self, docid: str) -> dict | None: ...


@dataclass
class BrowseCompPlusConfig:
    k: int = 5
    snippet_chars: int = 0


class BrowseCompPlusEnvironment:
    def __init__(self, retriever: Retriever, *, k: int = 5, snippet_chars: int = 0):
        if k < 1 or snippet_chars < 0:
            raise ValueError("k must be positive and snippet_chars must be nonnegative")
        self.config = BrowseCompPlusConfig(k, snippet_chars)
        self.retriever = retriever
        self.tool_call_counts = {"search": 0, "get_document": 0}
        self.retrieved_docids: set[str] = set()

    def execute(self, action: dict, cwd: str = "") -> dict:
        parts = action.get("command", "").strip().split(maxsplit=1)
        if len(parts) != 2 or parts[0] not in ("search", "get_document", "submit"):
            return {
                "output": "Use search QUERY, get_document DOCID, or submit FINAL_ANSWER. Shell commands are unavailable.",
                "returncode": 1,
            }
        command, argument = parts
        if command == "submit":
            raise Submitted(
                {"role": "exit", "content": argument, "extra": {"exit_status": "Submitted", "submission": argument}}
            )
        self.tool_call_counts[command] += 1
        if command == "search":
            documents = self.retriever.search(argument, self.config.k)
            self.retrieved_docids.update(str(document["docid"]) for document in documents)
            result = [
                {
                    "docid": str(document["docid"]),
                    "score": document["score"],
                    "snippet": document["text"][: self.config.snippet_chars or None],
                }
                for document in documents
            ]
        else:
            document = self.retriever.get_document(argument)
            if document is not None:
                self.retrieved_docids.add(str(document["docid"]))
            result = document if document is not None else {"error": f"Document {argument!r} not found"}
        return {"output": json.dumps(result, ensure_ascii=False), "returncode": 0}

    def get_template_vars(self, **kwargs) -> dict:
        return asdict(self.config) | kwargs

    def serialize(self) -> dict:
        return {"info": {"config": {"environment": asdict(self.config)}}}
