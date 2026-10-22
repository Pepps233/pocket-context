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
    tokenizer_name: str = ""
    snippet_tokens: int = 0
    seen_snippet_tokens: int = 0
    document_tokens: int = 0


class BrowseCompPlusEnvironment:
    def __init__(
        self,
        retriever: Retriever,
        *,
        k: int = 5,
        snippet_chars: int = 0,
        tokenizer_name: str = "",
        snippet_tokens: int = 0,
        seen_snippet_tokens: int = 0,
        document_tokens: int = 0,
    ):
        self.config = BrowseCompPlusConfig(
            k, snippet_chars, tokenizer_name, snippet_tokens, seen_snippet_tokens, document_tokens
        )
        if k < 1 or min(snippet_chars, snippet_tokens, seen_snippet_tokens, document_tokens) < 0:
            raise ValueError("k must be positive and text limits must be nonnegative")
        if (snippet_tokens or seen_snippet_tokens or document_tokens) and (not tokenizer_name or snippet_chars):
            raise ValueError("Token limits require tokenizer_name and cannot be combined with snippet_chars")
        self.tokenizer = None
        if tokenizer_name:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.retriever = retriever
        self.tool_call_counts = {"search": 0, "get_document": 0}
        self.retrieved_docids: set[str] = set()

    def truncate(self, text: str, limit: int) -> str:
        if not limit:
            return text
        return self.tokenizer.decode(
            self.tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=limit),
            skip_special_tokens=True,
        )

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
            result = [
                {
                    "docid": str(document["docid"]),
                    "score": document["score"],
                    "snippet": self.truncate(
                        document["text"],
                        (self.config.seen_snippet_tokens or self.config.snippet_tokens)
                        if str(document["docid"]) in self.retrieved_docids
                        else self.config.snippet_tokens,
                    )[: self.config.snippet_chars or None],
                    **({"previously_retrieved": True} if str(document["docid"]) in self.retrieved_docids else {}),
                }
                for document in documents
            ]
            self.retrieved_docids.update(str(document["docid"]) for document in documents)
        else:
            document = self.retriever.get_document(argument)
            if document is not None:
                self.retrieved_docids.add(str(document["docid"]))
                document = document | {"text": self.truncate(document["text"], self.config.document_tokens)}
            result = document if document is not None else {"error": f"Document {argument!r} not found"}
        return {"output": json.dumps(result, ensure_ascii=False), "returncode": 0}

    def get_template_vars(self, **kwargs) -> dict:
        return asdict(self.config) | kwargs

    def serialize(self) -> dict:
        return {"info": {"config": {"environment": asdict(self.config)}}}
