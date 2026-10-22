"""Corpus-only commands for the existing bash-tool model interfaces."""

import json
import re
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Protocol

from pocketcontextagent.exceptions import Submitted

if TYPE_CHECKING:
    from pocketcontextagent.agents.context import ContextManager


def extract_answer(text: str) -> dict:
    answers = re.findall(r"Exact Answer:[ \t]*\**[ \t]*([^\r\n]*)", text)
    confidence = re.findall(r"Confidence:\s*(\d+)\s*%?", text, re.IGNORECASE)
    return {
        "final_answer": answers[-1].strip().rstrip("*").strip() if answers else "",
        "confidence": int(confidence[-1]) if confidence else 100,
    }


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
        self.in_context_docids: set[str] = set()
        self.memory_tags: dict[str, list[str]] = {}

    @staticmethod
    def is_final_answer(text: str) -> bool:
        return bool(extract_answer(text)["final_answer"])

    def update_context(self, context: "ContextManager") -> None:
        self.in_context_docids = {
            docid
            for key in context.active
            for docid in context.records[context.records[key].source_record or key]
            .message.get("extra", {})
            .get("docids", [])
        }
        self.memory_tags = {}
        for event in context.events:
            for docid in {
                docid
                for key in event["selected_ids"]
                for docid in context.records[key].message.get("extra", {}).get("docids", [])
            }:
                self.memory_tags.setdefault(docid, []).append(f"ctx-{event['revision']:04d}")

    def document_state(self, docid: str) -> dict:
        if docid not in self.retrieved_docids:
            return {"context_status": "new"}
        if docid in self.in_context_docids:
            return {
                "previously_retrieved": True,
                "context_status": "in_context",
                "hint": f"Seen earlier and still in active context. Use get_document {docid} for source text if needed.",
            }
        return {
            "previously_retrieved": True,
            "context_status": "compressed",
            "memory_tags": self.memory_tags.get(docid, []),
            "hint": f"Seen earlier but compacted. Use expand_block with a memory_tags tag to recover archived evidence, or get_document {docid} for source text.",
        }

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
                "exception_info": "",
            }
        command, argument = parts
        if command == "submit":
            if not self.is_final_answer(argument):
                return {
                    "output": "Include an Exact Answer: line in your final response, with explanation and confidence.",
                    "returncode": 1,
                    "exception_info": "",
                }
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
                    **self.document_state(str(document["docid"])),
                }
                for document in documents
            ]
            docids = [str(document["docid"]) for document in documents]
            self.retrieved_docids.update(docids)
            self.in_context_docids.update(docids)
        else:
            document = self.retriever.get_document(argument)
            if document is not None:
                self.retrieved_docids.add(str(document["docid"]))
                document = document | {"text": self.truncate(document["text"], self.config.document_tokens)}
            docids = [str(document["docid"])] if document is not None else []
            self.in_context_docids.update(docids)
            result = document if document is not None else {"error": f"Document {argument!r} not found"}
        return {
            "output": json.dumps(result, ensure_ascii=False),
            "returncode": 0,
            "exception_info": "",
            "extra": {"docids": docids},
        }

    def get_template_vars(self, **kwargs) -> dict:
        return asdict(self.config) | kwargs

    def serialize(self) -> dict:
        return {"info": {"config": {"environment": asdict(self.config)}}}
