"""Shared transport, immutable audit history and accounting, not a shared memory policy."""

import copy
import json
import sqlite3
import time
from contextlib import closing
from typing import Literal

from pydantic import create_model

from pocketcontextagent.agents.context import (
    ContextAssemblyError,
    ContextManager,
    ContextOperationError,
    SelectionError,
    encode,
    provider_items,
    relationships,
    visible_text,
)
from pocketcontextagent.agents.default import AgentConfig, DefaultAgent
from pocketcontextagent.agents.utils.context_budget import append_text, estimate_tokens, output_token_setting
from pocketcontextagent.exceptions import FormatError, LimitsExceeded
from pocketcontextagent.models.test_models import DeterministicResponseAPIToolcallModel, DeterministicToolcallModel

MEMORY_TOOLS = ("compact_block", "expand_block", "manage_context", "query_memory", "rejected")


def public_messages(messages: list[dict]) -> list[dict]:
    return [
        copy.deepcopy({key: value for key, value in item.items() if key != "extra"})
        for message in messages
        for item in provider_items(message)
    ]


def conversation_text(messages: list[dict]) -> str:
    rendered = []
    for item in public_messages(messages):
        text = visible_text(item)
        for call in item.get("tool_calls") or []:
            text.append("<tool_call>" + encode(call["function"]) + "</tool_call>")
        if item.get("type") == "function_call":
            text.append("<tool_call>" + encode({"name": item["name"], "arguments": item["arguments"]}) + "</tool_call>")
        rendered.append(f"[{item.get('role', item.get('type', '?'))}] " + "\n".join(text))
    return "\n\n".join(rendered)


class ComparisonContext(ContextManager):
    """Deletion events preserve a complete audit trail without inventing PCA memory blocks."""

    def _catalog(self, model, blocks, records) -> list[dict]:
        return []

    def eligible_blocks(self, ids=None, records=None) -> dict[str, list[str]]:
        return {}

    def _expected_active(self, events: list[dict] | None = None) -> list[str]:
        expected = list(self.ingested)
        for event in self.events if events is None else events:
            selected = event["selected_ids"]
            start = expected.index(selected[0])
            if expected[start : start + len(selected)] != selected:
                raise ContextAssemblyError("Comparison history deletion is inconsistent")
            del expected[start : start + len(selected)]
        return expected

    def remove(
        self, selected: list[str], *, operation: str, summary_id: int | None = None, archive: bool = False
    ) -> None:
        self.projection()
        if not selected:
            raise SelectionError("Nothing to compress")
        start = self.active.index(selected[0])
        if self.active[start : start + len(selected)] != selected:
            raise SelectionError("Compression must remove one contiguous interval")
        messages = [self.records[key].message for key in selected]
        if any(message.get("role") in ("system", "developer") for message in messages):
            raise SelectionError("System instructions cannot be compressed")
        calls = [call for message in messages for call in relationships(message)[0]]
        results = [result for message in messages for result in relationships(message)[1]]
        if sorted(calls) != sorted(results) or len(calls) != len(set(calls)):
            raise SelectionError("Compression would split an unresolved tool exchange")
        proposed = self.active[:start] + self.active[start + len(selected) :]
        event = {
            "operation": operation,
            "revision": self.revision + 1,
            "selected_ids": list(selected),
            "active_before": list(self.active),
            "active_after": proposed,
            "summary_id": summary_id,
            "archived": archive,
            "tool_call_id": f"{operation}-{self.revision + 1}",
        }
        if proposed != self._expected_active(self.events + [event]):
            raise ContextAssemblyError("Comparison deletion changed unrelated history")
        try:
            with closing(self._connect()) as db, db:
                db.execute(
                    "CREATE TABLE IF NOT EXISTS comparison_archives "
                    "(run_id TEXT, summary_id INTEGER, record TEXT NOT NULL, PRIMARY KEY (run_id, summary_id))"
                )
                db.execute("BEGIN IMMEDIATE")
                if archive:
                    if type(summary_id) is not int or summary_id <= 0:
                        raise SelectionError("Archive summary_id must be a positive integer")
                    db.execute(
                        "INSERT INTO comparison_archives VALUES (?, ?, ?)",
                        (self.run_id, summary_id, encode(public_messages(messages))),
                    )
                db.execute(
                    "INSERT INTO events VALUES (?, ?, ?, ?)",
                    (self.run_id, event["revision"], event["tool_call_id"], encode(event)),
                )
        except (sqlite3.Error, OSError) as exc:
            raise ContextOperationError(f"Comparison storage failed: {exc}; history preserved") from exc
        self.active = proposed
        self.events.append(event)
        self.revision += 1
        self.projection()

    def archive(self, summary_id: int) -> list[dict]:
        if type(summary_id) is not int or summary_id <= 0:
            raise SelectionError("summary_id must be a positive integer")
        with closing(self._connect()) as db:
            exists = db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='comparison_archives'"
            ).fetchone()
            row = (
                db.execute(
                    "SELECT record FROM comparison_archives WHERE run_id=? AND summary_id=?", (self.run_id, summary_id)
                ).fetchone()
                if exists
                else None
            )
        if row is None:
            raise SelectionError(f"summary_id {summary_id} not found")
        return json.loads(row[0])


class ComparisonConfig(AgentConfig):
    context_pressure_fraction: Literal[0.95] = 0.95
    """Stop threshold applies to current history, not history plus the output reservation."""


class ComparisonAgent(DefaultAgent):
    context_tool_mode = "none"
    method_name = "react"

    def __init__(self, model, env, *, config_class=ComparisonConfig, **kwargs):
        if hasattr(model.config, "action_regex"):
            raise TypeError("Comparison methods require a native-tool model")
        if "context_tools" not in getattr(type(model.config), "model_fields", {}):
            if not isinstance(model, (DeterministicToolcallModel, DeterministicResponseAPIToolcallModel)):
                raise TypeError("Comparison methods require configurable native tools")
            model.config = create_model(
                f"{type(model.config).__name__}Comparison",
                __base__=type(model.config),
                context_tools=(Literal["pocket_context", "none", "acm"], self.context_tool_mode),
                research_tools=(Literal["bash", "corpus"], "corpus"),
            )(**model.config.model_dump())
        model.config.context_tools = self.context_tool_mode
        model.config.research_tools = "corpus"
        super().__init__(model, env, config_class=config_class, **kwargs)
        self.context = ComparisonContext(self.config.context_db_path)
        self.memory_queries: list[dict] = []
        self.reset_method_state()

    @property
    def question(self) -> str:
        return self.extra_template_vars.get("task", "")

    def reset_method_state(self) -> None:
        pass

    def run(self, task: str = "", **kwargs) -> dict:
        self.cost = 0.0
        self.n_calls = 0
        self.n_consecutive_format_errors = 0
        self._start_time = time.time()
        if reset := getattr(self.env, "reset", None):
            reset()
        return super().run(task, **kwargs)

    def add_messages(self, *messages, **kwargs) -> list[dict]:
        if not isinstance(self.context, ComparisonContext):
            if self.context.records:
                raise ContextOperationError("Cannot replace an initialized context")
            self.context = ComparisonContext(self.config.context_db_path)
            self.memory_queries = []
            self.reset_method_state()
        return super().add_messages(*messages, **kwargs)

    def query(self) -> dict:
        self.check_limits()
        options = {"allow_text_response": True} if getattr(self.env, "is_final_answer", None) else {}
        if self.config.context_token_limit:
            tokens = self.context_tokens(self.context.projection())
            if tokens >= self.config.context_token_limit * self.config.context_pressure_fraction:
                self._final_generation = True
                self.add_messages(
                    self.model.format_message(
                        role="user", content=self._render_template(self.config.context_limit_template)
                    )
                )
                options["final_answer"] = True
            remaining = self.config.context_token_limit - self.context_tokens(self.context.projection())
            if remaining <= 0:
                raise LimitsExceeded(
                    {
                        "role": "exit",
                        "content": "Context window exhausted",
                        "extra": {
                            "exit_status": "LimitsExceeded",
                            "submission": "",
                            "context_limit_reached": True,
                        },
                    }
                )
            key, allowance = output_token_setting(self.model, self.config.response_token_limit)
            options[key] = min(allowance, remaining)
        self.n_calls += 1
        message = self.model.query(self.context.assemble(self.model), **options)
        self.cost += message.get("extra", {}).get("cost", 0.0)
        self.add_messages(message)
        return message

    def prepare_observations(self, messages: list[dict], history: list[dict], *, truncate: bool = True) -> list[dict]:
        messages = copy.deepcopy(messages)
        if messages and self.config.context_token_limit:
            _, allowance = output_token_setting(self.model, self.config.response_token_limit)
            tokens = self.context_tokens(history + messages) + allowance
            append_text(messages[-1], f"[CURRENT CONTEXT TOKEN: {tokens}]")
        return messages

    def call_auxiliary(self, prompt: str, *, kind: str = "compaction", max_tokens: int = 4096) -> dict:
        self.check_limits(include_step_limit=False)
        messages = [self.model.format_message(role="user", content=prompt)]
        allowance = max_tokens
        if self.config.context_token_limit:
            remaining = self.config.context_token_limit - estimate_tokens(
                self.context._prepare(self.model, messages), getattr(self.env, "tokenizer", None), tools=[]
            )
            if remaining <= 0:
                raise ContextOperationError("Auxiliary prompt exceeds the context window; history preserved")
            allowance = min(allowance, remaining)
        call = {"messages": messages, "response": {}, "error": "", "attempted": True}
        (self.compactions if kind == "compaction" else self.memory_queries).append(call)
        key, _ = output_token_setting(self.model, allowance)
        try:
            call["response"] = self.model.query(messages, final_answer=True, tools=[], **{key: allowance})
            return call["response"]
        except Exception as exc:
            call["error"] = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, FormatError) and exc.messages:
                call["response"] = exc.messages[0]
            raise ContextOperationError(f"Auxiliary call failed ({type(exc).__name__}); history preserved") from exc
        finally:
            self.cost += call["response"].get("extra", {}).get("cost", 0.0)

    def _results(self, message: dict, text: str, *, error: bool = False) -> list[dict]:
        return self.add_observations(
            self.model.format_observation_messages(
                message,
                [
                    {"output": text, "returncode": -1 if error else 0, "exception_info": text if error else ""}
                    for _ in message.get("extra", {}).get("actions", [])
                ],
                self.get_template_vars(),
            )
        )

    def dispatch_context_actions(self, message: dict) -> list[dict] | None:
        actions = message.get("extra", {}).get("actions", [])
        if not any(action.get("type") in MEMORY_TOOLS for action in actions):
            return None
        key = (id(message), encode(actions))
        if key not in self._context_results:
            self._context_results[key] = self._results(
                message, "Memory tools are unavailable. No actions executed.", error=True
            )
        return self._context_results[key]

    def serialize(self, *extra_dicts) -> dict:
        return super().serialize(
            {
                "comparison": {"method": self.method_name, "implementation": "versioned shared-harness port"},
                "memory_queries": self.memory_queries,
                "info": {
                    "model_stats": {
                        "api_calls": self.n_calls + len(self.compactions) + len(self.memory_queries),
                        "compaction_api_calls": len(self.compactions),
                        "memory_query_api_calls": len(self.memory_queries),
                        "cost_available": (
                            sum(
                                "cost" in message.get("extra", {})
                                for message in self.messages
                                if message.get("role") == "assistant"
                                or message.get("object") == "response"
                                or "response" in message.get("extra", {})
                            )
                            >= self.n_calls
                            and all(
                                "cost" in call["response"].get("extra", {})
                                and call["response"].get("extra", {}).get("cost_available", True)
                                for call in self.compactions + self.memory_queries
                            )
                            and all(message.get("extra", {}).get("cost_available", True) for message in self.messages)
                        ),
                    }
                },
            },
            *extra_dicts,
        )
