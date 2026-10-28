"""ACM BASE memory policy, ported from the released runner.py and prompts.py.

Pinned source: https://github.com/lixiaochuan2020/agentic-context-management
Revision: f06f90e728af8580a4515812425c1620144145a2

This uses the paper/tool-description half-open interval, excluding the current
manage_context assistant call. Unlike the release's drop_call_message=True
setting, it retains that call alongside its native tool result, avoiding orphan
results. Successful mixed batches finalize the boundary after all sibling results
so future compression never separates their native calls/results. Provider wire
formats and SQLite archives are shared-harness adaptations, not upstream code.
"""

import copy
import re

from pocketcontextagent.agents.context import ContextOperationError, SelectionError, encode
from pocketcontextagent.agents.utils.context_budget import output_token_setting
from pocketcontextagent.comparisons.methods.acm_prompts import _SUMMARY_INSTRUCTION, QUERY_MEMORY_PROMPT
from pocketcontextagent.comparisons.methods.base import ComparisonAgent, conversation_text
from pocketcontextagent.models.utils.compact_tool import CONTEXT_TOOLS
from pocketcontextagent.models.utils.response import response_text

SUMMARIZER_MAX_TOKENS = 4096
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
_MEMORY_RE = re.compile(r"<memory>(.*?)</memory>", re.DOTALL | re.IGNORECASE)
ACM_PROVENANCE = {
    "method": "ACM BASE",
    "upstream": "https://github.com/lixiaochuan2020/agentic-context-management",
    "revision": "f06f90e728af8580a4515812425c1620144145a2",
    "sources": ["src/runner.py", "src/tools.py", "src/prompts.py"],
    "boundary_deviation": (
        "Paper-described exclusion of the current manage_context assistant call, rather than "
        "the released drop_call_message=True configuration; retain native call/result pairs. "
        "Mixed batches advance the boundary after all sibling results to preserve their pairs."
    ),
    "archive_adaptation": "Transactional SQLite archives keyed by positive integer summary_id, not JSON files.",
    "provider_adaptation": "Native provider messages are rendered with reasoning and tool arguments for auxiliary calls.",
}


def _strip_thinking(text: str) -> str:
    """The released runner's thinking-block stripping rule."""
    return _THINK_RE.sub("", text or "").strip()


def _extract_memory(text: str) -> tuple[str, bool]:
    """The released parser and final-attempt raw-text fallback, without repairs."""
    if not text:
        return "", False
    match = _MEMORY_RE.search(text)
    if match:
        content = match.group(1).strip()
        if len(content) >= 50 and "<memory>" not in content and "</think>" not in content:
            return content, True
    return text.strip(), False


class ACMAgent(ComparisonAgent):
    """Agent-triggered, incremental ACM BASE compression and model-mediated recall."""

    context_tool_mode = "acm"
    method_name = "acm"

    def reset_method_state(self) -> None:
        self._boundary = 2
        self._summary_id = 0

    def _format_result(self, message: dict, text: str, *, error: bool = False, metadata=None) -> list[dict]:
        return self.model.format_observation_messages(
            message,
            [
                {
                    "output": text,
                    "returncode": -1 if error else 0,
                    "exception_info": text if error else "",
                    "extra": metadata or {},
                }
            ],
            self.get_template_vars(),
        )

    def _summarize(self, history: list[dict], summary_id: int) -> tuple[str, bool]:
        prompt = _SUMMARY_INSTRUCTION.format(question=self.question, conversation=conversation_text(history))
        content, parse_ok = "", False
        for attempt in range(2):
            response = self.call_auxiliary(prompt, kind="compaction", max_tokens=SUMMARIZER_MAX_TOKENS)
            content, parse_ok = _extract_memory(_strip_thinking(response_text(response)))
            self.compactions[-1].update(summary_id=summary_id, attempt=attempt + 1, parse_ok=parse_ok, committed=False)
            if parse_ok:
                break
        return content, parse_ok

    def _manage_context(self, message: dict, response_id: str) -> list[dict]:
        end = self.context.active.index(response_id)
        selected = self.context.active[self._boundary : end]
        if not selected:
            raise SelectionError(
                "Error: nothing to compress. There are no messages between your "
                "previous manage_context call and this one."
            )
        summary_id = self._summary_id + 1
        history = [self.context.records[key].message for key in selected]
        summary, parse_ok = self._summarize(history, summary_id)
        results = self._format_result(
            message, f"[summary_id: {summary_id}] {summary}", metadata={"summary_id": summary_id, "parse_ok": parse_ok}
        )
        self.context.remove(
            selected,
            operation="manage_context",
            summary_id=summary_id,
            archive=True,
        )
        observations = self.add_observations(results)
        self._summary_id = summary_id
        self._boundary = len(self.context.active)
        self.compactions[-1]["committed"] = True
        return observations

    def _query_memory(self, message: dict, action: dict) -> list[dict]:
        summary_id, query = action.get("summary_id"), action.get("query")
        if type(summary_id) is not int or summary_id <= 0 or not isinstance(query, str) or not query.strip():
            raise SelectionError("query_memory requires a positive integer summary_id and a nonempty query")
        history = self.context.archive(summary_id)
        prompt = QUERY_MEMORY_PROMPT.format(summary_id=summary_id, history=conversation_text(history), query=query)
        _, allowance = output_token_setting(self.model, self.config.response_token_limit)
        response = self.call_auxiliary(prompt, kind="query", max_tokens=allowance)
        self.memory_queries[-1].update(summary_id=summary_id, query=query)
        return self.add_observations(
            self._format_result(
                message,
                f"[query_memory: summary_id={summary_id}]\n\n{_strip_thinking(response_text(response))}",
                metadata={"summary_id": summary_id},
            )
        )

    def dispatch_context_actions(self, message: dict) -> list[dict] | None:
        actions = message.get("extra", {}).get("actions", [])
        if not any(
            action.get("type") in (*CONTEXT_TOOLS, "rejected", "manage_context", "query_memory") for action in actions
        ):
            return None
        key = (id(message), encode(actions))
        if key in self._context_results:
            return self._context_results[key]
        if any(action.get("type") in (*CONTEXT_TOOLS, "rejected") for action in actions):
            results = self._results(message, "Unsupported memory operation. No actions executed.", error=True)
            self._context_results[key] = results
            return results
        try:
            self.context.projection()
            self.context._validate_response(message)
        except (SelectionError, ContextOperationError) as exc:
            results = self._results(message, str(exc), error=True)
            self._context_results[key] = results
            return results

        response_id = self.context._sources[id(message)]
        results = []
        managed = False
        for action in actions:
            single = copy.deepcopy(message)
            single["extra"]["actions"] = [action]
            operation = action.get("type")
            try:
                if action.get("error"):
                    raise SelectionError(action["error"])
                if operation in ("manage_context", "query_memory"):
                    allowed = {"type", "tool_call_id"} | (
                        {"summary_id", "query"} if operation == "query_memory" else set()
                    )
                    if set(action) - allowed:
                        raise SelectionError("Memory tools do not accept range selection or extra arguments")
                if operation == "manage_context":
                    observations = self._manage_context(single, response_id)
                    managed = True
                elif operation == "query_memory":
                    observations = self._query_memory(single, action)
                else:
                    if update := getattr(self.env, "update_context", None):
                        update(self.context)
                    observations = self.add_observations(
                        self.model.format_observation_messages(
                            single, [self.env.execute(action)], self.get_template_vars()
                        )
                    )
            except (SelectionError, ContextOperationError) as exc:
                observations = self.add_observations(self._format_result(single, str(exc), error=True))
            results.extend(observations)
        if managed:
            self._boundary = len(self.context.active)
        self._context_results[key] = results
        return results

    def serialize(self, *extra_dicts) -> dict:
        return super().serialize({"acm_provenance": ACM_PROVENANCE}, *extra_dicts)
