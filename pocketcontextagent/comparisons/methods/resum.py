"""Training-free ReSum port from Alibaba-NLP/DeepResearch.

Pinned upstream: f72f75d8c3eb842f2bbbab096a12206ff66e270f
Sources: WebAgent/WebResummer/src/{react_agent,summary_utils,prompt}.py.
https://github.com/Alibaba-NLP/DeepResearch/tree/f72f75d8c3eb842f2bbbab096a12206ff66e270f/WebAgent/WebResummer/src
Copyright the Alibaba-NLP/DeepResearch authors; upstream Apache-2.0 license.

The shared corpus agent supplies tools, token accounting and termination budgets;
only the released training-free summary/restart algorithm is ported here.
"""

import re

from pydantic import Field

from pocketcontextagent.agents.context import ContextOperationError, SelectionError
from pocketcontextagent.comparisons.methods.base import ComparisonAgent, ComparisonConfig, public_messages
from pocketcontextagent.comparisons.methods.resum_prompts import (
    CONTINUATION_TEXT,
    QUERY_SUMMARY_PROMPT,
    QUERY_SUMMARY_PROMPT_LAST,
)
from pocketcontextagent.models.utils.response import response_items

UPSTREAM_COMMIT = "f72f75d8c3eb842f2bbbab096a12206ff66e270f"
UPSTREAM_SOURCE = "https://github.com/Alibaba-NLP/DeepResearch/tree/" + UPSTREAM_COMMIT + "/WebAgent/WebResummer/src"


class ReSumConfig(ComparisonConfig):
    resum_trigger_ratio: float = Field(default=0.9, gt=0, le=1)
    resum_headroom_tokens: int = Field(default=1000, ge=0)
    resum_summary_token_limit: int = Field(default=4096, gt=0, le=4096)


def summary_prompt(question: str, recent_history_messages: list[dict], last_summary: str | None) -> str:
    """Use the released Python-dict history representation and literal substitution."""
    prompt = QUERY_SUMMARY_PROMPT_LAST if last_summary else QUERY_SUMMARY_PROMPT
    prompt = prompt.replace("{{{question}}}", question).replace(
        "{{{recent_history_messages}}}", "\n".join(str(message) for message in recent_history_messages)
    )
    return prompt.replace("{{{last_summary}}}", last_summary) if last_summary else prompt


def normalize_summary(content: str) -> str:
    """Match summary_utils.call_resum_server, including untagged output fallback."""
    if not content:
        return ""
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    if "<summary>" in content:
        content = content.split("<summary>")[1].split("</summary>")[0]
    return "<summary>" + content + "</summary>"


class ReSumAgent(ComparisonAgent):
    config_class = ReSumConfig
    method_name = "resum"

    def __init__(self, model, env, *, config_class=ReSumConfig, **kwargs):
        super().__init__(model, env, config_class=config_class, **kwargs)

    def reset_method_state(self) -> None:
        self.last_summary: str | None = None
        self.resum_summaries: list[dict] = []

    @property
    def summary_threshold(self) -> float:
        return self.config.resum_trigger_ratio * (self.config.context_token_limit - self.config.resum_headroom_tokens)

    def query(self) -> dict:
        self.check_limits()
        history = self.context.projection()
        if (
            not self._final_generation
            and self.config.context_token_limit
            and len(history) > 2
            and self.context_tokens(history) >= self.summary_threshold
        ):
            self.summarize_history(history)
        return super().query()

    def summarize_history(self, history: list[dict]) -> bool:
        """Replace ALL live history after the system only after a successful summary."""
        selected = list(self.context.active[1:])
        attempt = {
            "input_ids": list(self.context.active[2:]),
            "selected_ids": selected,
            "last_summary": self.last_summary,
            "threshold": self.summary_threshold,
            "success": False,
        }
        self.resum_summaries.append(attempt)
        prompt = summary_prompt(self.question, public_messages(history[2:]), self.last_summary)
        try:
            response = self.call_auxiliary(prompt, kind="compaction", max_tokens=self.config.resum_summary_token_limit)
        except ContextOperationError as exc:
            attempt["error"] = str(exc)
            self.logger.warning("ReSum summary failed; history preserved: %s", exc)
            return False
        content = "\n".join(
            block.get("text", "")
            for item in response_items(response)
            if item.get("role") == "assistant" or item.get("type") == "message"
            for block in (
                [{"type": "text", "text": item["content"]}]
                if isinstance(item.get("content"), str)
                else item.get("content") or []
            )
            if block.get("type") in ("text", "output_text")
        )
        summary = normalize_summary(content)
        if not summary:
            attempt["error"] = "Empty summary response"
            return False
        continuation = "Question: " + self.question + CONTINUATION_TEXT + "Summary: " + summary + "\n"
        message = self.model.format_message(role="user", content=continuation)
        try:
            self.context.remove(selected, operation="resum", summary_id=len(self.resum_summaries), archive=False)
        except (ContextOperationError, SelectionError) as exc:
            attempt["error"] = str(exc)
            self.logger.warning("ReSum reset failed; history preserved: %s", exc)
            return False
        self.add_messages(message)
        self.last_summary = summary
        attempt.update(success=True, summary=summary, continuation_id=self.context.active[-1])
        return True

    def serialize(self, *extra_dicts) -> dict:
        return super().serialize(
            {
                "resum": {
                    "upstream_repository": "Alibaba-NLP/DeepResearch",
                    "upstream_commit": UPSTREAM_COMMIT,
                    "upstream_source": UPSTREAM_SOURCE,
                    "training_free": True,
                    "adaptations": [
                        "Shared corpus tools, system prompt, tokenizer and episode budgets",
                        "Configured context window replaces MAX_CONTEXT * 1024",
                        "Shared auxiliary model endpoint replaces standalone ReSumTool server",
                    ],
                    "config": {
                        "context_token_limit": self.config.context_token_limit,
                        "trigger_ratio": self.config.resum_trigger_ratio,
                        "headroom_tokens": self.config.resum_headroom_tokens,
                        "summary_token_limit": self.config.resum_summary_token_limit,
                    },
                    "last_summary": self.last_summary,
                    "summaries": self.resum_summaries,
                }
            },
            *extra_dicts,
        )
