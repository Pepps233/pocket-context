"""Local, training-free context strategies sharing one benchmark configuration."""

import copy
from dataclasses import dataclass
from typing import Literal

MethodName = Literal["pocket_context", "react", "resum", "acm"]


@dataclass(frozen=True)
class Method:
    agent_class: str
    context_tools: str
    description: str


METHODS = {
    "pocket_context": Method("default", "pocket_context", "Agent-selected blocks with exact archived-text restoration"),
    "react": Method("react", "none", "Append-only ReAct without memory tools"),
    "resum": Method("resum", "none", "Training-free periodic summarization; local ReSum adaptation"),
    "acm": Method("acm", "acm", "Agent-triggered segment summaries and query-based recall; local ACM adaptation"),
}


def configure_method(config: dict, method: str) -> dict:
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {', '.join(METHODS)}")
    config = copy.deepcopy(config)
    strategy = METHODS[method]
    config.setdefault("agent", {})["agent_class"] = strategy.agent_class
    config.setdefault("model", {})["context_tools"] = strategy.context_tools
    config.setdefault("environment", {})["context_tools"] = strategy.context_tools
    return config
