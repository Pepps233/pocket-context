"""Versioned method selection; shared budgets do not overwrite published memory policies."""

import copy
import hashlib
import json
from dataclasses import dataclass

from pocketcontextagent.comparisons.methods.acm import ACM_PROVENANCE
from pocketcontextagent.comparisons.methods.acm_prompts import BCP_STOP_PROMPT
from pocketcontextagent.comparisons.methods.acm_prompts import build_system_prompt as acm_prompt
from pocketcontextagent.comparisons.methods.react import build_system_prompt as react_prompt
from pocketcontextagent.comparisons.methods.resum import UPSTREAM_COMMIT as RESUM_COMMIT


@dataclass(frozen=True)
class Method:
    agent_class: str
    context_tools: str
    description: str
    source: str
    revision: str


METHODS = {
    "pocket_context": Method(
        "default", "pocket_context", "Pocket Context", "this repository", "installed implementation"
    ),
    "react": Method(
        "react",
        "none",
        "Append-only ReAct (native corpus tools)",
        "lixiaochuan2020/agentic-context-management",
        ACM_PROVENANCE["revision"],
    ),
    "resum": Method("resum", "none", "ReSum (training-free, versioned port)", "Alibaba-NLP/DeepResearch", RESUM_COMMIT),
    "acm": Method(
        "acm",
        "acm",
        "ACM Base (paper-described boundary, versioned port)",
        "lixiaochuan2020/agentic-context-management",
        ACM_PROVENANCE["revision"],
    ),
}

METHOD_AGENT_FIELDS = {
    "resum_trigger_ratio",
    "resum_headroom_tokens",
    "resum_summary_token_limit",
    "context_pressure_fraction",
    "system_template",
    "instance_template",
    "context_limit_template",
}


def literal_template(text: str) -> str:
    return "{{ " + json.dumps(text, ensure_ascii=False) + " }}"


def configure_method(config: dict, method: str) -> dict:
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {', '.join(METHODS)}")
    config = copy.deepcopy(config)
    strategy = METHODS[method]
    agent = config.setdefault("agent", {})
    agent["agent_class"] = strategy.agent_class
    config.setdefault("model", {})["context_tools"] = strategy.context_tools
    config["model"]["research_tools"] = "corpus"
    config.setdefault("environment", {})["context_tools"] = strategy.context_tools
    config["environment"]["research_tools"] = "corpus"
    if method != "pocket_context":
        window = agent.get("context_token_limit", 128000)
        agent["system_template"] = literal_template(acm_prompt(window) if method == "acm" else react_prompt(window))
        agent["instance_template"] = "{{task}}"
        agent["context_limit_template"] = literal_template(BCP_STOP_PROMPT)
        if agent.get("context_pressure_fraction", 0.95) != 0.95:
            raise ValueError("Published baseline ports use the original 95% current-history stop threshold")
        agent["context_pressure_fraction"] = 0.95
    if method == "resum":
        agent.setdefault("resum_trigger_ratio", 0.9)
        agent.setdefault("resum_headroom_tokens", 1000)
        agent.setdefault("resum_summary_token_limit", 4096)
    return config


def method_provenance(config: dict, method: str) -> dict:
    spec = METHODS[method]
    agent = config["agent"]
    return {
        "name": method,
        "description": spec.description,
        "source": spec.source,
        "revision": spec.revision,
        "settings": {
            key: agent[key] for key in sorted(METHOD_AGENT_FIELDS) if key in agent and not key.endswith("template")
        },
        "prompts_sha256": hashlib.sha256(
            json.dumps(
                {key: agent.get(key) for key in ("system_template", "instance_template", "context_limit_template")},
                sort_keys=True,
            ).encode()
        ).hexdigest(),
        "evaluation": "shared-corpus, same-policy-weight comparison; not reproduction of published scores",
        "acm_boundary": "exclude issuing assistant; preserve native call/result pair" if method == "acm" else None,
        "auxiliary_model": "same model and endpoint as main policy",
    }
