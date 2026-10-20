"""Shared native tool schema and action parsing."""

import json

COMPACT_FUNCTION = {
    "name": "compact_block",
    "description": (
        "Decide when compaction is useful and replace a long context range with your own summary. "
        "Use exactly one compact_block call and no other tools in a dedicated response. "
        "Preserve the latest user request and latest model output and its results. Avoid compacting short information. "
        "Choose unique literal excerpts from the first and last complete messages in the active request, "
        "excluding this response. Select complete tool exchanges, including all calls, reasoning and results. "
        "Preserve task intent, constraints, relevant paths and identifiers, command purposes and outcomes. "
        "Preserve uncertainty and distinguish observations from assumptions. Include enough causal detail "
        "to make surviving references understandable, including references outside a tool exchange. "
        "Structural validation cannot prove semantic completeness. Earlier tagged memories may be selected. "
        "Defaults require 2000 visible text characters selected and 512 characters net request reduction, "
        "including this call and receipt. The archive is local; only the summary remains in active context."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "start_quote": {"type": "string", "description": "Unique excerpt from the first selected message"},
            "end_quote": {"type": "string", "description": "Unique excerpt from the last selected message"},
            "summary": {"type": "string", "description": "Information needed to continue working"},
        },
        "required": ["start_quote", "end_quote", "summary"],
        "additionalProperties": False,
    },
}
COMPACT_TOOL = {"type": "function", "function": COMPACT_FUNCTION}
COMPACT_TOOL_RESPONSE_API = {"type": "function", **COMPACT_FUNCTION}


def parse_native_calls(calls: list[dict]) -> list[dict]:
    """Normalize both API formats, retaining a result slot for every rejected call."""
    if any(call["name"] == "compact_block" for call in calls) and len(calls) != 1:
        return [
            {
                "type": "rejected",
                "tool_call_id": call["id"],
                "error": "compact_block requires exactly one call and no other tools. No actions executed.",
            }
            for call in calls
        ]
    actions = []
    for call in calls:
        error = ""
        try:
            args = json.loads(call["arguments"])
        except (ValueError, TypeError) as exc:
            args = {}
            error = f"Error parsing tool call arguments: {exc}."
        name = call["name"]
        required = ["start_quote", "end_quote", "summary"] if name == "compact_block" else ["command"]
        if name not in ("bash", "compact_block"):
            error += f"Unknown tool '{name}'."
        if not isinstance(args, dict) or any(not isinstance(args.get(key), str) or not args[key] for key in required):
            error += (
                "Missing 'command' argument in bash tool call."
                if name == "bash"
                else "Missing or invalid start_quote, end_quote, summary argument in compact_block tool call."
            )
        if name == "compact_block":
            actions.append({"type": name, "tool_call_id": call["id"], **(args if isinstance(args, dict) else {})})
            if error:
                actions[-1]["error"] = error
        elif error:
            raise ValueError(error)
        else:
            actions.append({"command": args["command"], "tool_call_id": call["id"]})
    return actions
