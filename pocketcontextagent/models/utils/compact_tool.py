"""Shared native tool schema and action parsing."""

import json
import re

CONTEXT_TOOLS = ("compact_block", "expand_block")


def canonical_tag(tag) -> bool:
    return (
        isinstance(tag, str) and re.fullmatch(r"ctx-(?:[0-9]{4}|[1-9][0-9]{4,})", tag) is not None and tag != "ctx-0000"
    )


COMPACT_FUNCTION = {
    "name": "compact_block",
    "description": (
        "Replace a long context range with a summary produced by a separate agent of the same model. "
        "Supply only start_quote and end_quote; the compaction agent receives history from its beginning "
        "through the selected end, with no later messages. The tool waits for summarization and a committed "
        "replacement before returning. Use exactly one compact_block call and no other tools. "
        "Choose unique literal excerpts from the first and last complete messages in the active request, "
        "excluding this response. Select complete tool exchanges, including all calls, reasoning and results. "
        "Preserve instructions, the latest user request and latest model exchange. Avoid compacting short information. "
        "Earlier tagged memories and expanded history may be selected. Recompaction archives only local messages "
        "plus direct references. Use full exp/part headers to select repeated expansions at original-message boundaries. "
        "Defaults require 2000 visible text characters selected and 512 characters net request reduction, "
        "including this call and receipt. The archive is local; only the generated summary remains in active "
        "context until expand_block retrieves local history."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "start_quote": {"type": "string", "description": "Unique excerpt from the first selected message"},
            "end_quote": {"type": "string", "description": "Unique excerpt from the last selected message"},
        },
        "required": ["start_quote", "end_quote"],
        "additionalProperties": False,
    },
}
COMPACT_TOOL = {"type": "function", "function": COMPACT_FUNCTION}
COMPACT_TOOL_RESPONSE_API = {"type": "function", **COMPACT_FUNCTION}
EXPAND_FUNCTION = {
    "name": "expand_block",
    "description": (
        "Retrieve a tagged archive from this run, keeping its summary in place. "
        "Use exactly one expand_block call and no other tools in a dedicated response. "
        "Returns complete local historical messages, never executes archived calls, and never follows references. "
        "Use exp/part markers as unique quote selectors when recompacting; select complete original exchanges. "
        "A fresh call may retrieve the same tag again. Expand referenced tags separately when needed."
    ),
    "parameters": {
        "type": "object",
        "properties": {"tag": {"type": "string", "description": "Canonical current-run tag, such as ctx-0001"}},
        "required": ["tag"],
        "additionalProperties": False,
    },
}
EXPAND_TOOL = {"type": "function", "function": EXPAND_FUNCTION}
EXPAND_TOOL_RESPONSE_API = {"type": "function", **EXPAND_FUNCTION}


def parse_native_calls(calls: list[dict]) -> list[dict]:
    """Normalize both API formats, retaining a result slot for every rejected call."""
    context_tool = next((call["name"] for call in calls if call["name"] in CONTEXT_TOOLS), None)
    if context_tool and len(calls) != 1:
        return [
            {
                "type": "rejected",
                "tool_call_id": call["id"],
                "error": f"{context_tool} requires exactly one call and no other tools. No actions executed.",
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
        required = {"compact_block": ["start_quote", "end_quote"], "expand_block": ["tag"]}.get(name, ["command"])
        if name not in ("bash", *CONTEXT_TOOLS):
            error += f"Unknown tool '{name}'."
        if not isinstance(args, dict) or any(
            not isinstance(args.get(key), str) or (name in CONTEXT_TOOLS and not args[key]) for key in required
        ):
            error += (
                "Missing 'command' argument in bash tool call."
                if name == "bash"
                else f"Missing or invalid {', '.join(required)} argument in {name} tool call."
            )
        if name in CONTEXT_TOOLS:
            if isinstance(args, dict) and set(args) != set(required):
                error += f"Use only {', '.join(required)} arguments."
            if name == "expand_block" and (not isinstance(args, dict) or not canonical_tag(args.get("tag"))):
                error += "Use a canonical tag such as ctx-0001 from this run."
            actions.append(
                {
                    **({key: args[key] for key in required if key in args} if isinstance(args, dict) else {}),
                    "type": name,
                    "tool_call_id": call["id"],
                }
            )
            if error:
                actions[-1]["error"] = error
        elif error:
            raise ValueError(error)
        else:
            actions.append({"command": args["command"], "tool_call_id": call["id"]})
    return actions
