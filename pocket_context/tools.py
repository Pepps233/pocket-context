"""Provider-neutral Pocket Context tool definitions."""

COMPACT_TOOL = {
    "name": "compact_block",
    "description": (
        "Choose an inclusive zero-based range of active messages for a separate compression agent. "
        "Do not include system instructions, the current task, or an incomplete tool exchange."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "start": {"type": "integer", "minimum": 0},
            "end": {"type": "integer", "minimum": 0},
        },
        "required": ["start", "end"],
        "additionalProperties": False,
    },
}

EXPAND_TOOL = {
    "name": "expand_block",
    "description": "Restore an archived range by its context tag when its full detail is needed.",
    "parameters": {
        "type": "object",
        "properties": {"tag": {"type": "string"}},
        "required": ["tag"],
        "additionalProperties": False,
    },
}
