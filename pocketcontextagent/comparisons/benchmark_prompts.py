"""Coding-domain adaptations; memory algorithms remain the versioned comparison ports."""

RESEARCH_MODES = {"browsecomp-plus": "corpus", "swebench-verified": "bash"}

MEMORY_INSTRUCTIONS = {
    "pocket_context": """Manage context to support accurate task completion.
Keep details available while useful for reasoning, verification, or upcoming work.
Use compact_block when replacing older material with a summary would meaningfully help you continue; select IDs from the current Eligible context blocks catalog.
Use expand_block when you need information missing or insufficiently detailed in a summary.
Prefer completing the task directly when existing context is sufficient.""",
    "acm": """Your active conversation is short-term memory; compressed segments are archived long-term memory.
manage_context takes no arguments. It compresses everything since the previous manage_context boundary, excluding the assistant message issuing this call. System instructions and the original task remain preserved.
The result contains [summary_id: N]. Use query_memory(summary_id, query) to extract relevant details from that archive's original messages.
When context grows with dead ends or duplicates, call manage_context to free space. Continue the investigation after memory operations; they do not submit an answer.""",
}


def benchmark_prompt(benchmark: str, method: str, window: int) -> str:
    if benchmark != "swebench-verified":
        raise ValueError(f"No domain prompt for {benchmark}")
    return "\n\n".join(
        [
            """You are a software engineering agent fixing the issue in the supplied repository.
Use bash(command) to inspect, edit, and test the checkout in /testbed. Commands run in fresh shells; file changes persist.
Implement a general fix and verify it with relevant tests. Do not fetch reference patches or alter evaluation infrastructure. Do not commit changes. Remove temporary reproduction artifacts before submitting.
When finished, submit with a successful shell command whose first output line is COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT, for example: echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
The framework captures the actual repository diff; prose is not a patch.
If instructed to conclude without tools, respond with Final Answer: followed by a brief explanation. The framework still captures the working-tree patch.""",
            f"Your active context window is {window} tokens. Read [CURRENT CONTEXT TOKEN: N] markers; do not generate them. When instructed to conclude, stop using tools and provide your best final response.",
            MEMORY_INSTRUCTIONS.get(
                method, "Keep investigating until the task is completed or a stopping instruction arrives."
            ),
        ]
    )


def benchmark_stop_prompt(benchmark: str) -> str:
    if benchmark != "swebench-verified":
        raise ValueError(f"No domain stop prompt for {benchmark}")
    return "The task must now conclude. Do not call tools or manage memory. Review available evidence and respond with Final Answer: followed by a brief description of the changes. This is your final generation."
