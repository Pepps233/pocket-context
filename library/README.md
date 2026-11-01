# Pocket Context Python library

Install from the repository root with `pip install -e .`.
The import remains `pocket_context` after installation.

```python
from pocket_context import ContextStore


def compress(request):
    return your_agent(request.messages)  # Adapt your agent to return a summary string.


with ContextStore("context.sqlite3", compact_at_percent=50) as pocket:
    pocket.record({"role": "system", "content": "You are a research agent."})
    pocket.record({"role": "user", "content": "Earlier question"})
    pocket.record({"role": "assistant", "content": "Earlier answer"})
    pocket.record({"role": "user", "content": "Current task"}, protected=True)

    tools = pocket.tools(used_tokens=4500, context_token_limit=10000, response_token_limit=1000)
    tag = pocket.compact(1, 2, agent=compress)
    next_model_messages = pocket.assemble()
    archived_messages = pocket.expand(tag)
```

`record()` adds messages; `assemble()` returns the current projection for the next model call.
`compact()` calls your compression agent with the default Pocket Context prompt and archives the selected inclusive range.
To customize its prompts, pass `system_prompt=` and/or `user_prompt_builder=` to `compact()`.
Use `await pocket.acompact(...)` for an asynchronous agent.
`expand()` returns archived messages without removing the summary; format the result as historical evidence for your model and record that result.

`pocket.tools()` returns provider-neutral tool definitions.
`compact_block` appears at the configured context usage percentage, 50% by default; `expand_block` stays available after an archive exists.
Your agent loop supplies token counts, adapts tool definitions to its model provider, and validates complete tool exchanges before dispatching compaction.
