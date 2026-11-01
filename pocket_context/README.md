# Pocket Context library

`pocket_context/` is a small, provider-neutral library for agent builders.
It stores messages in the order they arrive, lets your own summarizer replace a range with a summary, and keeps the removed messages in a SQLite archive for later expansion.

It has no model SDK dependencies and accepts ordinary message dictionaries, so the agent loop remains responsible for calling the model and generating summaries.

## Quickstart

```python
from pocket_context import ContextStore

with ContextStore("agent-context.sqlite3") as context:
    context.record({"role": "user", "content": "Investigate the failing test."})
    context.record({"role": "assistant", "content": "The failure comes from the parser."})

    # Supply a summary produced by your own model or summarizer.
    tag = context.compact(0, 1, "The parser causes the test failure.")
    messages_for_next_model_call = context.assemble()

    # Restore the original messages when their detail is needed again.
    context.expand(tag)
```

## API

- `record(message)` appends a message and returns its stable message ID.
- `compact(start, end, summary)` replaces the inclusive, zero-based range in the active projection and returns an archive tag.
- `expand(tag)` replaces an active summary with the original messages and returns those messages.
- `assemble()` returns a copy of the active message list.

The summary entry is a dictionary with `_context_block`, `role`, and `content` keys.
Map it to your provider's preferred memory-message shape before sending it to a model.
Archive data is durable in SQLite, while the active projection is held in the `ContextStore` instance for the lifetime of that instance.
