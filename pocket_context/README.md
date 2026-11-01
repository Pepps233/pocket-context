# Pocket Context library

`pocket_context/` provides a Python context store for agent builders.
It keeps an active message projection, calls a separate compression agent for selected history, and archives the original messages in SQLite.
The compression agent may use any model provider if you adapt its call to the small request interface below.
Replace `your_agent` in the example with a call to your own SDK or agent that returns a string.

## Quickstart

```python
from pocket_context import ContextStore


def compressor(request):
    return your_agent(request.messages)  # Return the agent's complete summary as a string.


with ContextStore("agent-context.sqlite3", compact_at_percent=50) as pocket:
    pocket.record({"role": "system", "content": "You are a coding agent."})
    pocket.record({"role": "user", "content": "Earlier task: investigate the failing test."})
    pocket.record({"role": "assistant", "content": "The failure came from the parser."})
    pocket.record({"role": "user", "content": "Now fix the regression."}, protected=True)

    # Pass these provider-neutral definitions to your model's tool adapter.
    visible_tools = pocket.tools(used_tokens=4500, context_token_limit=10000, response_token_limit=1000)

    tag = pocket.compact(1, 2, agent=compressor)
    messages_for_next_model_call = pocket.assemble()
    original_messages = pocket.expand(tag)
```

The default `compact_at_percent` is 50.
The threshold uses active input tokens divided by the context limit after reserving response tokens.
`compact_block` appears when that threshold is reached.
`expand_block` remains available whenever an archived summary is active, so an agent can recover details even after compaction reduces usage below the threshold.
The returned tool definitions use provider-neutral `name`, `description`, and `parameters` fields; map them to your provider's tool format.

## Compression agent and prompts

`pocket.compact(start, end, agent)` accepts a callable or an object with `run(request)`.
The agent receives a `CompressionRequest` with `history`, inclusive `start` and `end` indexes, `system_prompt`, `user_prompt`, and standard role/content `messages`.
It must return a non-empty plain-text summary.
An asynchronous agent can be used with `await pocket.acompact(start, end, agent)`.
The store keeps the original history unchanged if the agent raises or returns an invalid summary.

The default system instructions are exported as `COMPRESSION_SYSTEM_PROMPT` and match the existing Pocket Context compaction agent.
The default user prompt includes history up to the selected end and marks which messages are selected.
Both prompts can be customized for a task or provider:

The next example runs inside an open `ContextStore` context manager.

```python
def research_history_prompt(history, start):
    return "Summarize the selected research evidence:\n" + "\n".join(
        f"{index}: {message['content']}" for index, message in enumerate(history) if index >= start
    )


tag = pocket.compact(
    1,
    2,
    agent=compressor,
    system_prompt="Preserve citations, decisions, and unresolved questions.",
    user_prompt_builder=research_history_prompt,
)
```

## Message flow

`record(message)` appends a message to Pocket Context's trajectory.
System and developer messages are protected automatically; use `record(message, protected=True)` for any other message that must not be compacted.
`assemble()` returns a copy of the active projection to send to the next model call.
These correspond to `ingest()` and `assemble()` in the repository's full agent implementation: a store needs a way to receive new messages and a way to return the compacted request.
`compact(start, end, agent)` replaces an inclusive, zero-based active range with an archived summary.
`expand(tag)` restores that range in place and returns the originals.
Builders must group complete tool exchanges and protect the current task before dispatching compaction tool calls.

The summary entry has `_context_block`, `role`, and `content` keys.
Its content is labeled as assistant-generated memory rather than a new instruction.
Map it to your provider's preferred memory-message shape before sending it to a model.
The SQLite archive is durable, while the active projection lasts only for the `ContextStore` instance.
