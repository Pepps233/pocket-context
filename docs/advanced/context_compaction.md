# Agent-controlled context compaction

Native tool models can call `compact_block` to replace a long, contiguous context range with an agent-written memory.
The calling agent decides when to compact and writes the summary itself.
Compaction takes a dedicated turn with exactly one call and no bash actions.
Text-based bash models keep their existing syntax.

```json
{
  "name": "compact_block",
  "arguments": {
    "start_quote": "Unique excerpt from the first selected message",
    "end_quote": "Unique excerpt from the last selected message",
    "summary": "Information needed to continue working"
  }
}
```

Quotes match literal visible text or tool arguments in the request that produced the call.
They never match hidden provider metadata or the new compaction response.
Each quote must identify one message uniquely, and the inclusive range must contain complete model exchanges, including reasoning, every tool call, and every result.
Instructions, the latest genuine user request, and the latest model response and its results are protected.
A rejected selection returns a tool error and leaves active context intact.
A mixed tool batch rejects every call before executing any command.

A successful replacement appears at the original position as `[ctx-0001]: <summary>`, explicitly labeled as assistant-generated memory.
The next model request contains this replacement before the agent chooses further work.
The receipt reports the tag and measured character reduction.
Summaries may be compacted again; earlier archives and their child references remain available.

The tool instructions require task intent, constraints, paths, identifiers, command purposes and outcomes, uncertainty, and causal detail needed by surviving references.
Structural validation cannot guarantee semantic completeness, so summary quality remains the calling agent's responsibility.

## Configuration

These settings belong under `agent`:

```yaml
agent:
  context_db_path: /path/to/contexts.sqlite3
  compact_min_chars: 2000
  compact_min_reduction: 512
```

`context_db_path` defaults to `global_config_dir/contexts/<run_id>.sqlite3`.
A shared path keeps separate records for each run.
The minimum range counts visible text characters, excluding images and hidden metadata.
The minimum reduction compares serialized request characters before and after compaction, including the retained compaction response, replacement, receipt, and rebuilt cache markers.
These limits are characters, not token estimates.

## Archives and trajectories

SQLite stores versioned `blocks`, committed replacement `events`, and request manifests in `requests`.
Blocks retain exact selected message JSON, raw tool outputs, provider payloads, hidden IDs, original ordering, exchange relationships, child block tags, and the originating tool-call ID.
The archive and replacement event commit in one transaction before the active projection changes.
Storage failures return tool errors; unexpected context assembly errors stop the run.
Reprocessing a committed tool-call ID is idempotent, while conflicting arguments fail.

Trajectory JSON retains the full chronological `messages` field for existing inspectors.
Its additional `context` field records the archive path, run ID, revision, active IDs, message provenance, structured item IDs, replacement events, and block relationships.
Hidden IDs live in sidecar records and are never sent to providers.
Reasoning signatures and multimodal payloads remain unchanged, and cache markers are rebuilt on prepared copies.
