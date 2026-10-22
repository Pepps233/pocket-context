# Agent-controlled context compaction and expansion

Native tool models can call `compact_block` to replace a long, contiguous context range with an agent-written memory.
The calling agent decides when to compact and writes the summary itself.
Compaction and expansion each take a dedicated turn with exactly one call and no other tools.
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
Summaries and expanded historical messages may be compacted again; earlier archives and their direct references remain available.

The tool instructions require task intent, constraints, paths, identifiers, command purposes and outcomes, uncertainty, and causal detail needed by surviving references.
Structural validation cannot guarantee semantic completeness, so summary quality remains the calling agent's responsibility.

## Expansion

Call `expand_block` with one canonical tag from the current run:

```json
{"name": "expand_block", "arguments": {"tag": "ctx-0001"}}
```

Expansion keeps the existing summary in place and appends a tool receipt followed by the block's local archived messages in their original order.
It returns complete archived text and tool results, preferring saved raw output over truncated observation text, without silently truncating the expansion.
Each original message becomes one historical input message with labeled roles, tool names, arguments, and results.
Archived calls are inert historical data and never execute.
Supported images remain input content, while opaque signatures and provider metadata stay in the original archive.
Historical input has explicit sidecar provenance and cannot become the latest genuine user request.

Summary and expansion headers list direct source tags in first-appearance order:

```text
[ctx-0001]: Original summary
[ctx-0002 refs=ctx-0001]: Summary using inherited material
[ctx-0005 refs=ctx-0001,ctx-0002,ctx-0004]: Summary using several sources
```

Every expanded original message has a unique selection marker:

```text
[ctx-0002 refs=ctx-0001]: (exp=3 part=2/4 role=tool)
<local archived content>
```

Use these full headers as quote selectors to distinguish original messages and repeated expansions.
Expanded content is selectable at original-message boundaries, and a selected original tool exchange must include its response and every result.
Selecting one complete exchange from a larger expansion leaves its other messages intact.
Quotes still must be unique across the active request, including earlier tool arguments that may repeat an excerpt.

A fresh tool-call ID can expand the same tag again with a new occurrence marker.
Redispatching a committed expansion never appends another historical copy.
Invalid tags, missing current-run archives, and storage failures return actionable tool errors without appending historical content.
There is no cross-run lookup or resume feature.

## Provenance-aware recompaction

Compaction validates the selected range and reduction thresholds before storing a new block.
The agent-written summary stays unchanged, including conclusions drawn from inherited material.
The new block stores only locally owned messages in its payload and records previous summaries and expanded messages as direct references.
Reference metadata retains source record IDs, expansion occurrences, original exchange groups, and selection order, including partial-block selections.
Ownership comes from sidecar provenance, never tag-shaped text or semantic similarity.
Newly written reasoning remains local even when it discusses earlier material.

For example, selecting expanded `ctx-0001` together with new local messages creates `ctx-0002` containing the new local messages and a reference to `ctx-0001`.
Its summary may describe both sources.
Expanding `ctx-0002` returns only its new local messages and lists `ctx-0001` without expanding it.

References stay direct and are deduplicated for display.
If `ctx-0003` summarizes `ctx-0002`, which references `ctx-0001`, the new header references only `ctx-0002`.
Expand each tag explicitly to reveal the next level.
Reference-only blocks are allowed when the normal compaction thresholds pass.
Their expansion reports `No local archived messages` and lists the referenced tags.

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

SQLite stores versioned `blocks`, committed replacement `events`, request manifests in `requests`, and a separate expansion journal in `operations`.
Version 2 blocks retain exact local message JSON, raw tool outputs, provider payloads, hidden IDs, original ordering, exchange relationships, direct reference metadata, and the originating tool-call ID.
Version 1 archives are read through a compatibility layer that recognizes their recorded child summaries and excludes those summaries from local expansion content.
Existing archives are never rewritten.
The archive and replacement event commit in one transaction before the active projection changes.
Successful expansions commit their receipt, historical projection, and provenance to the operation journal before appending messages.
Expansion does not advance compaction revisions or `ctx-NNNN` numbering.
Storage failures return tool errors; unexpected context assembly errors stop the run.
Reprocessing a committed tool-call ID is idempotent, while conflicting arguments or reuse across context operations fail.

Trajectory JSON retains the full chronological `messages` field for existing inspectors.
Its additional `context` field records the archive path, run ID, revision, active IDs, message and expansion provenance, structured item IDs, replacement events, expansion operations, and block relationships.
Hidden IDs live in sidecar records and are never sent to providers.
Reasoning signatures and multimodal payloads remain unchanged, and cache markers are rebuilt on prepared copies.
