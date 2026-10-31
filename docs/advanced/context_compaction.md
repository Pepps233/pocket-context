# Agent-controlled context compaction and expansion

Native tool models can call `compact_block` to replace a long, contiguous context range with an agent-written memory.
The calling agent selects the bounds, then waits for a separate compaction agent using the same model to generate the summary.
The compaction agent has an independent conversation and cannot execute tools.
Compaction and expansion each take a dedicated turn with exactly one call and no other tools.
Text-based bash models keep their existing syntax.

Pocket Context offers `compact_block` when estimated input reaches `compact_pressure_fraction * max(0, context_token_limit - output_allowance)`.
The default fraction is `0.5`: a 64,000-token context with a 16,384-token output allowance has a 23,808-token threshold.
The pressure check excludes the generated catalog and the unavailable compaction schema, so their overhead cannot activate the tool.
Input uses the configured tokenizer when available, otherwise the existing serialized-byte estimate; it is not an exact provider token count.
Availability is recalculated from the reduced history after compaction, so the tool and eligible-block catalog disappear below the threshold and return as input grows.
Set `compact_pressure_fraction=0` to disable gating; an unlimited context also leaves compaction available.
Archive expansion remains available below the threshold, with a separate compact archive-tag list when archives exist.
The working prompt encourages batching completed investigation and obsolete outputs into substantial ranges while retaining current reasoning and recent evidence.
It discourages repeatedly summarizing the same memory without substantial new information.
These pressure and prompt changes apply only to Pocket Context, not ReAct, ReSum, or ACM.

```json
{
  "name": "compact_block",
  "arguments": {
    "start_id": "b001",
    "end_id": "b004"
  }
}
```

When eligible history exists and compaction is available, the request ends with a fresh block catalog in history order.
Each stable ID identifies a complete model exchange, including reasoning, every tool call, and every result, or one standalone memory or older request.
Each row shows its current chronological position, stable `block_id`, type, and associated `archive_tag` when present.
Positions may change after compaction; block ID numbers do not indicate chronological order.
Pass `block_id` values to `start_id` and `end_id`, following the displayed positions rather than numerical ID order.
For example, if position 1 is `b003` and position 2 is `b002`, selecting `b003` through `b002` is valid.
Protected gaps are marked in the catalog, and a selected range must not cross them.
The framework resolves the inclusive range and rejects unknown, stale, or protected IDs.
IDs remain stable while their messages remain active; compacted IDs disappear and expanded exchanges receive fresh IDs.
Errors list eligible IDs or identify the required complete exchange.
Reversed selections report both endpoint positions and require an explicit retry in catalog order; endpoints are never swapped automatically.
Instructions, the latest genuine user request, and the latest model response and its results are protected.
A rejected selection returns a tool error and leaves active context intact.
A mixed tool batch rejects every call before executing any command.

During migration, callers may still supply exactly `start_quote` and `end_quote` instead of IDs.
Quotes match literal visible text, serialized tool arguments, or decoded argument strings, excluding earlier memory-control arguments and hidden metadata.
Matching is deterministic: each quote must identify exactly one message and both boundaries must include complete exchanges.
Mixing ID and quote selectors is rejected.

After selection validation, the compaction agent receives the entire active history prefix from the first message through the inclusive selected end.
Messages before the selected start provide interpretation and task context; only the selected range is summarized and replaced.
For history A, B, C, D and selection B through C, it receives A, B, C and never D or the triggering compaction call.
History is rendered as inert evidence, preserving visible reasoning, tool arguments, results, and supported images while excluding hidden metadata.
Context-management calls and their receipts are omitted from the summarizer input using tool-call metadata; their original records remain archived.
Existing tagged summaries remain summaries; offloading does not recursively expand their archives.
No task, hints, or instructions are derived from later messages.

A successful replacement appears at the original position as `[ctx-0001]: <summary>`, explicitly labeled as assistant-generated memory.
The next model request contains this replacement before the agent chooses further work.
The receipt reports the tag and measured character reduction.
It identifies the same archive as the earlier memory, not a second memory or a reused tag.
Summaries and expanded historical messages may be compacted again; earlier archives and their direct references remain available.

Pocket Context summaries use a decision record with sections for the current patch, established facts, failed approaches, test results, open questions, and next action.
Empty or inapplicable sections are omitted, and next actions must be supported by the selected history.
Summaries distinguish applied edits from proposals, observed test outcomes from unrun checks, and abandoned attempts from untested ideas.
Consequential decisions retain their reasons and supporting evidence so the agent can avoid repeating settled investigation.
The prompt targets 150-300 words while allowing extra space for essential patch or verification details; this is guidance, not a truncation limit.
These instructions apply to Pocket Context's compaction agent; the comparison methods retain their own summarization prompts.
Earlier context only resolves references; task instructions, answer formats, counters, context-management history, and exhaustive search logs are omitted from the summary.
An unsuccessful search does not rule out a lead or make its archive unhelpful.
Archive identity and provenance come from framework headers rather than generated prose.
Empty, tool-calling, truncated, or failed child responses return a diagnostic without changing active history, allowing the main agent to continue.
Internal formatting failures do not request new selection boundaries.
Adapters serialize child requests and expanded input content for their own API.
Structural validation cannot guarantee semantic completeness.
The main agent receives a receipt only after a successful summary passes the reduction check and the archive transaction commits.

## Expansion

Call `expand_block` with one canonical tag from the current run:

```json
{"name": "expand_block", "arguments": {"tag": "ctx-0001"}}
```

The catalog lists available archives separately from eligible live blocks, including archives whose summaries have been compacted again.
Choose an existing archive tag from that list; block IDs and chronological positions are not archive tags.
Ordinary live exchanges have no archive because their content is already visible.
An unknown archive tag returns the available archive tags for this run, or `none` when no archives exist.

Expansion keeps the existing summary in place and appends a tool receipt followed by the block's local archived messages in their original order.
It returns complete archived text and tool results, preferring saved raw output over truncated observation text.
If the expansion would exhaust the context budget, it returns an error before journaling or appending history instead of silently truncating it.
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

Expanded exchanges appear in the eligible block catalog with distinct IDs for each occurrence.
Selecting one complete exchange from a larger expansion leaves its other messages intact.
Legacy quote callers can use full occurrence headers to distinguish repeated expansions, while still selecting complete exchanges.

A fresh tool-call ID can expand the same tag again with a new occurrence marker.
Redispatching a committed expansion never appends another historical copy.
Invalid tags, missing current-run archives, and storage failures return actionable tool errors without appending historical content.
There is no cross-run lookup or resume feature.

## Provenance-aware recompaction

Compaction validates the selected range and reduction thresholds before storing a new block.
The generated summary stays unchanged, including conclusions drawn from inherited material.
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
  compact_response_token_limit: null
```

`context_db_path` defaults to `global_config_dir/contexts/<run_id>.sqlite3`.
A shared path keeps separate records for each run.
The minimum range counts visible text characters, excluding images and hidden metadata.
The minimum reduction compares serialized request characters before and after compaction, including the retained compaction response, replacement, receipt, updated block catalog, and rebuilt cache markers.
These limits are characters, not token estimates.
`compact_response_token_limit` defaults to the main agent's output allowance: the model's configured output limit, or `response_token_limit` (16,384 by default).
Set a positive value to override it only for summarization; the allowance is capped by the remaining context window.
It includes reasoning tokens when the provider counts them toward its output limit.

## Archives and trajectories

SQLite stores versioned `blocks`, committed replacement `events`, request manifests in `requests`, and a separate expansion journal in `operations`.
Version 2 blocks retain exact local message JSON, raw tool outputs, provider payloads, hidden IDs, original ordering, exchange relationships, direct reference metadata, and the originating tool-call ID.
Version 1 archives are read through a compatibility layer that recognizes their recorded child summaries and excludes those summaries from local expansion content.
Existing archives are never rewritten.
The archive and replacement event commit in one transaction before the active projection changes.
Successful expansions commit their receipt, historical projection, and provenance to the operation journal before appending messages.
Expansion does not advance compaction revisions or `ctx-NNNN` numbering.
Storage and optional-operation formatting failures return tool diagnostics without replacing history.
A corrupted active projection still stops the run.
Reprocessing a committed tool-call ID is idempotent, while conflicting arguments or reuse across context operations fail.

Trajectory JSON retains the full chronological `messages` field for existing inspectors.
Its additional `context` field records the archive path, run ID, revision, active IDs, message and expansion provenance, structured item IDs, replacement events, expansion operations, and block relationships.
Internal message and item IDs live in sidecar records and are never sent to providers.
Public block IDs and their membership are recorded in request manifests; the catalog is generated request metadata, not user history.
Reasoning signatures and multimodal payloads remain unchanged, and cache markers are rebuilt on prepared copies.

Compaction-agent requests, responses, and failure diagnostics are saved separately in trajectory `compactions`; they never become main-agent conversation messages.
Their costs count toward the episode cost, and `model_stats.api_calls` includes both agents while `compaction_api_calls` identifies offloaded calls.
The main agent's step limit counts only its own decisions.
Each archive records `summary_input_ids` to audit the masked prefix used for generation.
Committed calls are replayed without regenerating a summary; a new call ID creates a fresh compaction attempt.
