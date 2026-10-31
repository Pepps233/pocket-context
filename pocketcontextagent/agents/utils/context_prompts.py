"""Pocket Context's shared working-memory guidance."""

COMPACTION_GUIDANCE = """Continue solving while context has sufficient room.
Use compact_block only when it is offered and compaction is needed; combine completed investigation and obsolete outputs into one substantial range.
Preserve current reasoning and recent evidence.
Treat summaries as historical snapshots; newer edits and test results take precedence.
Avoid repeatedly summarizing the same memory unless substantial new information has accumulated.
Select block IDs from the current eligible-block catalog.
Use expand_block when details missing from an archived summary are needed."""
