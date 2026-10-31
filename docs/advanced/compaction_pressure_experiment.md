# Pocket Context pressure and decision-summary experiment

This smoke test checks Pocket Context's tool availability, catalog suppression, and decision-focused summary format.
ReAct, ReSum, and ACM prompts and method implementations were unchanged, and regression checks verify their context accounting remains unchanged.

## Setup

The task was sampled uniformly from the 125 previously prepared SWE-bench Verified public tasks using seed `3797013628`.
The selected task was `django__django-11490`.
All three autonomous attempts used DeepSeek Flash, the same public task, and its pinned official image in a fresh Daytona sandbox.
The dataset revision was `c104f840cc67f8b6eec6f759ebc8b2693d585d4a`.
The task image digest was `sha256:efe0a8e57dfa92d296fb017ad7931d842b506ad05ef7582195b4bd9a890c3a9e`.
Each attempt had a 100-step limit, a 1 USD soft model-cost limit, and a 1,200-second task limit.

Compaction becomes available at half of `max(0, context_window - output_allowance)`.
The gate uses estimated input without the generated catalog or compaction schema; provider-reported input is recorded separately.
The eligible-block catalog is shown while compaction is available, and archive retrieval remains available below the threshold.
Availability is recalculated after every context change, including compaction and expansion.

## Autonomous observations

| Context window | Output allowance | Pressure threshold | Peak provider input | Requests offering compaction | Autonomous compactions | Forced final generation |
|---:|---:|---:|---:|---:|---:|---|
| 64,000 | 16,384 | 23,808 | 11,835 | 0 / 13 | 0 | No |
| 16,000 | 4,096 | 5,952 | 13,390 | 9 / 14 | 0 | Yes |
| 8,000 | 2,048 | 2,976 | 6,113 | 5 / 7 | 0 | Yes |

Availability counts come from the per-request manifests, including requests without a successful-response budget annotation.
The 64K run kept compaction unavailable and completed with ample context remaining.
The smaller-window probes activated the tool, but the agent did not choose to compact before forced final generation.
All attempts produced captured patches and confirmed cleanup of their owned Daytona sandboxes and snapshots.
Submission and patch capture are not official SWE-bench correctness judgments; these attempts were not officially graded.

These observations verify the availability gate, but do not demonstrate that the batching prompt causes spontaneous compaction or improves accuracy.

## Controlled decision-summary check

A separate operator-directed compaction used an active checkpoint from the 16K Daytona trajectory, below its forced-final threshold.
It selected blocks `b001` through `b010`, preserving the two most recent exchanges.
This was a controlled summarizer test on real task history, not an autonomous agent decision or another benchmark attempt.

The final summary contained 218 whitespace-delimited words and all six nonempty sections in the requested order.
It retained the applied compiler edit, the shared-query mutation mechanism, the reproduction results, targeted test results, the failed test-module selection, open questions, and next action.
The checks prompted an additional instruction to omit empty-section filler and distinguish failed investigative steps from the original bug.
The main guidance also states that newer edits and test results take precedence over historical summaries.

Estimated request input fell from 14,802 to 3,977 tokens after compaction.
With the 5,952-token pressure threshold, `compact_block` and the eligible-block catalog became unavailable, while `bash`, `expand_block`, and the archive-tag list remained available.
This is one checkpoint's observed behavior, not a cohort-level comparison with the previous 482-word median.

## Validation and artifacts

The regression suite covers threshold boundaries, tool and catalog hiding, expansion-driven reactivation, protected history, committed archives, and compaction accounting.
Real local HTTP clients verify filtered tool serialization for the main Responses adapter, Portkey, Portkey Responses, and Requesty without mocking provider calls.
The final validation passed 546 tests, with lint and formatting checks passing for the changed Python files.

The autonomous attempts used implementation commit `2e22ec8`; the final controlled summary used `f01c679`.
Local plans, trajectories, cleanup journals, observation JSON, and the final summary are saved under `runs/context-evaluation/compaction-pressure-smoke-20261031/`.
The final summary and its measurements are in `controlled-summary-final/summary.txt` and `controlled-summary-final/observation.json` within that directory.
The local `analyze.py` script regenerates the autonomous-run observations from the saved trajectories.
