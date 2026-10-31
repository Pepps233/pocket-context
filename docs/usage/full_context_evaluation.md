# Full context evaluation: prepare, do not launch

This is the full-run handoff, **not authorization to launch inference or grading**.
Preparation and local code/configuration checks make zero generation calls. Live
infrastructure checks require separate permission and can incur cloud charges.
Do not infer live full-scale readiness from these instructions or a successful dry run.
See [controller operations](benchmark_controller.md), [artifact storage](controller_volumes.md)
and [method fidelity](context_comparison.md) for supporting workflows.

## Frozen allocation and shared policy

| Benchmark | Selection | Tasks | Attempts across all four methods |
| --- | --- | ---: | ---: |
| BrowseComp-Plus | Entire `eval150`, seed 42 | 150 | 600 |
| SWE-bench Verified | `test`, seed-42 selection | 125 | 500 |
| Total | Two separate cohorts | 275 | **1,100** |

Methods are `pocket_context`, `react`, `resum` (training-free) and `acm` (ACM Base).
Each gets identical selected IDs within its benchmark. Preserve the selection, code,
configuration and prompt fingerprints; never substitute tasks, retry failed attempts,
or cherry-pick successful shards. Failed attempts remain in the full denominator.
**These are not unseen cohorts:** 17 BrowseComp IDs and five SWE tasks have prior
pilot exposure. Disclose that exposure rather than describing this as a fresh holdout.

Layer `context_full_browsecomp` **after** `browsecomp_plus_deepseek`, and
`context_full_swebench` **after** `context_benchmark_deepseek`. The full layers override
historical smoke defaults; the main-call cap is **not 100**.

| Shared setting | BrowseComp-Plus | SWE-bench Verified |
| --- | --- | --- |
| `agent.step_limit` | `0` (no arbitrary main-step cap) | `0` |
| `agent.cost_limit` | USD 1 per attempt, including auxiliary calls | Same |
| `agent.wall_time_limit_seconds` | `1800` | `7200` |
| Context / response allowance | 128,000 / 16,384 tokens | Same |
| Model | `deepseek/deepseek-flash`, direct DeepSeek API | Same |

Existing method-specific memory policies remain unchanged: no new ReAct compaction,
ReSum trigger, ACM boundary or Pocket Context policy is introduced. Keep the existing
95% context-pressure final-stop behavior, including append-only ReAct. Baseline ports
use current-history pressure; Pocket Context retains its projected-window guard.
Retain the three-consecutive-format-error termination rule. Managed model-call retry
configuration is two wrapper transport attempts, **not two new benchmark attempts**.
Full profiles set a 180-second model request timeout and `num_retries=0` in the
adapter, avoiding an intentionally configured second layer of adapter retries.

BrowseComp retrieval stays fixed: `k=10`, new/seen snippets **512/128 tokens**, document
cap **8,192 tokens**, tokenizer `Qwen/Qwen3.5-9B`, revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`. Retrieval token counts are not a claim of
exact DeepSeek chat-template accounting. Freeze the dataset/index revisions as well.
The controller runtime pins the MCP-enabled inference stack and Transformers/tokenizers;
verify the installed versions and import-only checks against `controller_runtime.py`,
not just a successful provider metadata request.

## Local plans: zero API calls and zero generation

Run from the intended checkout with prepared local datasets. These commands only print
plans: no uploads, workers, provider requests, inference or grading. Inspect the exact
selected IDs and effective configurations, not merely the reported attempt totals.

```bash
mini-extra benchmark-controller submit \
  --benchmark browsecomp-plus --split eval150 \
  --data-dir data/browsecomp-plus --limit 150 --seed 42 \
  --workers-per-method 12 \
  -c browsecomp_plus_deepseek -c context_full_browsecomp \
  -c runs/context-evaluation/full-preparation/browsecomp-assets.yaml \
  --corpus-snapshot pocket-browsecomp-plus-20261022 --dry-run

mini-extra benchmark-controller submit \
  --benchmark swebench-verified --split test \
  --data-dir data/swebench-verified --limit 125 --seed 42 \
  --workers-per-method 12 \
  -c context_benchmark_deepseek -c context_full_swebench \
  -c runs/context-evaluation/full-preparation/swebench-images.yaml --dry-run
```

The BrowseComp asset layer binds the exact corpus snapshot UUID and all 15 index-file
SHA-256 hashes. Preflight checks the index contents; worker creation uses the resolved
snapshot UUID, not a mutable name alone. The exact tokenizer revision must be cached on
the controller before launch; managed jobs load HF assets offline.

The SWE image layer must contain **all 125 selected canonical image repositories with
resolved immutable digests**, not mutable `:latest` tags or arbitrary replacements.
Preserve its resolution record and hash; missing or mismatched digests block readiness.
The preparation owner records plans, checks, exposure and remaining blockers under
`runs/context-evaluation/full-preparation/`, with a final `ready.json` handoff.
That report must distinguish local checks, actual live checks and unverified assumptions;
a planned report is not evidence that those checks passed.

After future explicit launch approval, use the same commands **without `--dry-run`**
and with the approved controller manifest (`--manifest PATH` if not the default).
Do not remove the flag during preparation. `--preflight` means infrastructure-only,
not read-only: a non-dry submission can allocate workers and refresh credentials even
though it performs no generation. No inference or grading is launched by preparation.

## Sequential execution and capacity

Run BrowseComp and SWE **sequentially**, with one active job per controller. Twelve
workers per method means at most **48 concurrent corpus or coding workers**, not 48
attempts total. Provision worker pools only at authorized launch; retain the fixed
BrowseComp base snapshot `pocket-browsecomp-plus-20261022` rather than replacing it.

| Peak allocation | CPU | RAM (GiB) | Disk (GiB) |
| --- | ---: | ---: | ---: |
| 48 SWE workers, each 2 CPU / 8 GiB / 10 GiB | 96 | 384 | 480 |
| 48 BrowseComp workers, each 1 CPU / 4 GiB / 10 GiB | 48 | 192 | 480 |
| Trusted controller | 2 | 8 | 10 |
| SWE peak including controller | **98** | **392** | **490** |

Against a 250-CPU / 500-GiB-RAM / 2,000-GiB-disk allocation, the SWE plan leaves
**108 GiB RAM**, before other resources. Thirty workers per method would require
960 GiB worker RAM and is not feasible within that RAM quota. These are provisioning
arithmetic, not proof of available capacity, image fit or controller memory sufficiency.
Verify organization quotas, existing allocations, snapshot limits and actual lifecycle
settings before approval; 48 simultaneous model histories still need runtime headroom.

SWE task timeout is two hours; each coding worker retains **240-minute absolute TTL**
and **120-second command timeout**. The managed BrowseComp helper derives pool TTL as
`max(240, ceil(tasks / active_workers_per_method) * 30 + 120)` minutes:
`ceil(150 / 12) * 30 + 120 = 510` minutes for this plan. The full profile explicitly
sets 510 minutes so the small preflight verifies that same server policy; reducing
concurrency requires re-evaluating the TTL, not silently retaining an insufficient one.
Check the helper and actual
server deadline; do not assume the historical 240-minute minimum is sufficient.
Controller TTL is **0**, with idle stop/pause disabled. Eleven worst-case SWE waves
at two hours each imply **22 hours**, before setup/collection overhead; this estimate
is not a hard global job timeout.

## Trust, collection and recovery

The managed BrowseComp path keeps agents, memory and provider/Daytona credentials on
the trusted controller. Credential-free corpus VMs provide only `search`/`get_document`
through SDK-mediated retrieval. SDK process execution implements that protocol; it is
**not an agent shell**. No provider tokens, gold answers, agent loop or artifact-volume
mount belong on corpus VMs. SWE coding VMs likewise get public task inputs and coding
tools only, never provider credentials or private gold patches/tests.

Use a controller-only artifact volume. Keep **live SQLite, locks, credentials and private
grading inputs off FUSE**; local controller disk remains necessary. Closed task archives
are published and verified before local bulk trajectories are pruned. Collection is
automatic: preserve root selection/results, method summaries, SWE predictions, task
receipts and the committed volume job bundle. Archive success does not imply workload
success. Keep failed attempts and recovery journals; never rerun to repair collection.
Recorded budget/format exits remain attempts and do not silently retry. Unhandled
provider, transport, retrieval or infrastructure exceptions are fail-fast controls:
the phase can stop with unstarted IDs still present in its frozen selection. Preserve
partial results and report missing-attempt coverage; do not claim 1,100 completed
attempts merely because 1,100 were allocated.

The default local low-disk floor is **1 GiB** (`--min-free-bytes 1073741824`). It is an
admission check, not a per-active-case reservation or hard quota. Concurrent histories,
compression spool, small retained results and caches can still exhaust the 10-GiB disk.
Verify archive receipts, capture completeness and cleanup before pruning/deleting anything.

Detached cloud execution survives laptop shutdown or SSH disconnect, **not controller
stop, crash, OOM or loss**. A stopped controller does not continue the job, and restarting
does not automatically resume/retry attempts. Inspect ambiguous acceptance and retained
worker state instead of resubmitting. Keep independent exports of important artifacts.

## Budget approval and separate grading

The configured model allocation is **USD 600 BrowseComp + USD 500 SWE = USD 1,100**,
including auxiliary memory calls. It is **not an exact hard maximum bill**: thresholds
are checked between calls, so the final in-flight call can overshoot. Daytona compute,
storage and separate judge/grading costs are excluded. Future launch approval must
explicitly cover the full allocation, overshoot risk and infrastructure costs.

Grading is a separate authorized phase, not a side effect of preparation or submission:

- **BrowseComp:** official grading uses `Qwen/Qwen3-32B` with vLLM on a suitable Linux
  GPU host. Prior GPT-4.1 judging is an explicitly alternate grader, not official Qwen
  evidence; label it separately and do not silently merge grader results.
- **SWE:** use the [Daytona evaluation adapter](context_benchmarks.md#swe-bench-verified-official-harness), official harness **v4.1.0**, commit `726c5461e2ef52d83cf1ea2107870a8bb3328d57`, and the pinned private grading dataset.
  The trusted launcher runs the official parser while fresh Daytona grading sandboxes execute the official test scripts.
  The adapter records the resolved grading-image digest and requires no local Docker daemon.
- **Prepared inference plans do not mean official grading is ready.**
  Validate the adapter with a live smoke test before claiming operational readiness or correctness.

Retain all selected attempts when reporting accuracy, and disclose missing grades,
coverage, budget/time exits and prior exposure. Submitted answers or captured patches
are not correctness evidence. Compare benchmarks separately; do not pool their scores.
