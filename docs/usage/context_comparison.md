# Comparing context-management methods

Run **Pocket Context, ReAct, training-free ReSum, and ACM Base** on the same BrowseComp-Plus tasks, corpus and policy weights. The comparison reports success rate, total tokens, cost, latency and peak input context. It does not run paid evaluations automatically or contain benchmark performance results.

## Preserve the methods, share the evaluation plumbing

Published methods live separately from Pocket Context:

```text
pocketcontextagent/comparisons/
  methods/
    base.py             # Native transport, audit history and resource accounting
    react.py            # Append-only ReAct baseline
    resum.py            # ReSum threshold, full restart and recurrent summaries
    resum_prompts.py    # Pinned original summary/continuation prompts
    acm.py              # ACM Base compression boundaries and archive recall
    acm_prompts.py       # Pinned original policy, compressor and recall prompts
  tools.py              # Native corpus tools and original ACM memory schemas
  registry.py           # Method selection and source/settings provenance
  report.py             # Matched-run validation and judge imports
  upstream/             # Source pins, fidelity notes and upstream license notices
```

These are **versioned shared-harness ports**, not loose “X-style” implementations. They do not use Pocket Context's generic compressor or its block-selection interface. They also do **not** claim reproduction of published scores: backbone, corpus, tokenizer, transport, serving configuration and budgets are deliberately controlled here.

| Method | Preserved mechanism | Source |
| --- | --- | --- |
| `pocket_context` | Agent-selected `compact_block` ranges; exact archived-text restoration through `expand_block` | This repository |
| `react` | Append-only reasoning/action/observation loop; no memory tools or block catalog | ACM's released no-memory BCP ReAct baseline |
| `resum` | Original summary and continuation prompts; near-limit trigger; restart from original question plus summary, retaining **no** recent exchange | ReSum / DeepResearch `f72f75d8c3eb842f2bbbab096a12206ff66e270f` |
| `acm` | Original structured working-memory and recall prompts; argument-free interval compression; summary returned as a tool result; integer-ID, model-mediated recall | ACM `f06f90e728af8580a4515812425c1620144145a2` |

ReSum defaults to `0.9 * (context_window - 1000)` and a 4,096-token summary output cap. Its previous summary is passed separately when making the next summary. A successful restart leaves only system instructions and the summary-conditioned question in live history. Failed calls/storage resets preserve the preceding live history and summary. There is no retrieval primitive or ReSum-GRPO training.

ACM emits the original `<memory>` structure with **Knowledge state** and **Thoughts**, including concrete next steps. Compression has the released 4,096-token cap, one retry for malformed memory output, and final-attempt fallback. `query_memory(summary_id, query)` uses an integer ID, loads one raw archive, and follows the original recall prompt and normal per-turn output allowance. Earlier summary exchanges are excluded from subsequent compression. Archives are losslessly stored; recall is model-mediated, not guaranteed-perfect extraction or verbatim expansion.

### Necessary, explicit adaptations

- **ACM boundary choice:** the paper/tool description excludes the assistant requesting compression, while the released evaluation defaults to dropping it. This port follows the **paper-described exclusion**, retaining a valid native call/result pair. Mixed batches finalize the boundary after all sibling results to avoid splitting their exchange. This is not bit-for-bit reproduction of the release's `mc_drop_call_message=True` configuration.
- **Storage and wire format:** SQLite replaces per-summary JSON files; native Chat/Responses messages replace Qwen XML serialization. Prompts see visible reasoning, tool names/arguments and results, not private usage metadata.
- **Corpus setting:** all four explicit methods expose native `search(query)` and `get_document(docid)` backed by the same corpus. ReSum's original live-web `search`/`visit` stack is not used. Its main prompt is the published BCP ReAct prompt; its original summary/continuation prompts are retained.
- **Weights:** the same configured model/endpoint handles policy and auxiliary calls. ReSumTool-30B, ReSum-GRPO and ACM-Post-Trained are separate system configurations, not silently substituted into this same-policy comparison.
- **Pressure policy:** the baseline ports check current-history usage at 95% and do not silently truncate tool observations. Pocket Context retains its existing projected-window guard. These method-specific stop rules and the shared window must be reported; the defaults do not reproduce the authors' complete serving setups.

Detailed source-file provenance, transport/batching decisions and licenses ship in `pocketcontextagent/comparisons/upstream/SOURCES.md`. Copied prompts/schemas have offline hash-parity tests against the pinned originals. For the papers, see [ReSum](https://arxiv.org/html/2509.13313v3), [ACM](https://arxiv.org/html/2607.23809v1), and [BrowseComp-Plus](https://arxiv.org/html/2508.06600v1).

## Develop before touching the held-out set

Follow [BrowseComp-Plus setup](browsecomp_plus.md), including Java 21, the pinned corpus and model credentials. Tune only on `development680`.

The examples below use `browsecomp_plus_deepseek`: direct `deepseek/deepseek-flash` inference through LiteLLM at `https://api.deepseek.com`, using `DEEPSEEK_API_KEY` from your shell or ignored `.env`. No OpenRouter key is required. Thinking is enabled with high effort. The experimental cap remains 128,000 tokens even when the provider supports a larger window. The Qwen corpus tokenizer is deliberately retained for consistent preview/document caps; context estimates are not exact DeepSeek chat-template token counts, while measured usage comes from the provider.

Use this profile **instead of**, not after, `browsecomp_plus_openrouter`; merging provider profiles can retain incompatible routing options. DeepSeek changes the policy model, not just its credentials. All four methods must use this same profile and fresh output directories, separate from any Qwen/OpenRouter results. DeepSeek's model alias can change its served checkpoint; retain raw responses and record the evaluation date rather than assuming immutable weights. See [DeepSeek's current model documentation](https://api-docs.deepseek.com/) and [thinking/tool-call requirements](https://api-docs.deepseek.com/guides/thinking_mode).

```bash
for method in pocket_context react resum acm; do
  mini-extra browsecomp-plus run \
    --method "$method" --split development680 --limit 1 \
    -c browsecomp_plus_deepseek \
    --output "runs/browsecomp-plus/dev-comparison/$method"
done
```

These commands incur model costs. The comparison/reporting command below makes no model calls. Alternatively pass identical `--model`, `--model-class` and common YAML overrides to each run.

### Muse Spark 1.3 Contributor

Use `-c browsecomp_plus_muse_spark_1_3_contributor` **instead of** the DeepSeek profile to evaluate the exact `meta/muse-spark-1.3-contributor` model. It calls Meta directly at `https://api.meta.ai/v1`, loading `META_API_KEY` from your shell or ignored `.env`. It never substitutes `muse-spark-1.3` (Standard) or `muse-spark-1.2-contributor`. Keep separate output directories, for example `runs/browsecomp-plus/muse-1.3-contributor-dev/METHOD/`, and use the same profile for all four methods. Never layer provider profiles or mix their result directories.

The profile uses stateless Responses API requests with `store: false` and `include: [reasoning.encrypted_content]`, preserving opaque reasoning items during active tool-turn replay. Reasoning effort is `high`, not `max`, which Contributor does not support. Human-readable reasoning summaries are requested with `summary: auto` but may be absent; summarizers cannot read the encrypted reasoning itself. The same 128,000-token experimental window, main-call/output allowances, corpus limits and cost/time thresholds apply. Provider-reported usage includes reasoning tokens; estimated context uses the fixed corpus tokenizer and retained-reasoning accounting.

**Contributor is a data-contribution mode, not just a model name or discount.** Meta's [pricing and data-use documentation](https://ai.developer.meta.com/docs/pricing-rate-limits) says it may use your submitted prompts, code, files, context and returned responses to train future models. `store: false` disables server-managed response storage; it does not revoke Contributor training permission. Confirm that this is acceptable for your benchmark/data policy before submitting held-out tasks. Configuration and read-only model-catalog checks do not submit benchmark content. No automatic inference, account-mode changes or Standard-mode fallback are performed. See [Meta reasoning/replay behavior](https://ai.developer.meta.com/docs/reasoning).

`--method` is applied after YAML overrides. It selects the method's class, native research/memory schemas and, for baseline ports, canonical published policy/answer prompts. Literal prompt placeholders are preserved rather than accidentally interpreted as Jinja variables. Method-specific prompts are allowed to differ: forcing Pocket Context's prompt onto every baseline would change the methods being tested.

ReSum-specific settings are `agent.resum_trigger_ratio` (0.9), `agent.resum_headroom_tokens` (1000), and `agent.resum_summary_token_limit` (4096 maximum). Freeze any development-set changes before evaluation. Shared model routing, reasoning effort, generation settings, retrieval configuration and worker count should stay fixed. Main policy and auxiliary model settings are recorded; independent summarizer-model selection is not yet supported by this runner.

Cost and wall-time thresholds include auxiliary inference and are checked between calls, not hard billing caps. `agent.step_limit` counts main-agent calls; summaries/queries have separate counts and contribute fully to usage and cost. There is no additional hard cumulative-token cap. Repeat or interleave runs before interpreting small latency differences caused by provider load or local contention.

## Freeze and evaluate all 150 tasks

```bash
for method in pocket_context react resum acm; do
  mini-extra browsecomp-plus run \
    --method "$method" --split eval150 --workers 4 \
    -c browsecomp_plus_deepseek \
    --output "runs/browsecomp-plus/eval-comparison/$method"
done
```

A run may exit with status 1 when an attempt hits a budget. Inspect the retained failure record; keep it in the denominator. Missing task artifacts cause comparison to fail rather than evaluate only the intersection or successful tasks.

Explicit method runs and all local `eval150` runs retain **all** attempts on resume, including failures. They do not silently retry until success. `--redo-existing` explicitly replaces attempts; use new directories for independent repetitions, not best-of-reruns Pass@1. Legacy development runs without `--method` retain retry-incomplete behavior.

Manifests record selected IDs, source/index revisions, inventory, workers, attempt policy, implementation hash, and **per-method upstream revision, prompt hash and method settings**. Changing code, selection, configuration or data requires fresh output directories. Old local “X-style” runs lack the new method provenance and are intentionally rejected by the comparison importer.

The loops above use the local `run` command. For matched cloud runs, use `daytona matrix` below, not the legacy OpenRouter-only `daytona run` launcher; its parent output is not a matched comparison input.

## Matched Daytona matrix

`daytona matrix` runs all four methods concurrently with dedicated sandboxes per method and **one active question per sandbox**. `--workers-per-method` defaults to `1`, preserving the original sequential-per-method behavior. Each method uses `min(requested workers per method, selected question count)` effective workers; the pool must contain at least **four times that many distinct workers**, with no sharing between methods or oversubscription. Each method has a dynamic queue: a free worker takes its next selected question. Failed attempts remain in the cohort, with no retries or dropped failures.

`daytona create --workers` supports pools of up to **120 workers**, starting at most **10 concurrently**; this startup limit does not cap matrix task concurrency. The legacy `daytona run` still supports at most 10 workers. Prepare the snapshot as described in [BrowseComp-Plus setup](browsecomp_plus.md#ten-parallel-daytona-workers), then create a fresh pool after application or credential changes. A separate `--pool` path is recommended to avoid reusing the legacy launcher's pool.

```bash
mini-extra browsecomp-plus daytona create \
  --snapshot pocket-browsecomp-plus --workers 4 \
  --provider-key DEEPSEEK_API_KEY \
  --pool data/browsecomp-plus/daytona-matrix-pool.json
mini-extra browsecomp-plus daytona check \
  --pool data/browsecomp-plus/daytona-matrix-pool.json
mini-extra browsecomp-plus daytona matrix \
  --pool data/browsecomp-plus/daytona-matrix-pool.json \
  --split eval150 --limit 1 --seed 42 -c browsecomp_plus_deepseek \
  --output runs/browsecomp-plus/matched-smoke --dry-run
```

`create --provider-key` forwards only the named provider credential; its default is `OPENROUTER_API_KEY`. Any name ending in `_API_KEY` is accepted except `DAYTONA_API_KEY`, which stays on the launcher. Use `DEEPSEEK_API_KEY` for DeepSeek or `META_API_KEY` with the Meta profile; the old single-provider `daytona run` remains OpenRouter-only. `check` performs corpus/tokenizer checks without model calls.

The matrix defaults to `browsecomp_plus_deepseek` when no `-c` is supplied. Explicit configs replace that default provider profile; the base configuration is always applied. Do not layer provider profiles. The pool's forwarded credential must match the selected provider.

Selection is shared across methods: repeat `--query-id ID` to select specific questions (add `--limit 0` to run all specified IDs), use `--limit 0` without IDs for the whole split, or use `--seed N` for reproducible random selection without replacement. The cohort is randomized **once**, not independently per method or worker; without a seed, selection follows dataset order. `selection.yaml` in the parent output records the shared selected IDs, seed, requested `workers_per_method`, effective workers per method and per-method sandbox lists. There is no hidden resampling or resume: matrix requires a new or empty output directory.

`--dry-run` validates selection and configuration, writes nothing and calls no API. To launch the paid smoke run after validation, omit that flag:

```bash
mini-extra browsecomp-plus daytona matrix \
  --pool data/browsecomp-plus/daytona-matrix-pool.json \
  --split eval150 --limit 1 --seed 42 -c browsecomp_plus_deepseek \
  --output runs/browsecomp-plus/matched-smoke
```

### Thirty questions across 120 sandboxes

To run all four methods with up to 30 dedicated sandboxes each on the same 30-question cohort:

```bash
mini-extra browsecomp-plus daytona create \
  --snapshot pocket-browsecomp-plus --workers 120 \
  --provider-key DEEPSEEK_API_KEY \
  --pool data/browsecomp-plus/daytona-matrix-120-pool.json
mini-extra browsecomp-plus daytona check \
  --pool data/browsecomp-plus/daytona-matrix-120-pool.json
mini-extra browsecomp-plus daytona matrix \
  --pool data/browsecomp-plus/daytona-matrix-120-pool.json \
  --workers-per-method 30 --split eval150 --limit 30 --seed 42 \
  -c browsecomp_plus_deepseek \
  --output runs/browsecomp-plus/matched-30
```

This launches a paid run; append `--dry-run` to the matrix command to validate without launching tasks. Fewer selected questions clamp the worker count: with `--workers-per-method 30 --limit 10`, only 10 workers per method (40 active task slots total) are used, even in a 120-worker pool. Unused workers do not receive duplicate questions.

For a single effective worker per method, the original `METHOD/run.yaml`, `METHOD/summary.yaml`, root query results and `METHOD/trajectories.tar.gz` layout is unchanged. Multiworker runs retain task shards at:

```text
METHOD/tasks/QUERY_ID/run.yaml
METHOD/tasks/QUERY_ID/QUERY_ID.json
METHOD/tasks/QUERY_ID/trajectories.tar.gz
```

Each method also publishes its parent `METHOD/run.yaml`, root `METHOD/QUERY_ID.json` results and `METHOD/summary.yaml` using the common full-cohort manifest and fingerprint. Original shard provenance is preserved rather than rewritten during aggregation. Shared selection metadata and sandbox lists stay in the matrix parent's `selection.yaml`. Compare and grade the **root method directories**, not individual task shards; the comparison and grading interface is unchanged.

Successful collection removes the remote workspace. On timeout, available partial artifacts are collected before raising; collection failures retain the remote workspace at the recovery path printed when launching. Recover it before the sandbox's idle deletion. With this profile, the $0.50 threshold applies per method per question, including auxiliary calls. It is a soft stopping threshold checked between calls, not a billing cap. Plan for model costs across all four methods, provider request/token rate limits, Daytona account quotas and infrastructure costs, plus any judging costs. At the snapshot's 1 CPU / 4 GB per worker, provisioning 120 sandboxes provisions 120 CPUs / 480 GB even if fewer task slots are active. Creating a pool provisions billable infrastructure; `check` makes no model calls, and matrix `--dry-run` calls no API. Editing these docs or validating with `--dry-run` starts no paid run. These examples do not imply an evaluation has been run.

Inspect workers with the existing status command; after matrix completion, compare its parent output directly:

```bash
mini-extra browsecomp-plus daytona status \
  --pool data/browsecomp-plus/daytona-matrix-pool.json
mini-extra browsecomp-plus compare \
  --runs runs/browsecomp-plus/matched-smoke \
  --output runs/browsecomp-plus/matched-smoke/comparison.yaml
```

This ungraded comparison reports resources, not correctness. Run the judge separately for each method directory, as below, and pass the corresponding grade parent with `--grades` to report success rates.

## Grade and compare

Run the pinned official Qwen judge from [BrowseComp-Plus evaluation](browsecomp_plus.md#evaluate) separately for each method, on the GPU judge machine:

```bash
for method in pocket_context react resum acm; do
  data/browsecomp-plus/judge-venv/bin/python \
    data/browsecomp-plus/upstream/scripts_evaluation/evaluate_run.py \
    --input_dir "runs/browsecomp-plus/eval-comparison/$method" \
    --ground_truth data/browsecomp-plus/queries.jsonl \
    --qrel_evidence data/browsecomp-plus/upstream/topics-qrels/qrel_evidence.txt \
    --eval_dir runs/browsecomp-plus/judgments \
    --tensor_parallel_size 1
done
```

The upstream evaluator mirrors the path after `runs/`. With these paths, judgments live beneath `runs/browsecomp-plus/judgments/browsecomp-plus/eval-comparison/METHOD/`.

```bash
mini-extra browsecomp-plus compare \
  --runs runs/browsecomp-plus/eval-comparison \
  --grades runs/browsecomp-plus/judgments/browsecomp-plus/eval-comparison \
  --output runs/browsecomp-plus/eval-comparison/comparison.yaml
```

Omit `--grades` to inspect resource usage before grading; correctness remains unknown for ungraded completed submissions. Reports must be written outside individual method result directories.

| Measurement | Definition |
| --- | --- |
| Success rate | Correct / all attempted tasks, including budget failures. Unknown until every completed submission has a valid grade; completion alone is not correctness. |
| Total tokens/task | Mean provider-reported input + output tokens over every attempt, including all compressor/querier attempts and persisted format errors. Repeated inputs count on every call. |
| Cost USD/task | Mean accumulated model cost, including auxiliary calls. OpenRouter reports cost; LiteLLM estimates from its pricing registry. Infrastructure and external judge costs are excluded. |
| Latency s/task | Mean monotonic time inside `agent.run`, including retries, tools, auxiliary calls and per-step saving; excludes environment/model setup, queueing and grading. |
| Peak input tokens/task | Mean per-task maximum provider-reported input across main and auxiliary calls; not KV-cache size. YAML also reports the global maximum. |
| Input cache hit rate | Total provider-reported cached input tokens / total input tokens across all attempts, including auxiliary calls and failures; token-weighted, not an average of task percentages. |

Missing or partial measurement yields `N/A`, not zero or a measured-subset average. YAML includes grading/usage coverage, token and cached-input totals, tool counts and per-method provenance. Cached input is an input subset, never counted twice. Native LiteLLM/OpenRouter unavailable-cost markers make ignored-pricing fallback zeros unknown rather than free; older adapters without provenance cannot make that distinction.

The report validates shared model/retrieval/budget settings, selected IDs, worker count, code/data/index provenance, and each port's canonical policy prompts/tool configuration. It allows explicitly recorded method-specific settings and prompts. Upstream judge records must match IDs and exact submissions, contain boolean correctness without parse errors, and identify matching judge metadata. Metadata validation cannot establish complete grading provenance: keep judge source/version/settings fixed externally. Do not mix published paper scores into this matched table; their judge, retriever, windows or weights may differ.

## Offline verification

```bash
pytest -q tests/comparisons tests/models/test_comparison_tools.py \
  tests/models/test_method_tools.py tests/run/test_benchmark_metrics.py \
  tests/run/test_context_comparison.py tests/run/test_browsecomp_plus.py
```

Replay tests cover copied prompt/schema parity, full resets, repeated archive boundaries, native call/result integrity, query-only recall, parser fallback/retry, failure rollback and billed auxiliary errors without mocks or paid calls. Corpus tests build a small real Lucene index. These tests establish implementation behavior, not statistical superiority: inspect memory usage, run ablations and report paired uncertainty/repetitions for benchmark claims.
