# Comparing context-management methods

This foundation runs **Pocket Context, ReAct, training-free ReSum, and ACM** through the same BrowseComp-Plus harness and reports five measurements. It does not contain benchmark results or run paid evaluations automatically.

ReSum and ACM are **local, training-free adaptations**, not the authors' official implementations or trained policies. Treat results as comparisons of context strategies in this harness, not reproductions of paper scores. Before claiming paper-level parity, validate their prompts, boundaries, retrieval semantics, and hyperparameters against the upstream implementations.

## Methods

| `--method` | Working-context behavior | Recall |
| --- | --- | --- |
| `pocket_context` | Existing agent-selected `compact_block` ranges; separate same-model compressor | `expand_block` returns archived text |
| `react` | Append-only history; no context catalog or memory tools | Retrieve source documents again |
| `resum` | Automatically rolls older working history into a summary at an estimated 64,000 input tokens | No archive-retrieval interface |
| `acm` | Agent calls argument-free `manage_context`; framework summarizes the unsummarized interval | `query_memory(tag, query)` calls a separate same-model querier |

All methods retain system instructions, the original question, and complete tool-call/result exchanges. Both baseline compressors conservatively retain the latest exchange. ReSum folds its previous summary into the next one; ACM keeps earlier memories and archives separate and advances its boundary only after successful compaction. Audit trajectories remain on disk for every method, but are not accessible through the corpus interface. ReAct and ReSum cannot access memory archives. ACM cannot invoke Pocket Context's verbatim expansion.

The native LiteLLM and OpenRouter interfaces, including their Responses variants, support per-method tool schemas. The deterministic native-tool replay models support offline tests. Text-action and other unsupported baseline interfaces fail explicitly rather than silently exposing Pocket Context's tools.

## Develop without touching the held-out set

First follow [BrowseComp-Plus setup](browsecomp_plus.md), including Java 21, the pinned corpus, and model credentials. Tune prompts, thresholds, and budgets only on `development680`.

For example, smoke-test one development task for each method:

```bash
for method in pocket_context react resum acm; do
  mini-extra browsecomp-plus run \
    --method "$method" --split development680 --limit 1 \
    -c browsecomp_plus_openrouter \
    --output "runs/browsecomp-plus/dev-comparison/$method"
done
```

These commands make model calls and incur costs. Nothing in the comparison/reporting command makes model calls. The model profile is optional; alternatively pass the same `--model`, `--model-class`, and common YAML settings to every method.

`--method` is applied after YAML overrides and controls the agent class and memory-tool exposure. The shared system template branches on the available tools; retrieval settings, the question, final-answer format, and research instructions stay shared. ReSum's trigger can be tuned with `-c agent.resum_token_threshold=64000` on its run. Freeze it before held-out evaluation.

The cost and wall-time thresholds include auxiliary summarization and memory-query calls. They are checked between calls, not hard billing caps. `agent.step_limit` counts **main-agent calls**, not all inference calls; auxiliary calls are separately counted and included in total token usage and cost. The context limit and response allowance are the same across methods. There is no additional hard cumulative-token cap. Keep provider routing, reasoning effort, generation settings, and worker count fixed when comparing. Latency is sensitive to provider load and local contention; repeat or interleave runs before interpreting small differences.

## Freeze and run all 150 evaluation tasks

Use separate output directories and the same selection for every method:

```bash
for method in pocket_context react resum acm; do
  mini-extra browsecomp-plus run \
    --method "$method" --split eval150 --workers 4 \
    -c browsecomp_plus_openrouter \
    --output "runs/browsecomp-plus/eval-comparison/$method"
done
```

A method can exit with status 1 when a task hits a budget; its failed-task artifact must remain in the evaluation denominator. Inspect failures before continuing. Missing task artifacts cause the comparison command to reject the run, rather than compare only an intersection or successful tasks.

Explicit method runs and all `eval150` runs retain every recorded attempt on resume, including failures. They do not silently retry until success. `--redo-existing` explicitly replaces attempts; use a fresh directory for independent repeats and do not select the best result across reruns as Pass@1. Legacy development runs without `--method` retain their previous retry-incomplete behavior.

Each `run.yaml` records the selected question IDs, effective configuration, worker count, attempt-retention policy, pinned data/index revisions, index inventory, method, and a hash of the installed Python implementation. Results carry the manifest fingerprint. Changing code, selection, data, or configuration requires a new output directory; do not combine old runs with new ones. Existing manifests created before this foundation do not contain the required provenance: rerun into fresh directories.

The matched-comparison workflow uses the **local `run` command**. The existing Daytona launcher does not yet have a `--method` matrix or matching parent-manifest support; do not feed Daytona parent directories into `compare`.

## Grade with the upstream evaluator

Use the pinned official Qwen judge described in [BrowseComp-Plus evaluation](browsecomp_plus.md#evaluate), separately for each method's run directory. For example, on the Linux GPU judge machine:

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

The upstream evaluator mirrors the path following `runs/`, so these judgments are under `runs/browsecomp-plus/judgments/browsecomp-plus/eval-comparison/METHOD/`. The comparison importer expects one subdirectory per method and reads upstream `*_eval.json` records recursively beneath each one.

It validates query IDs and the exact submitted response against the run, rejects duplicate judgments and differing judge models/settings, and accepts only boolean `judge_result.correct` without a parse error. Unparseable or missing grades for completed submissions remain unknown, never a success. Noncompleted tasks count as incorrect. Keep the judge revision/configuration fixed externally as well; the importer can only validate metadata present in the upstream artifacts. Reference answers and judge inputs never enter the agent context.

## Generate the five-column comparison

Before grading, inspect costs, usage, and latency with success rates shown as `N/A`:

```bash
mini-extra browsecomp-plus compare \
  --runs runs/browsecomp-plus/eval-comparison
```

Then import the judgments and save a machine-readable report outside individual method directories:

```bash
mini-extra browsecomp-plus compare \
  --runs runs/browsecomp-plus/eval-comparison \
  --grades runs/browsecomp-plus/judgments/browsecomp-plus/eval-comparison \
  --output runs/browsecomp-plus/eval-comparison/comparison.yaml
```

| Measurement | Definition |
| --- | --- |
| Success rate | Correct / all attempted tasks, including budget failures. `N/A` until every completed submission has a valid correctness grade. A completed submission is not itself a correct answer. |
| Total tokens/task | Mean provider-reported input + output tokens over all attempts, including compressor/querier calls and persisted format errors. Repeated input tokens are counted on every call. |
| Cost USD/task | Mean accumulated model cost over all attempts, including auxiliary calls. OpenRouter reports cost; LiteLLM derives it from its pricing registry. Retrieval infrastructure and judge costs are excluded. |
| Latency s/task | Mean monotonic elapsed time inside `agent.run`, including model retries, tools, auxiliary calls, and per-step trajectory saving. Excludes model/environment initialization, queueing, and external grading. |
| Peak input tokens/task | Mean per-task maximum provider-reported input usage across all main and auxiliary calls. This is not KV-cache size or a token estimate. The YAML also reports the maximum across tasks. |

Missing or partial provider usage yields `N/A`, not a misleading zero or an average over only measured tasks. YAML summaries include measurement coverage, grading counts, input/output totals, cached-input totals where reported, and semantic tool counts. Cached input is a subset of input tokens, never added a second time. Use normal cost tracking for cost comparisons. Native LiteLLM/OpenRouter responses mark unavailable cost explicitly, so `cost_tracking: ignore_errors` fallback zeros become `N/A` rather than free runs; genuine provider-reported zero prices remain zero. Legacy or third-party adapters without cost provenance cannot make this distinction.

The report rejects differing model settings, common prompts, retrieval configuration, budgets, selected IDs, implementation hashes, and dataset/index inventories. This is intentionally strict for matched local experiments, including index file timestamps; moved/rebuilt indexes may require fresh matched runs.

This is an evaluation foundation, not statistical evidence of superiority. Report paired uncertainty and repeat runs before interpreting small accuracy differences on 150 tasks. Check compaction/recall activity and add the planned ablations before attributing a gain to selective compaction or exact restoration.

## Offline verification

```bash
pytest -q tests/agents/test_baselines.py tests/models/test_method_tools.py \
  tests/run/test_benchmark_metrics.py tests/run/test_context_comparison.py \
  tests/run/test_browsecomp_plus.py
```

The corpus tests build a small real Lucene index. Replay tests exercise compaction, recall, safe exchange grouping, auxiliary costs, failure preservation, and schema isolation without paid calls, mocks, or patches.
