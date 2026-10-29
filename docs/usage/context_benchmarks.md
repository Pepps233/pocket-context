# Context benchmarks: BrowseComp-Plus and SWE-bench Verified

Compare **Pocket Context, ReAct, training-free ReSum, and ACM Base** on a fixed cohort within each benchmark. Method IDs are `pocket_context`, `react`, `resum`, and `acm`. Each method receives the same selected task IDs, model profile and shared budgets; failed attempts stay in the cohort, without retries or replacement tasks.

This page describes the generic `mini-extra benchmark-matrix` interface and the existing BrowseComp-Plus matrix workflow. The BrowseComp-Plus corpus, pool lifecycle and grading path remain unchanged; see [BrowseComp-Plus setup](browsecomp_plus.md) and [method fidelity and comparison](context_comparison.md). Benchmark-specific prompt/tool adaptations are recorded in provenance. This is a controlled shared-policy comparison, **not a reproduction of published paper scores**, and does not include ReSum-GRPO training.

## Evaluation allocation

| Benchmark | Split and selection | Unique tasks | Methods | Attempts |
| --- | --- | ---: | ---: | ---: |
| BrowseComp-Plus | Entire `eval150`, seed 42 | 150 | 4 | 600 |
| SWE-bench Verified | `test`, 125 tasks selected with seed 42 | 125 | 4 | 500 |
| Total | Two distinct benchmark cohorts | **275** | 4 | **1,100** |

The allocation is configured in `pocketcontextagent/config/benchmarks/context_evaluation.yaml`. After preparing both datasets, freeze and inspect the selected IDs without calling APIs:

```bash
mini-extra benchmark-matrix allocation \
  --output runs/context-evaluation/allocation.yaml
```

The seed fixes selection, not model determinism. Selection happens once per benchmark, not independently per method or worker. The 1,100 attempts refer to the full allocation only; separate one-task/four-method smoke runs add four attempts per benchmark and are not silently merged into the full evaluation. Freeze code, prompts, configuration and selection before the full evaluation. Keep smoke outputs separate from full-evaluation outputs; use fresh output directories rather than replacing failed attempts or selecting the best rerun.

**No paid runs are launched as part of setup or documentation validation.** Data preparation downloads public artifacts without model calls. Matrix `--dry-run` calls no API and creates no sandboxes. Pool creation, live matrix execution and model-based grading are separate, explicitly paid operations.

## Prerequisites and execution boundaries

- Put `DAYTONA_API_KEY` and `DEEPSEEK_API_KEY` in the orchestrator's environment or ignored `.env`. Do not print credentials or commit them.
- Prepare the pinned BrowseComp-Plus corpus/index, tokenizer and compatible snapshot using the [existing setup](browsecomp_plus.md). These examples retain its `pocket-browsecomp-plus` snapshot name; confirm it exists in your Daytona account and contains the current application and required corpus/runtime dependencies.
- Use `browsecomp_plus_deepseek` for BrowseComp-Plus and `context_benchmark_deepseek` for SWE-bench Verified, consistently across all four methods within each benchmark. Do not layer provider profiles.
- Plan provider request/token rate limits, per-task cost/time limits, Daytona sandbox quotas and infrastructure costs before increasing concurrency. Stopping thresholds checked between calls are not hard billing caps. Auxiliary memory/model calls contribute to usage and cost.
- SWE-bench grading needs a **separate trusted Docker-capable host** and the pinned official harness. It is not run inside a coding sandbox.

### Different Daytona lifecycles

| | BrowseComp-Plus | SWE-bench Verified |
| --- | --- | --- |
| Provisioning | Existing precreated `--pool` | Automatically create and delete one fresh sandbox per method/task attempt |
| Runtime | Existing corpus worker application | Official instance image for that task |
| Inference | Existing remote worker execution, with the explicitly forwarded provider key | Model agents and orchestration run locally; coding tool operations run remotely in Daytona |
| Sandbox credentials | Existing `create --provider-key DEEPSEEK_API_KEY`; Daytona key stays on launcher | **No model keys or Daytona keys in the coding sandbox** |

Do not pass a precreated `--pool` to the SWE-bench workflow. Each fresh coding sandbox starts from the task's official instance image. Successful artifact capture is followed by deletion. Before agent execution, automatic deletion is disabled; if capture fails, the launcher prints the sandbox ID, stops the sandbox and retains its disk for recovery. Retained disks can incur storage charges. After recovery, remove remaining run sandboxes with `mini-extra benchmark-matrix delete --pool RUN_ROOT/sandbox-cleanup.json`; run that only after execution has stopped. Coding uses that image directly, **without nested Docker**, with outbound network access disabled. Official grading is a separate operation.

Both matrices support up to **30 active task slots per method, 120 total**, with one active task per sandbox. Effective concurrency is `min(workers_per_method, selected task count)` for each method. Thus a one-task smoke with `--workers-per-method 1` means four attempts, while requesting 30 workers for one task still gives only four active slots. Slots are concurrency limits, not the number of attempts in the cohort.

## Prepare SWE-bench Verified data

```bash
mini-extra benchmark-matrix prepare \
  --benchmark swebench-verified \
  --data-dir data/swebench-verified
```

Preparation uses `princeton-nlp/SWE-bench_Verified`, revision **`c104f840cc67f8b6eec6f759ebc8b2693d585d4a`**, split `test`. Keep evaluator-only data on the trusted orchestrator/grading host. Agent inputs are allowlisted public fields: task ID, problem statement, repository, version and base commit, plus the canonical runtime image identifier. Gold patches, test patches, hidden evaluation labels and hints are not agent inputs and must not be copied or mounted into coding sandboxes.

Each task uses the canonical Docker Hub image:

```text
docker.io/swebench/sweb.eval.x86_64.<normalized-instance-id>:latest
```

Normalize the instance ID by replacing `__` with `_1776_` and lowercasing. For example, `django__django-11099` maps to `docker.io/swebench/sweb.eval.x86_64.django_1776_django-11099:latest`. Image names are derived from instance IDs, not arbitrary dataset-supplied image URLs. The official `:latest` tags are **mutable**, not immutable revision pins. Record the requested image and any resolved digest available from provisioning; a pinned dataset alone does not pin image contents.

The coding sandbox allocation is **2 CPU, 8 GB RAM and 10 GB disk per sandbox**. At 120 concurrent slots that is up to 240 CPUs, 960 GB RAM and 1,200 GB disk provisioned. Verify account quotas and image availability before a paid launch; lower concurrency when necessary without changing the selected cohort. The orchestrator also holds active model histories and performs token accounting, so monitor its RAM/CPU before scaling to 120 local agent threads.

## One-task smoke: validate first

### SWE-bench Verified: no precreated pool

This selects one task, shared by all four methods, and makes no API calls:

```bash
mini-extra benchmark-matrix matrix \
  --benchmark swebench-verified --data-dir data/swebench-verified \
  --split test --limit 1 --workers-per-method 1 --seed 42 \
  -c context_benchmark_deepseek \
  --output runs/swebench-verified-smoke --dry-run
```

The existing dispatcher is equivalent for the new benchmark:

```bash
mini-extra browsecomp-plus daytona matrix \
  --benchmark swebench-verified --data-dir data/swebench-verified \
  --split test --limit 1 --workers-per-method 1 --seed 42 \
  -c context_benchmark_deepseek \
  --output runs/swebench-verified-smoke --dry-run
```

After reviewing the selection, configuration and budget, **omit `--dry-run` from one of these commands** to authorize the paid one-task/four-method smoke. It automatically provisions fresh task sandboxes; do not run `daytona create` for SWE-bench Verified.

### BrowseComp-Plus: retain the existing pool workflow

Pool creation provisions billable infrastructure, even before model inference. Run these commands only when ready for the paid smoke, after completing the existing corpus/snapshot setup:

```bash
mini-extra browsecomp-plus daytona create \
  --snapshot pocket-browsecomp-plus --workers 4 \
  --provider-key DEEPSEEK_API_KEY \
  --pool data/browsecomp-plus/context-smoke-pool.json
mini-extra browsecomp-plus daytona check \
  --pool data/browsecomp-plus/context-smoke-pool.json
mini-extra browsecomp-plus daytona matrix \
  --pool data/browsecomp-plus/context-smoke-pool.json \
  --workers-per-method 1 --split eval150 --limit 1 --seed 42 \
  -c browsecomp_plus_deepseek \
  --output runs/browsecomp-plus-smoke --dry-run
```

`check` checks the remote corpus/runtime without model calls, but uses the provisioned pool. Matrix `--dry-run` itself calls no API. To authorize four paid smoke attempts, omit `--dry-run` from the matrix command. The existing BrowseComp-Plus pool management and cleanup commands remain unchanged; follow the [pool lifecycle instructions](browsecomp_plus.md).

## Full allocation: only after smoke review

These examples retain `--dry-run` intentionally. Removing it launches paid inference. Do not run both a smoke and a full evaluation into the same output directory.

### All 150 BrowseComp-Plus evaluation tasks

Create a separate 120-worker pool only after budget/quota approval. Pool startup permits at most 10 concurrent creations; that startup limit is not the matrix task-concurrency limit.

```bash
mini-extra browsecomp-plus daytona create \
  --snapshot pocket-browsecomp-plus --workers 120 \
  --provider-key DEEPSEEK_API_KEY \
  --pool data/browsecomp-plus/context-eval150-pool.json
mini-extra browsecomp-plus daytona check \
  --pool data/browsecomp-plus/context-eval150-pool.json
mini-extra browsecomp-plus daytona matrix \
  --pool data/browsecomp-plus/context-eval150-pool.json \
  --workers-per-method 30 --split eval150 --limit 150 --seed 42 \
  -c browsecomp_plus_deepseek \
  --output runs/bcp150 --dry-run
```

This is **600 attempts**, at most 120 active at once. Run the two benchmark phases sequentially if you want to keep the overall maximum at 120; separate matrix invocations do not share a global concurrency limit. The existing corpus snapshot's resource allocation determines pool infrastructure cost, including idle workers. A dry-run matrix does not make the preceding pool creation free.

### Fixed 125-task SWE-bench Verified cohort

```bash
mini-extra benchmark-matrix matrix \
  --benchmark swebench-verified --data-dir data/swebench-verified \
  --split test --limit 125 --seed 42 --workers-per-method 30 \
  -c context_benchmark_deepseek \
  --output runs/swe125 --dry-run
```

This is **500 attempts**, with fresh task-image sandboxes and at most 120 active at once. Do not precreate a pool. Preserve the exact selected IDs and original full-cohort fingerprints; do not substitute a new cohort if an image, task or provider call fails. Failures remain visible with no automatic task retries. Normal agent budget exits are retained and the queue continues. Infrastructure/model exceptions fail closed rather than reusing a sandbox with uncertain state; completed artifacts remain available, and incomplete cohort coverage blocks a purported complete comparison.

## Artifacts and provenance

The matrix parent records shared selection and configuration. Inspect the root method directories, not an intersection of successful task shards:

```text
RUN_ROOT/
  selection.yaml
  METHOD/
    run.yaml
    summary.yaml
    ID.json
    tasks/ID/
      run.yaml
      ID.json
      trajectories...
    predictions.jsonl       # SWE-bench predictions, for official evaluation
```

`METHOD` is one of the four method IDs. Task-shard provenance and the original full-cohort fingerprints are preserved during aggregation; parent manifests must still describe every selected attempt. The existing BrowseComp-Plus single-worker trajectory/archive layout is unchanged. SWE-bench predictions contain actual patches, not final-answer prose; empty/failed attempts must not silently disappear. Inspect trajectories, termination reasons, model/auxiliary usage, sandbox cleanup and exported patches before interpreting a smoke as operationally successful.

Submission or an agent's claim of completion is **not correctness**. A budget-terminated SWE-bench attempt can still contain a grade-relevant patch: export and evaluate that patch rather than discarding it because of its termination status.

## Grade separately, then compare

### SWE-bench Verified official harness

Use official SWE-bench **v4.1.0**, commit **`726c5461e2ef52d83cf1ea2107870a8bb3328d57`**, on a separate trusted Docker-capable Linux x86_64 host. Keep the pinned private grading dataset outside all coding sandboxes. Public/private dataset hashes, expected harness revision, prediction hashes and run fingerprints are bound into the exported handoff. Import requires that matching handoff and run prefix; do not edit data or predictions after export. The v4.1.0 evaluator has no dataset-revision CLI flag; use the prepared pinned grading data rather than silently evaluating against a newly downloaded dataset revision.

Export one prediction file and print the official harness command for each method:

```bash
mini-extra benchmark-matrix export-swebench \
  --runs runs/swebench-verified-smoke \
  --data-dir data/swebench-verified \
  --run-prefix swe-smoke-42
```

Exporting does not run the harness. Copy the run artifacts and private grading data to the trusted grading host, preserving the relative paths, and activate its pinned-harness environment before running the printed commands. They default to `python`; use `--harness-python /path/to/grading-venv/bin/python` when exporting for a different interpreter. Review and execute the printed per-method commands only after installing the pinned harness. They perform official patch application and test evaluation using Docker. Use fresh run prefixes and retain the associated logs; do not reuse stale official reports.

Import the resulting official reports and saved patches, then compare:

```bash
mini-extra benchmark-matrix import-swebench \
  --runs runs/swebench-verified-smoke \
  --logs /trusted/swebench/logs/run_evaluation \
  --run-prefix swe-smoke-42 \
  --output runs/swebench-verified-smoke-grades
mini-extra benchmark-matrix compare \
  --runs runs/swebench-verified-smoke \
  --grades runs/swebench-verified-smoke-grades
```

Use the log root corresponding to the printed harness commands. Import binds official report instance IDs to the **actual exported patches**, rejecting mismatched/stale results. Retain `report.json` and `patch.diff` from the same trusted harness execution. This checks artifact consistency, not the authenticity of untrusted logs.

Only official test evidence establishes `resolved`; neither `Submitted`, nonempty patches nor successful patch application is sufficient. A successfully captured, genuinely empty patch is explicitly marked `confirmed_empty_patch` and counted as a known no-op failure, not an official test evaluation. Failed patch capture blocks prediction export and requires recovery; it is never replaced with a fabricated empty patch. Report evaluation/test coverage explicitly. **Missing grades mean unknown correctness**, not success, and partial coverage must not be presented as a fully evaluated resolved rate. Keep every selected attempt in the denominator, including budget failures and empty predictions.

For the full SWE-bench run, substitute `runs/swe125`, a fresh prefix such as `swe125-42`, and a separate grade root in all three commands.

### BrowseComp-Plus existing judge

Use the pinned official BrowseComp-Plus judge and existing per-method evaluation procedure in [Grade and compare](context_comparison.md#grade-and-compare). Grading is separate from matrix execution and has its own model/compute requirements. Pass the resulting grade parent, containing grades for all four methods, to the unchanged comparison interface:

```bash
mini-extra browsecomp-plus compare \
  --runs runs/browsecomp-plus-smoke \
  --grades PATH_TO_BCP_SMOKE_GRADE_ROOT
```

The generic alias also supports `mini-extra benchmark-matrix compare --runs ROOT --grades ROOT`; the old `mini-extra browsecomp-plus compare` interface remains supported. Compare each benchmark separately: their graders and correctness definitions differ. Do not pool their correctness scores into a single purported benchmark score. Without `--grades`, comparison can report resource usage, but correctness remains unknown.

## Ready-to-run handoff

1. Prepare public data and verify local configuration, credentials, snapshot/image prerequisites and quotas. Run the **one-task SWE-bench dry-run** above; it must not call an API or provision infrastructure.
2. With explicit spending approval, launch **one task across all four methods per benchmark**, using the separate smoke output roots. Inspect retained failures, all four method artifacts, resource accounting and cleanup; grade separately and verify coverage.
3. Stop for review. Only after accepting both smoke results and the budget should you create the 120-worker BrowseComp-Plus pool and authorize the full **150 + 125 task / 1,100 attempt** allocation by removing `--dry-run`. Preparing these instructions does not authorize or launch that paid evaluation.
