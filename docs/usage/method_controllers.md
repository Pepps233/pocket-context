# One benchmark controller per method

Use four independent trusted controllers to isolate the RAM and process lifetime of Pocket Context, ReAct, ReSum and ACM.
Each controller runs one method with **six concurrent agents**, for **24 workers total**, plus four controllers.
This workflow supports managed BrowseComp-Plus and SWE-bench Verified runs.
Omitting `--method` preserves the existing all-method behavior and its defaults.

The controller owns inference, live histories and tokenizers.
Corpus workers execute retrieval operations, and SWE workers execute coding tools in fresh official instance images.
Neither receives provider credentials.
Completed traces are verified on the controller's artifact volume before their bulk local copies are deleted.
A controller failure interrupts only that controller's job; failed jobs are not resumed or retried automatically.
Workers belonging to a killed workload may still need explicit cleanup.

For SWE-bench Verified, use `--benchmark swebench-verified --split test --data-dir data/swebench-verified --limit 125` and the `context_benchmark_deepseek` / `context_full_swebench` profiles with the prepared image-digest map.
The selected 125 tasks produce 500 attempts across the four controllers.
Use `--preflight --limit 6 --workers-per-method 6` for infrastructure checks, including edit capture and the repository-specific test runner, without model inference.

## Plan locally

Run from the same checkout and prepared dataset for all four methods.
Use the same configuration, seed, split, task selection and worker count so the final comparison remains matched.
The default worker count with `--method` is six; `--workers-per-method` can override it.
Actual concurrency is capped by the number of selected tasks.
For the full `eval150` cohort, each method makes 150 planned attempts and all four together make 600.

The following commands only print local plans.
They do not require controller manifests, call cloud APIs, create sandboxes or invoke models.

```bash
for method in pocket_context react resum acm; do
  mini-extra benchmark-controller submit \
    --manifest "runs/context-evaluation/controllers/$method.json" \
    --benchmark browsecomp-plus --method "$method" \
    --data-dir data/browsecomp-plus --split eval150 \
    --limit 150 --seed 42 --workers-per-method 6 \
    -c browsecomp_plus_deepseek --dry-run
done
```

Each plan records exactly one entry in `methods`, six `maximum_active_sandboxes`, and 150 `planned_attempts`.
Submission checks the remote method, concurrency, cohort hash, implementation and other execution settings against this plan before launching the workload.

## Switch the context window

Use `--context-window 64000` to set the shared working context budget for a new run.
Use `--context-window 128000` to select 128K instead.
The value is an exact token count including the next response allowance, not a cumulative token budget across calls.

```bash
mini-extra benchmark-controller submit \
  --benchmark swebench-verified --data-dir data/swebench-verified \
  --limit 125 --seed 42 --workers-per-method 6 \
  -c context_benchmark_deepseek -c context_full_swebench \
  --context-window 64000 --dry-run
```

This prints a local plan without creating sandboxes or running inference.
Use the same option on each method's submission when using separate controllers.
The option is also available on `browsecomp-plus daytona matrix`, `browsecomp-plus run`, and the corpus controller worker CLI.

`agent.context_token_limit` remains the single configuration value consumed by the agents.
The CLI option overrides that value after all `-c` profiles are merged; omitting the option preserves the configured limit, which defaults to 128,000 in the benchmark profiles.
You can also put `agent: {context_token_limit: 64000}` in a shared YAML override.
All method system prompts are generated from the effective value; memory tool descriptions have no separate context-window setting to edit.
Runtime context checks and auxiliary model calls use this same limit, while each method retains its existing compaction policy and summary output allowance.

Dry-run plans show `context_token_limit`, and controller submission checks that the remote plan agrees.
Saved run configurations and method provenance record the effective limit.
Changing the limit applies to new runs; it does not update existing controller jobs or historical results.
Use fresh run outputs for a different context budget so the experiment remains distinguishable.

## Create controllers when authorized

These commands create billable infrastructure; they are instructions for a later authorized deployment.
Use a distinct manifest and artifact volume for each controller.
The artifact store has a single-writer namespace, so do not point concurrent controllers at the same volume.
An existing controller can be reused for one method after upgrading its idle runtime to this checkout; do not overwrite its manifest or discard its archives.

```bash
for method in pocket_context react resum acm; do
  mini-extra benchmark-controller create \
    --manifest "runs/context-evaluation/controllers/$method.json" \
    --project . --cpu 2 --memory 8 --disk 10 --ttl-minutes 0 \
    --provider-key DEEPSEEK_API_KEY \
    --volume-name "pocket-context-$method-results"
done
```

This example allocates 8 GiB per controller, 32 GiB total.
Six agents on an 8-GiB controller is a starting configuration to measure, not a validated memory guarantee.
Measure controller cgroup memory and OOM counters during initialization and several complete task waves before relying on this concurrency.
Increasing artifact-volume capacity does not increase controller RAM.

## Preflight and submit each method

First check all six corpus workers for each method without model inference.
Preflight still creates billable sandboxes and requires trusted provider credentials for metadata checks.
Run this only after infrastructure authorization.

```bash
for method in pocket_context react resum acm; do
  mini-extra benchmark-controller submit \
    --manifest "runs/context-evaluation/controllers/$method.json" \
    --benchmark browsecomp-plus --method "$method" \
    --data-dir data/browsecomp-plus --split eval150 \
    --limit 150 --seed 42 --workers-per-method 6 \
    -c browsecomp_plus_deepseek --preflight
done
```

Save each returned job ID with its method and controller manifest.
Wait for all preflights to finish successfully, including worker cleanup.
After separate inference authorization, repeat the same submission commands without `--preflight` to start the real runs.
Each submission is detached, so the four jobs can run concurrently without keeping the laptop connected.
Use fresh job IDs for status, collection and cleanup; never resubmit a pending job automatically.

```bash
mini-extra benchmark-controller status \
  --manifest runs/context-evaluation/controllers/react.json --job ACTUAL_REACT_JOB_ID
mini-extra benchmark-controller logs \
  --manifest runs/context-evaluation/controllers/react.json --job ACTUAL_REACT_JOB_ID --lines 100
```

## Collect and compare

Each controller generates a report for its selected method only.
Fetch each inactive job to its own fresh directory, substituting its actual ID.
Repeat for all four methods.

```bash
mini-extra benchmark-controller fetch \
  --manifest runs/context-evaluation/controllers/react.json \
  --job ACTUAL_REACT_JOB_ID --output runs/context-evaluation/collected/react
```

The fetched directory contains `output/react/`, including the method's manifest and results.
After fetching all four jobs, assemble a local comparison directory with links to the unchanged method outputs.
Use a fresh `combined` directory; this does not copy, modify or rerun the recorded attempts.

```bash
mkdir runs/context-evaluation/combined
for method in pocket_context react resum acm; do
  ln -s "../collected/$method/output/$method" "runs/context-evaluation/combined/$method"
done
mini-extra browsecomp-plus compare \
  --runs runs/context-evaluation/combined \
  --output runs/context-evaluation/combined/comparison.yaml
```

The full comparison still requires all four methods and validates matching settings, provenance, selected tasks and result fingerprints.
Add `--grades PATH` when separate upstream judgments are available.
Completed submissions alone are not correctness judgments.

## Cleanup and controller lifetime

Normal completion cleans up that method's corpus workers and preserves its archived results.
For failed runs, inspect the job status and recovery journal, then use the existing [cleanup workflow](benchmark_controller.md#collect-cancel-and-reconcile) with the corresponding manifest and job ID.
Stopping or deleting one controller does not stop the others.
Idle controllers continue to incur compute charges until explicitly stopped, and stopping does not resume interrupted jobs later.
See [artifact volume lifecycle](controller_volumes.md) for archive verification, recovery and volume retention.
