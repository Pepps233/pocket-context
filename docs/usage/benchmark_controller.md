# Trusted cloud benchmark controller

Use `mini-extra benchmark-controller` to run trusted SWE-bench Verified or BrowseComp-Plus orchestration on Daytona rather than on your laptop.
BrowseComp-Plus can run all four methods together or use [one controller per method](method_controllers.md), with six concurrent agents per controller by default when `--method` is specified.
Its existing pool workflow remains available separately.
See [context benchmarks](context_benchmarks.md) for dataset preparation, matched methods, budgets and separate trusted grading.

## Architecture and lifetime

```text
Laptop -- native SSH / management API --> trusted cloud controller
                                         model agents + provider/Daytona credentials
                                         | coding tools only
                                         +--> isolated credential-free coding VMs
```

The controller owns orchestration, inference, status and cleanup journals. Coding VMs
receive public task inputs and execute coding commands, not model calls. Neither model
keys, Daytona keys nor private grading data belong in those VMs. The official grading
harness still needs a separate trusted Docker-capable host; the controller image is not
an assumed grading environment.

Daytona was chosen for native SSH attachment and configurable long-running sandboxes.
**E2B does support real SSH**, through a custom OpenSSH template and WebSocket proxy;
it is not just an SDK terminal. Standard E2B continuous runtime is **1 hour on Hobby /
24 hours on Pro**. Pause/resume resets that window but interrupts execution and network
connections; Enterprise limits are custom. Daytona avoids that standard continuous-run
cap, subject to verified organization policy, available balance and service availability.

The controller is explicitly private and persistent (`ephemeral=False`), with idle
stop/pause disabled (`0`), stopped-state auto-delete disabled (`-1`), and default absolute
TTL **`0`**. Creation verifies actual server lifecycle/resource settings, including the
absence of a destruction deadline; a shorter organization policy is not silently accepted.
A positive `--ttl-minutes` is an independent hard destruction deadline, not an idle timer.
Coding workers retain their separate configured safety TTL; controller TTL 0 does not
remove worker deadlines.

Accepted jobs use detached cloud-side runner and workload process groups. **They continue
after laptop shutdown, SSH disconnect or management-client exit.** One active job is
allowed per controller. This is **not server-crash-proof**: stop, restart, OOM, provider
failure or controller loss can interrupt execution. No automatic job/task retries or
restart/resume are promised. Existing bounded model-call retries are not job recovery.
A one-time claim prevents duplicate launch; after a stale claim/restart, unfinished status
is reported as `interrupted`, never automatically resumed or fabricated as successful.

## Resources, billing and trust
- Controller defaults: **2 CPU, 8 GiB RAM, 10 GiB disk**. These are controller resources,
  not a guarantee that full-concurrency orchestration fits; histories/token accounting
  consume controller RAM. Begin with a four-worker smoke and measure before scaling.
- `--workers-per-method 1` means at most **4 coding VMs**, one per matched method.
  `30` means at most **120**, limited by the selected task count; allowed range is 1–30.
- SWE workers each request **2 CPU / 8 GiB / 10 GiB**. At 120 concurrent workers, budget
  **240 CPU / 960 GiB RAM / 1,200 GiB disk**, plus controller and other resources.
  Do not size them as 4-GiB medium workers. Daytona's published Tier-3 500-GiB RAM pool
  is insufficient; verify actual quotas, per-sandbox caps, image sizes and capacity.
- The 10-GiB controller disk can fill with trajectories, staged inputs and fetch archives.
  Without a volume, fetch needs additional archive headroom. Opt into a controller-only
  [artifact volume](controller_volumes.md) at creation for verified completed-trace offload
  and volume-streamed fetch. Live state, staged inputs and temporary spool remain local;
  volumes do not guarantee that the active working set fits. Lower concurrency or arrange
  approved larger storage when needed.
- Running controller compute is billed continuously, **including every idle day** until
  explicitly stopped (about **$5.53/day** for 2 CPU/8 GiB at audited public Linux rates,
  before disk). Stopped disks and snapshots can still cost storage. Workers and inference
  are additional; check current prices and keep balance/payment and budgets valid.
- Tier-1/2 Daytona egress restrictions cannot be overridden per sandbox. Check access
  from the cloud; `network_block_all=False` alone does not guarantee arbitrary internet.
- Supply control/provider API keys only through the trusted host's environment or ignored
  private credential files. Default provider key name is `DEEPSEEK_API_KEY`; supported
  profiles can instead use `META_API_KEY` or `OPENROUTER_API_KEY`. Do not paste key values
  into CLI options or config overrides. Choose a dedicated trusted organization: Daytona
  resource scopes do not isolate toolbox access between that organization's sandboxes.
- Setup builds a credential-free runtime snapshot, then installs the frozen project wheel
  and uploads credentials separately to private `ROOT/credentials.json` (mode 600).
  Keep secrets out of images, snapshots, argv, public manifests, logs and downloads.
  Never snapshot the credential-bearing controller filesystem for distribution.
- `grading_dataset.jsonl` stays on trusted hosts only: submission stages it on the trusted
  controller for prepared-data hash validation, never in coding VMs or agent inputs.
  Gold patches/tests are not coding-agent context. Do not download private inputs as results.

## Prepare and inspect without launching inference
Examples below are instructions, not authorization to create production resources or
invoke a model. Infrastructure operations also require approval and incur cloud charges.
Run from the intended checkout with the Daytona extra installed. Prepare the dataset via
`mini-extra benchmark-matrix prepare --benchmark swebench-verified` if not already prepared.

The default local manifest is **`runs/context-evaluation/controller.json`**; remote
**`ROOT` is `/workspace/pocket-controller`**. Examples use those defaults. Add
`--manifest PATH` consistently for a different controller; preserve this identity/journal.
A public backup is stored at `ROOT/controller.json`, separate from credentials. Keep the
local manifest accessible from another trusted device; reconnecting with `status` or `ssh`
needs Daytona access, not local connectivity to the model provider. If the local handle is
lost, recover the backup through authenticated Daytona filesystem access or native SSH.

```bash
# Local plan only: no API calls, uploads, controller or keys required.
mini-extra benchmark-controller submit \
  --data-dir data/swebench-verified --limit 1 --seed 42 \
  --workers-per-method 1 -c context_benchmark_deepseek --dry-run

# Billable controller/snapshot setup; no benchmark inference.
mini-extra benchmark-controller create --project . \
  --cpu 2 --memory 8 --disk 10 --ttl-minutes 0 \
  --provider-key DEEPSEEK_API_KEY

# Authenticated provider metadata check FROM the cloud; no model inference.
mini-extra benchmark-controller check -c context_benchmark_deepseek

# Detached, billable infrastructure check: four matched workers, no model inference.
mini-extra benchmark-controller submit \
  --data-dir data/swebench-verified --limit 1 --seed 42 \
  --workers-per-method 1 -c context_benchmark_deepseek --preflight
```

Controller images pin the tested inference SDK versions, including LiteLLM's MCP extra.
Creation and `check` validate the actual MCP/DeepSeek/OpenAI adapter imports without
requesting inference. The matrix also warms these imports in its own process before
starting concurrent agents or allocating coding workers. A successful metadata request
alone does not establish that the inference adapter is installed correctly.

`check` and non-dry submissions refresh the private controller credentials, validate the
pinned adapter runtime, and verify an allowlisted official provider metadata endpoint. Even managed preflight therefore needs
provider credentials, unlike a direct matrix infrastructure-only preflight. `--dry-run`
can be combined with `--preflight`; it stays local. Submission verifies the prepared
public/private file hashes, frozen implementation and remote selection before acceptance.
Keep smoke/preflight and full-evaluation outputs distinct; each submission gets a job ID.

## Authorize, submit and observe

**Only after explicit model-run authorization**, budget/quota review and successful
preflight, submit a paid smoke by omitting both `--dry-run` and `--preflight`:

```bash
mini-extra benchmark-controller submit \
  --data-dir data/swebench-verified --limit 1 --seed 42 \
  --workers-per-method 1 -c context_benchmark_deepseek

# Save the returned ID; substitute the actual value below.
JOB=returned-job-id
mini-extra benchmark-controller status
mini-extra benchmark-controller status --job "$JOB"
mini-extra benchmark-controller logs --job "$JOB" --lines 100
mini-extra benchmark-controller ssh
mini-extra benchmark-controller ssh --command \
  "tail -f /workspace/pocket-controller/jobs/$JOB/run.log"
```

SSH uses a **fresh 60-minute token** by default (`--expires-minutes 60`), invokes native
`ssh`, and revokes the token when that invocation exits. Tokens are not persisted in the
manifest or printed as credentials. Request a fresh token for later attachment; token
expiry is not a job deadline. Do not rely on an ordinary foreground SSH shell for launch.

`status` does not start a stopped controller. `ssh`, `check`, `submit`, `fetch`, `cleanup`
and `delete` may start an existing stopped controller. An ambiguous/pending acceptance is
an instruction to inspect status and journals, **not to resubmit automatically**.
For the full 125-task cohort, first review a local plan with
`--limit 125 --seed 42 --workers-per-method 30 --dry-run`; removing that flag requires
separate authorization.

## Collect, cancel and reconcile

```bash
# Wait for completion, or cancel first; fetch requires an inactive job.
mini-extra benchmark-controller fetch --job "$JOB" \
  --output "runs/context-evaluation/collected/$JOB"

# Stop only this job's owned workload process group; inspect status afterwards.
mini-extra benchmark-controller cancel --job "$JOB"
mini-extra benchmark-controller status --job "$JOB"
mini-extra benchmark-controller logs --job "$JOB" --lines 200

# Reconcile journaled worker/snapshot resources after the job is inactive.
mini-extra benchmark-controller cleanup --job "$JOB"
# Destructive alternative ONLY after recovery/discard authorization:
mini-extra benchmark-controller cleanup --job "$JOB" --discard-uncollected
```

`fetch` packs **output and public job metadata only**: `output/`, `status.json`, `plan.json`,
redacted `run.log`, and any `controller-cleanup.json`. It excludes private inputs,
credentials and the launch request; it verifies archive SHA-256 and rejects unsafe archive
members. Use a fresh/empty destination. Default download budget is 2,000,000,000 bytes
(`--max-bytes`). Without an artifact volume, packing also requires free cloud disk;
with one, committed bundles stream to a temporary export on the volume instead. Fetch
is not remote artifact deletion. See [volume lifecycle and migration](controller_volumes.md).

Cancellation retains uncollected worker state; it does not invent patches or discard
recovery data. Inspect `output/sandbox-cleanup.json` and worker artifacts on the trusted
controller before cleanup. Cleanup refuses active jobs and uncollected work unless
`--discard-uncollected` is explicit. Worker safety TTLs still apply during recovery.

## Repair or update an idle controller

```bash
mini-extra benchmark-controller upgrade --project .
mini-extra benchmark-controller check -c context_benchmark_deepseek
```

`upgrade` refuses active jobs and temporarily disables new submissions. It installs the
pinned inference dependencies and frozen project wheel, validates adapter imports, and
updates the controller's code/dependency record. Existing jobs, original results,
fingerprints and credentials are preserved; **no task is retried**. A failed upgrade
leaves maintenance mode enabled for inspection/recovery rather than launching work with
partially updated dependencies. The original clean runtime snapshot is not a snapshot of
the credential-bearing upgraded filesystem; new controllers build the updated image recipe.
Only submit a fresh job after reviewing a failed attempt and authorizing another run.

## Stop, start or delete explicitly

```bash
mini-extra benchmark-controller stop
mini-extra benchmark-controller start
mini-extra benchmark-controller delete
# Only if intentionally abandoning uncollected state:
mini-extra benchmark-controller delete --discard-uncollected
```

`stop` refuses an active runner or owned orphaned child. Stop preserves disk, not process
memory; **no job continues across a controller stop**. Start reuses the existing filesystem
and enables new submissions, but never reruns old jobs. Completed artifacts remain fetchable.
Delete also blocks active work, saves worker recovery journals locally, reconciles owned
workers, and confirms deletion of the controller and its owned runtime snapshot. Preserve
the resulting local manifest/recovery records. Export important results independently;
laptop-disconnect survival and provider availability SLAs are not backup/durability guarantees.

## Source references
- Daytona [SSH access](https://www.daytona.io/docs/en/ssh-access/),
  [lifecycle and TTL](https://www.daytona.io/docs/en/sandboxes/),
  [limits](https://www.daytona.io/docs/en/limits/) and [billing](https://www.daytona.io/docs/en/billing/).
- E2B [SSH workflow](https://docs.e2b.dev/sandbox/ssh-access),
  [continuous-runtime limits](https://docs.e2b.dev/faq/sandbox-lifetime) and [plans](https://e2b.dev/pricing).
