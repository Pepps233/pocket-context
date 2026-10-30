# Persistent controller result volumes

Use one Daytona volume per project to retain managed SWE-bench artifacts independently
of a trusted controller's root disk. This is **cloud-only result storage**, not a coding-VM
filesystem or a way to resume inference. See the [benchmark controller guide](benchmark_controller.md)
for credentials, dataset preparation, detached jobs, billing and worker cleanup.

## Storage and trust boundary

The volume is mounted **only on the trusted controller**, at sandbox creation, at
`/mnt/pocket-context-artifacts`. CodeVMs never receive the mount. Live SQLite databases,
locks, credentials and temporary archive spool stay on the controller's local disk.
Only finalized, closed task outputs are archived; a closed SQLite file can be included
inside a trajectory archive, but no live database runs on FUSE.

The managed store root is `/mnt/pocket-context-artifacts/artifacts-v1`:

```text
artifacts-v1/
  objects/<sha256>.tar.gz                   # immutable content-addressed bundles
  jobs/<JOB>/tasks/<METHOD>/<TASK>.json      # immutable task receipts
  jobs/<JOB>/job.json                       # final job commit receipt
  exports/                                 # temporary streamed fetch bundles
```

Task bundles contain result JSON, run metadata and trajectories. Job bundles contain
public output, plan, redacted log and original final status; the job receipt references
its task receipts. A task prefix alone is **not** a committed complete job.

Publication hashes source members, writes the object, reads it back to verify its hash,
size and members, then writes and verifies the receipt. **Only after verification** are
local bulk `trajectories/` and `trajectories.tar.gz` pruned. Small results, manifests,
receipts and private staged inputs remain local and still consume disk. Immutable-path
conflicts or corruption fail closed rather than overwriting existing data. The store
is a trusted single-writer namespace, not a FUSE lock service or shared database. The
managed controller serializes completed-task uploads using a local thread lock; do not
run concurrent writer controllers against the same namespace.

Private gold patches/tests, `grading_dataset.jsonl`, API keys, credentials and launch
requests do **not** belong in the volume or exported results. Keep them on trusted local
controller storage; never copy them into CodeVMs or a shared artifact namespace.

## Create and inspect without generation

Examples are instructions, not authorization to create cloud resources or invoke models.
Creation and preflight can incur infrastructure charges. Use the intended checkout and
prepared SWE-bench Verified data, with the Daytona extra and trusted credentials ready.

```bash
# New controller: create the named volume or reuse it if it already exists.
mini-extra benchmark-controller create --manifest new.json \
  --project . --volume-name pocket-context-results
mini-extra benchmark-controller volume-info --manifest new.json

# Local planning only: no API calls, uploads, workers or generation.
mini-extra benchmark-controller submit --manifest new.json \
  --data-dir data/swebench-verified --limit 1 --workers-per-method 1 \
  -c context_benchmark_deepseek --dry-run

# Cloud infrastructure check: no model generation; metadata checks need credentials.
mini-extra benchmark-controller submit --manifest new.json \
  --data-dir data/swebench-verified --limit 1 --workers-per-method 1 \
  -c context_benchmark_deepseek --preflight --min-free-bytes 1073741824
```

`volume-info` verifies the actual managed FUSE mount, not merely a directory with the
right name. A missing, wrong or invalid mount fails closed; archival does not silently
fall back to the root disk. Keep `new.json` as the controller identity/recovery journal.
`--dry-run --preflight` can be combined and remains local. Omitting both flags is a paid
inference submission requiring separate authorization; it is not a storage migration step.

## Archival, status and space admission

Volume-backed runs archive each finalized task and then commit job metadata after the
workload exits. The runner reports `archiving` while finalizing and `archive_failed` if
publication fails. The original workload exit code and outcome are retained separately
from archive state: archive success does not turn a failed workload into a successful
one. Inspect status and repair storage, not inference, after an archive failure.

The local free-space admission guard defaults to **1 GiB**. Set a positive threshold
with `submit --min-free-bytes BYTES`; it checks local headroom before execution and new
case allocation. This is **not a reservation or hard per-active-case growth limit**.
Concurrent trajectories, temporary compression spool and FUSE caches can still fill the
root disk between checks. Retained small results/manifests and private inputs also need
ongoing space/admission even when completed bulk trajectories have been pruned.

Start small and measure disk and RAM. Thirty workers per matched method permits up to
**120 concurrent coding VMs**; this is a concurrency ceiling, **not** a 120-worker
controller RAM/load-test result. Volume storage does not increase controller RAM.

## Archive or import existing results; never rerun them

Mounts are selected at controller creation, **not live-attached** by `upgrade` or
`volume-info`. For an existing controller without a volume, wait for all active jobs to
finish, then create a replacement using `new.json`. Preserve the old controller and its
results until the import is committed and verified. Do not stop/delete active work.

```bash
JOB=actual-existing-job-id
SOURCE=actual-old-controller-id

# Inactive legacy/recovered job already on this volume-backed controller.
mini-extra benchmark-controller archive --manifest new.json --job "$JOB"

# Otherwise import from the old trusted controller into the replacement's volume.
mini-extra benchmark-controller import-results --manifest new.json \
  --source-controller "$SOURCE" --job "$JOB" --max-bytes 2000000000
mini-extra benchmark-controller status --manifest new.json --job "$JOB"
```

`archive` requires an inactive terminal job. `import-results` requires a trusted source
controller in the same organization and an inactive source job. It performs a bounded
legacy import **cloud-to-cloud**, with no result-data relay through the laptop. The
source packs public results; the target verifies the checksum and safe members, stages
locally, then commits verified archives. The default compressed transfer budget is
2,000,000,000 bytes (`--max-bytes`); source packing and target staging/extraction still
need local headroom. This is not an unbounded disk-free migration. Original source
results remain intact; retain them until you have verified the target archive/export.

Existing committed archives are immutable. If local output gains recovered or changed
files after commitment, archive/cleanup/delete reject the divergence instead of claiming
those bytes are stored. Preserve the recovery under a separate job namespace/export
before cleanup; do not replace the original failed attempt.

Neither command reruns paid inference, retries benchmark cases, changes predictions or
regrades results. Storage migration is not evidence of a new model smoke or grading run.

## Fetch, recreate and delete explicitly

```bash
mini-extra benchmark-controller fetch --manifest new.json --job "$JOB" \
  --output "runs/context-evaluation/collected/$JOB"

# After inspecting results and reconciling workers; active jobs block deletion.
mini-extra benchmark-controller delete --manifest new.json

# New controller identity, same project volume: committed jobs remain fetchable.
mini-extra benchmark-controller create --manifest recreated.json \
  --project . --volume-name pocket-context-results
mini-extra benchmark-controller fetch --manifest recreated.json --job "$JOB" \
  --output "runs/context-evaluation/recovered/$JOB"
```

For committed jobs, `fetch` streams verified stored bundles into an export archive **on
the volume**, without extracting/repacking trajectories on the controller root disk.
It downloads and verifies that archive before safe local extraction; use a fresh/empty
output directory. The default download budget is also 2,000,000,000 bytes (`--max-bytes`).
A recreated controller can fetch committed jobs from the same named volume without the
old local job directories. Uncommitted local-only state is not recovered this way.

Controller deletion preserves the volume, including when cleaning up failed creation.
Stop/start also preserves it, but does not resume interrupted jobs. Export important
results independently: a persistent volume is **not a backup**.

After independent backup and removal of all mounts, deliberate volume destruction is:

```bash
mini-extra benchmark-controller volume-delete \
  --name pocket-context-results --confirm-destroy-data
```

This is explicit and **irreversible**; controller cleanup never invokes it implicitly.

## Service limits and references

Daytona currently documents volumes as included at no additional cost, separate from
sandbox disk allocation/quota, with **100 volumes per organization**. Use one project
volume, not one per case. Verify current terms before provisioning; this does not imply
unlimited bytes. Per-volume/organization byte caps and FUSE cache disk behavior are not
established here. Do not assume volume capacity or local cache usage from `--disk`.
A backend bucket limit can reject volume creation even below the documented volume
count limit (for example, `You have attempted to create more buckets than allowed`).
Resolve that service limit before deployment; do not silently fall back to controller
disk or repurpose another project's volume. Failed creation retains the requested name
in the controller journal; reconcile it explicitly, including any error-state volume.
A real mounted-volume persistence/recreation test is still required before large runs.

See official [Daytona Volumes](https://www.daytona.io/docs/en/volumes/),
[limits](https://www.daytona.io/docs/en/limits/) and
[billing](https://www.daytona.io/docs/en/billing/) documentation.
