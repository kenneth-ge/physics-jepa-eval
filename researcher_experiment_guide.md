# Pantheon GPU Cluster — Experiment Guide

<!--
This file lives in the modal-skypilot repo, and that is deliberate.

It used to live in a Google Doc and as a Slack attachment. Both drifted: on 2026-08-09 the
Doc copy was still telling agents to set `disk_size` in every template, which is the single
largest cause of failed launches on this cluster. The numbers below are the same numbers the
shim holds as constants in `src/modal_skypilot/mapping.py`, and `tests/test_docs.py` asserts
that they still match. If a test in that file fails, one of the two is stale — find out
which before editing either.

Date every NEW measurement you add here. That is the repo's rule for cluster facts, not a
stylistic choice: a number with no date cannot be told apart from a number that was true
once. Several figures arrived with this document undated — the 47 preemptions behind the
SIGTERM rule, the 4 vCPU / 16 GB an unspecified job is given, the three-preemption resume
— and they are left as they were found rather than stamped with a date nobody measured
them on.

2026-08-12 — cluster changes, not shim changes:
  memory: is now BOTH request and hard limit (set_pod_resource_limits: true). New box in
  §4, a FOR AGENTS bullet, two §6.5 rows, three §9 rows. The units did NOT change with it:
  memory: 1840 still renders 1840G, measured across every post-change pod, so §8's
  decimal-G arithmetic and the §4 table both stand.
  A job submitted BEFORE this change keeps its unbounded spec across every recovery,
  indefinitely — the pod spec is fixed at job submission time. Cancel and relaunch.
  RETRACTED in §5.9: "the cause was never attributed — not OOM". Zero OOMKilling events
  cluster-wide was an invalid test; with no cgroup limit the kernel's global OOM killer
  fires and emits no Kubernetes event.
  zstd images pull 2.7x faster (128s -> 48s); in-cluster skopeo recipe in §5.9.
  config.kubernetes.quota.queue in a TASK yaml does nothing, silently — workspace only.
  /checkpoints is 15 TiB, not 10 (four references).
  Xid 31 measured at 13,156 fleet-wide over three days, 94% of all GPU error events.
  93.13/111.48 is 83.5%, not 84%.
  Still missing: W&B guidance for the external research fellows (personal accounts,
  personal-scoped secrets) is decided but unwritten.

2026-08-14 — cluster changes, not shim changes:
  A CPU nodepool now exists and works. CPU-only jobs need no configuration at all: omit
  accelerators and submit. Part I's "there is no CPU nodepool" note is retracted. The
  detail lives in docs/cpu_only_guide.md, which is deliberately self-contained and
  deliberately states several figures differently (8.7 GiB per core, not 11) — do not
  reconcile the two by copying numbers between them.
  §5.9's prebuilt-image cost is one job shape, not a general figure. ~10 min/launch and
  66 GPU-hours per 50-job sweep describe an 8-GPU job doing a from-scratch CUDA install;
  a 63-job sweep of 1-GPU jobs installing nine small wheels measured 92s of setup and
  1.4 GPU-hours total.
  /checkpoints is 45% used, so "mostly empty" is retracted. Its growth rate is an order of
  magnitude lower than an earlier sweep showed; exact usage and growth figures live in the
  infra tracker rather than here, because a rate quoted beside the volume reads as a size.
-->

## FOR AGENTS: Read this section first

Complete reference for submitting experiments to Pantheon's 128× H200 cluster via SkyPilot.
Install the SkyPilot Agent Skill: `Fetch and follow https://github.com/skypilot-org/skypilot/blob/HEAD/agent/INSTALL.md`

Constraints:
- GPU type is `H200` — remap if the user says H100/A100
- Default priority is `p3` — never set higher without the user asking. `p4` (10) sits *below* it, for work that should yield to everything else
- Use `sky jobs launch` for all unattended work, never `sky launch`
- Do NOT hardcode `infra:` — SkyPilot selects automatically
- Always set `cpus` and `memory` proportional to GPU count: **20 CPU + 230 GB per GPU**. This is the starting point, not a floor — see below
- **Never request more than the table without a measurement.** A node is 175.8 CPU / 1928 GiB across 8 GPUs (≈22 CPU and 241 GiB per GPU, ≈1 CPU per 11 GiB). Asking above that share on a sub-node job leaves GPUs on your node that nobody — including you — can use, and skewed CPU:memory strands whichever resource you under-ask for. Both are invisible to you and surface as somebody else's job failing to schedule
- ⚠️ **`memory:` now sets BOTH the request and a hard limit — the same number for each (changed 2026-08-12).** Whatever you write becomes `requests.memory` *and* `limits.memory`, so it is a ceiling the process cannot exceed, not a hint. Go over it on *anonymous* memory and the container is `OOMKilled`. Page cache is reclaimed first, so cached file reads are still free. There is no separate limit to raise: more headroom means raising `memory:` itself, within the per-GPU share. See §4
- **Request what the job uses, not what looks safe.** After a first run, size from *anonymous* memory in `/sys/fs/cgroup/memory.stat` plus headroom — not from `memory.current`, which counts reclaimable page cache. On 2026-08-09 two jobs requested 1630 GiB and used 54 and 10 GiB of real memory. Do not reduce below the table without that measurement: under-requesting memory OOMs the job and under-requesting CPU starves the dataloader
- For any long run, set `resources.job_recovery.max_restarts_on_errors: 3` so a transient crash restarts instead of ending the job — but only alongside working checkpointing, since each restart starts from the last checkpoint. **Never list 137 in `recover_on_exit_codes`**
- Checkpoints go to `/checkpoints/{PANTHEON_USER}/{EXPERIMENT_TAG}/` (shared volume, survives eviction)
- **Never write output to `/workspace`, `~`, `/root`, or a relative path.** Those land in the container's writable layer, which is shared node disk with no quota — one job wrote 692 GiB there on 2026-08-09 and pushed a node into disk pressure, excluding 8 GPUs from scheduling for everyone. Write to `/checkpoints/...`, `/mnt/raid0`, or `/tmp`
- Bulk/temp data goes to node-local scratch (`/tmp`, or `/mnt/raid0` for anything large) — NEVER set `disk_size` **at any value**, never put an `ephemeral-storage` request in `pod_config`, and never put bulk data on `/checkpoints`. A node has only 111.5 GiB of ephemeral-storage across all 8 GPUs, so `disk_size: 100` blocks 7 of them; `/tmp` gives ~1.5 TiB and needs no request. Lowering the value does not help — remove it
- Platform: https://pantheon.svc.skypilot.co

---

---

# Part I — What every job must do

*(This part is Sambhav's; it's about how to write a job. Part II is the cluster
mechanics — how to size, submit and store.)*

## Writing code for a job

Jobs should be written as Modal jobs, so they can execute on either our infra or
Modal. Modal jobs run on our infra via the
[`modal-skypilot`](https://github.com/Pantheon-Industries-Inc/modal-skypilot)
package — a shim that converts Modal code into the equivalent SkyPilot YAML and
launches it. Expect SkyPilot infra to be the main target.

## Preemption safety

**Expect all jobs to be preemptible.** A job must lose no more than **5 minutes
of compute** if preempted. For training, that means the dataloader, optimizer and
model are all checkpointed to non-ephemeral disk. For data processing, workers
must register completed work atomically and resume from arbitrary states.

This is what allows aggressive scheduling, including defragmenting jobs that take
less than a node.

Jobs must also be robust to hardware failure — especially multi-node. A job must
not corrupt itself or its past results when anything other than non-ephemeral
storage fails.

⚠️ **Do not write a SIGTERM handler and rely on it.** Measured across 47 real
preemptions on this cluster: no signal reaches the process, it ends at SIGKILL.
Durability has to come from checkpoint frequency, not from a graceful-shutdown
path. See §5.

## MFU

Jobs should attain high MFU — at worst 10%, ideally 40%+. Make CPU work async
from GPU work.

- CPU-only data processing should run on CPU machines, and as of 2026-08-14 the
  cluster has them. **Write the job with no `accelerators` and submit it
  normally** — no `nodeSelector`, no workspace switch, nothing to configure. It
  runs on a CPU machine if one is up and on a GPU node if none is; neither case
  fails or hangs. The sizing rule there is different (~8.7 GiB per core, not
  230 GB per GPU) and so are several other numbers, so use
  **`docs/cpu_only_guide.md`** for anything CPU-only rather than this document.
- Dataloaders should prefetch and never block the GPU. If dataloading blocks the
  GPU, resize the model or the GPU count until it doesn't. `/mnt/raid0` (§5) is
  16.6× faster than the shared volume on small random reads and is usually the
  right fix.

## Performance and durability via queueing

Use producer–consumer queues:

- **Performance** — things that needn't wait for each other shouldn't. Checkpoint
  writes should be async.
- **Durability** — queues checkpoint progress durably.
- **Consistency** — always append data before updating metadata, so preemption
  cannot leave a ghost shard that never gets processed.

**Do not use Python threading for parallelism — only multiprocessing.** Keep it
simple and deadlock-free; prefer vendored queue abstractions.

## W&B

W&B logging governs the quality of information an experiment yields.

On preemption, or a deliberate pause and resume, **reuse the same W&B run** — not
a fresh one. See the `WANDB_RESUME` / `WANDB_RUN_ID` pattern in §5.

Be precise with naming. No vague run names like "training", and no extraneous
hyperparameters that aren't being tuned. Run and project names should convey what
the project is and what the run is testing.

## Training telemetry contract (`training-v1`)

A custom SkyPilot dashboard expects telemetry from every running job, and
complains loudly without it.

Every active GPU training job must:

- Publish a W&B run URL discoverable by SkyPilot.
- Refresh these W&B summary fields every ~10s:
  - `pantheon_telemetry/contract_version: training-v1`
  - `pantheon_telemetry/heartbeat_at` — Unix timestamp
  - `pantheon_telemetry/total_steps` — positive integer
  - `pantheon_telemetry/active_step` — current step
  - `pantheon_telemetry/step_time_s` — latest step duration
  - `pantheon_telemetry/effective_mfu` — fraction in `[0, 1]`
  - `pantheon_telemetry/wandb_url` — canonical run URL
- Log scalar history under the first metric matching `train/loss*`; log
  `val/loss` when validation begins.
- Report **only from global rank 0**. Heartbeats older than 30s are violations.

Progress is `active_step / total_steps`. Missing links, missing fields, missing
train loss, invalid values or stale heartbeats all produce loud dashboard errors.

Non-training jobs must explicitly opt out with a reason:

```yaml
resources:
  labels:
    telemetry-contract: opt-out
    telemetry-opt-out-reason: embedding-extraction
```

**MFU must be end-to-end "effective MFU", not compute-only.** Over all steps, the
sum of step times in the denominator must be nearly equal to the length of the
run. A compute-only MFU is not an accurate reflection of how well we are using
the compute. A separate compute-only MFU may be logged to assess the ceiling once
dataloading stalls are eliminated, but must not go to telemetry.

## Data processing jobs

Data processing jobs should be producer–consumer queues: consumers take many
small units of work, process, write out, then atomically update metadata. An
orchestrator assigns tasks dynamically so no worker waits in a long tail.

## InfiniBand

InfiniBand setup is involved — **use `modal-skypilot` and let it emit the
config**. Validated 2026-08-09 at 437.5 GB/s busbw with GDRDMA on a converted
2-node job. Do not hand-copy the block; see §6.

---

# Part II — Cluster reference

## 1. Setup (~5 minutes)

```bash
pip install "skypilot[kubernetes]"
export SKYPILOT_API_SERVER_ENDPOINT=https://pantheon.svc.skypilot.co
sky api login
echo 'export SKYPILOT_API_SERVER_ENDPOINT=https://pantheon.svc.skypilot.co' >> ~/.zshrc
```

---

## 2. Cluster

| Detail | Value |
|---|---|
| GPUs | 128× NVIDIA H200 (16 nodes × 8 GPUs) |
| Accelerator string | `H200` |
| Per-node resources | 175.8 CPU, 1928 GiB RAM, ~14 TiB of NVMe (see below) |
| Per-GPU share | 21.9 CPU, 241 GiB — the divisor everything in §4 is about |
| Platform + Dashboard | https://pantheon.svc.skypilot.co |

The NVMe figure is **not** an `ephemeral-storage` quota and must never be requested as one:
~1.8 TiB of it is the container filesystem you already have at `/tmp`, and 12.2 TiB is the
opt-in `/mnt/raid0` array (§5). What Kubernetes advertises as `ephemeral-storage` is a
different, much smaller device — the node's boot disk, of which **111.5 GiB is
allocatable**. See §5.

---

## 3. Priority

| Name | Value | Use | Default |
|---|---|---|---|
| p0 | 1000 | Large PFM runs only (Mo/Sambhav) | |
| p1 | 80 | Time-critical ablations ≤32 GPU | |
| p2 | 60 | Standard ablations & sweeps | |
| p3 | 20 | Single-GPU experiments, sweeps | ✓ |
| p4 | 10 | Below the default — backfill and opportunistic work that should yield to everything else | |

```bash
sky jobs launch experiment.yaml                        # p3 (default)
sky jobs launch experiment.yaml --priority p1          # higher priority
```

**Always use the tier names, never bare numbers.** The CLI silently accepts any
integer in ±1000: a bare `--priority 2` lands *below* p3 (=20) — last in the
admission line and first evicted — and a hand-rolled `900` silently outranks
every p1 on the cluster. Both happened. Names only.

---

## 4. CPU / RAM sizing

Kubernetes gives unspecified jobs almost nothing (observed: 4 vCPU / 16 GB on a
1-GPU job) — enough to starve any real dataloader. Size to GPU count:

| GPUs | cpus | memory (GB) |
|---|---|---|
| 1 | 20 | 230 |
| 2 | 40 | 460 |
| 4 | 80 | 920 |
| 8 (full node) | 160 | 1840 |

Rule: `cpus = 20 × N`, `memory = 230 × N`. The templates below already carry
these. Going higher than the table risks an unschedulable pod (node allocatable
is ~176 CPU / ~1928 GiB).

⚠️ **This table is for GPU nodes only.** A CPU machine gives ~8.7 GiB per core
against a GPU node's ~11, so `230 × N` produces an unschedulable pod there. For
CPU-only jobs use `docs/cpu_only_guide.md`, which sizes as `memory ≈ 8 × cpus`.

> ### ⚠️ Changed 2026-08-12: `memory:` is enforced, and it is also your limit
>
> Until 8/12, `memory:` was a **scheduling hint with nothing behind it** — a container
> could exceed its request indefinitely. One did: it climbed to ~1872 GiB of 1929 GiB,
> the node's page cache collapsed, and kubelet and containerd were starved into
> failure. **Four nodes lost `Ready` without rebooting.** The kernel's own protection
> made it worse, not better: a large memory request *lowers* a pod's OOM score, so the
> offending job was the last thing the kernel would kill and it took a 2.6 GiB
> infrastructure pod instead.
>
> `kubernetes.set_pod_resource_limits: true` is now set cluster-wide, so
> **`limits.memory` = `requests.memory`.** What that means for you:
>
> - **Page cache is reclaimed first.** If your usage is mostly cached file reads,
>   nothing changes — the kernel drops them and you keep running.
> - **Anonymous memory over your number now kills the container.** `OOMKilled`,
>   immediately and attributably. A job that quietly ran 5% over for weeks will die.
> - **This is a better failure than the old one.** Previously your job survived and
>   took the node — and everyone else's jobs on it — down with it.
> - **The units did not change.** `memory: 1840` still renders `1840G` (1713 GiB), so
>   the sizing table above is unchanged and a whole-node job still leaves ~215 GiB for
>   the system. Measured across every post-change pod on the cluster.
>
> If a job starts dying at a number that used to work, this is why. Check
> `memory.events` (below) — a nonzero `oom` or `max` confirms it — then size to **anon
> plus headroom** rather than adding 10% and hoping.
>
> ⚠️ **A job submitted before 2026-08-12 does not get a limit, and never will.** The
> pod spec is fixed at *job submission* time, not pod creation time, so a managed job
> from before the change regenerates an unbounded pod on **every recovery,
> indefinitely** — restarting or preempting it does not help. Measured 8/12: a clean
> split by job ID, everything below one number unbounded and everything above it
> bounded, while every pod was hours old. **Cancel and relaunch** long-running jobs
> from before 8/12. Until you do, those are the jobs that can still take a whole node
> down, and it lands on everyone else on that machine rather than on you.

### Keep CPU and memory in the node's proportion

A node is **175.8 CPU : 1928 GiB — roughly 1 CPU per 11 GiB.** Request far from
that ratio and you strand the *other* resource on your node for everyone else.

Measured 2026-08-09: jobs running 64 CPU / 1630 GiB (1:26) left nodes with 46–74
CPU free and 5–22 GiB of memory, while guide-shaped jobs left the mirror image —
212 GiB free with 14 CPU. A 1-GPU job asking 24 CPU + 168 GiB found **zero of 16
nodes** could take it, with nine GPUs sitting idle. It looks like a scheduler
problem and is not.

The `20 × N` / `230 × N` table is already in proportion. Stay on it unless you
have measured a reason not to.

### Anything you write outside a mounted volume fills a shared node disk

Your container's filesystem is **not yours** — it is a writable layer on the
node's 1.8 TiB NVMe, shared with every other pod on that machine, with no quota.
That includes:

- **`/workspace`** — the NGC PyTorch image's default `WORKDIR`, so *plain
  relative paths land here*
- `/root`, `~`, `/opt`, and anywhere else not listed under `volumes:`

When that disk passes 85% used, kubelet declares disk pressure and **the whole
node stops accepting work** — every other job on it is affected, and the node is
excluded from scheduling until it recovers.

Measured 2026-08-09: one job held **692.5 GiB in `/workspace`**, 93% of its
node's container-filesystem usage. Another node crossed the threshold the same
evening and had **8 GPUs excluded** from scheduling while it drained.

Write instead to:

| For | Path | Notes |
|---|---|---|
| Anything you need after the job | `/checkpoints/$PANTHEON_USER/$EXPERIMENT_TAG/` | durable, replicated |
| Large scratch, dataset cache | `/mnt/raid0` | 12.2 TiB/node, free, fast — see §5 |
| Small temp files | `/tmp` | node NVMe, shared and unpoliced — keep it small |

Check your own job with `du -sh /workspace` before you scale it up.

### Why going over the table costs other people GPUs

A node has **1928 GiB across 8 GPUs = 241 GiB per GPU.** That is the physical
share. Ask for more than 241 GiB per GPU and the excess is taken from the GPUs
you did *not* request — they stay physically idle and **no other job can ever
land on them**, because there is no memory left to run with.

Measured on 2026-08-09: two 4-GPU jobs requesting `memory: 1750` (1630 GiB,
407 GiB/GPU) each stranded 4 GPUs on their node. Across six nodes, **11 of 16
free GPUs were unusable by any 1-GPU job** while 44 workloads sat pending. CPU
was never the constraint — every node had 55+ cores spare. It looked like a
scheduler problem and was not.

It got worse before it got better. Measured **2026-08-10**: twelve sub-node jobs were
asking **1676 GiB for 4 GPUs** (419 GiB/GPU), one of them 1676 GiB for **3** (559
GiB/GPU). Result: **24 of 128 GPUs idle**, with about 1,088 GiB of free memory spread
across the nodes holding them — enough for roughly five of the twenty-four. **Zero of
sixteen nodes** could have fitted another 4-GPU job at this guide's `memory: 920`.

Five of those twelve were checked from the inside, `/sys/fs/cgroup/memory.stat`: **11,
11, 0, 58 and 9 GiB anon**, with `memory.events` `max = 0` on every one. None had ever
come near its limit. All twelve were hand-written SkyPilot YAML rather than converted
Modal code, which is why `modal-skypilot convert` never got a chance to say so — and why
`modal-skypilot submit` now reads `resources:` out of a hand-written document and warns
with the same arithmetic.

If your memory-per-GPU needs to exceed ~241 GiB, the honest options are to take
a **full node** (8 GPUs, 1840) or to use **fewer GPUs with the same memory** —
not to reserve a neighbour's share.

### Do you actually need that much? Usually not — check before asking

Linux counts **page cache** in a container's memory usage, and page cache is
**reclaimable** — the kernel drops it under pressure and your job keeps running.
Only *anonymous* memory (real allocations) is unreclaimable. A job showing
700 GiB of "usage" may be holding 50 GiB of real memory and 650 GiB of cached
file reads it never needed reserved.

Check your own running job:

```bash
kubectl exec <your-pod> -- sh -c '
  echo "current: $(( $(cat /sys/fs/cgroup/memory.current) / 1073741824 )) GiB"
  awk "/^anon /{printf \"anon (real, unreclaimable): %d GiB\n\", \$2/1073741824}" /sys/fs/cgroup/memory.stat
  awk "/^file /{printf \"file cache (reclaimable): %d GiB\n\", \$2/1073741824}" /sys/fs/cgroup/memory.stat
  echo "--- has it ever hit the ceiling? max>0 means yes ---"
  cat /sys/fs/cgroup/memory.events'
```

Size `memory` to **anon plus headroom**, not to `current`. If `max` in
`memory.events` is `0`, your job has never come close to its limit and the
request is too high.

Real examples from 2026-08-09, same cluster:

| Job | GPUs | requested | anon (real) | file cache | verdict |
|---|---|---|---|---|---|
| A | 4 | 1630 GiB | **54 GiB** | 630 GiB | 30× over |
| B | 5 | 1630 GiB | **10 GiB** | 584 GiB | 163× over |
| C | 4 | 1118 GiB | **643 GiB** | 299 GiB | honest, leave alone |

Job C is a legitimate exception — 643 GiB of real memory with a sensible margin.
The rule is not "always request less," it is **request what you use.**

---

## 5. Storage & checkpoints

Three tiers — use the right one:

| Tier | What | Survives eviction | Use for |
|---|---|---|---|
| `/checkpoints` volume | **15 TiB** shared filesystem, all nodes (expanded from 10 on 2026-08-10; expansion is one-way) | ✅ | checkpoints and small durable state ONLY |
| Node scratch (`/tmp`) | ~1.5 TiB free local NVMe, already mounted | ❌ | datasets, temp files, anything bulk |
| `s3://encodings` bucket | Crusoe object storage | ✅ | bulk data in/out of the cluster |

### Choosing: local vs volume vs bucket

The underlying difference: a **volume is a filesystem**, a **bucket is a
warehouse with an API**. Everything else follows. Ask three questions in order:

**1. Does the data only matter while this job runs?** → node-local NVMe: just write to `/tmp`.
Fastest option on the cluster, costs nothing, and eviction wiping it is fine
because the job was its whole lifetime. Staged datasets, decompression scratch,
intermediate files.

**2. Must the job find it again after eviction, at a normal file path?** →
volume (`/checkpoints`). It's a real filesystem: `torch.save` works directly,
renames are atomic, no credentials, and the eviction-recovery contract (job
relaunches → `load_latest_checkpoint()` → resumes) depends on the files being
at the same path on whatever node the job lands on. Checkpointing to a bucket
instead is actively worse: SIGTERM mid-upload leaves a truncated object and a
corrupt resume.

**3. Does it cross the cluster boundary, or is it bulk?** → bucket
(`s3://encodings`). The bucket is the *only* door in or out of the cluster —
hades staging, embeddings from external machines, artifacts you keep after the
job. It's also the only place bulk belongs: effectively unlimited (256 TiB org
quota, billed per byte stored), whereas the volume is a fixed 15 TiB **shared
by everyone** and billed on provisioned size — one 5 TiB dataset parked there
puts every job's checkpoint saves on the cluster at ENOSPC risk simultaneously.

Rule of thumb: **local for during-the-job, volume for across-evictions, bucket
for across-the-boundary and anything big.** A typical training job uses all
three in one YAML — sync a slice from the bucket to `/tmp` scratch, train
reading local, checkpoint to the volume.

(The friction difference is deliberate: the volume is zero-ceremony because
checkpoints are touched constantly; the bucket needs the three env exports
because its contents are touched once per run.)

One hard rule: **checkpoints go on `/checkpoints`, bulk data does not.** The
volume is shared and capped — filling it breaks checkpoint saves for every job
on the cluster at once.

**Creating volumes** (if you make your own): size must be ≥ 1 TiB in whole-Ti
steps — Crusoe's provisioner rejects anything smaller (binary TiB: 1000Gi
fails, 1024Gi works), and an invalid size wedges permanently NOT_READY instead
of erroring. And **never enable Auto Mount**: it injects the volume into every
new job on the cluster, so one broken volume blocks all scheduling (this took
the queue down on Aug 5). Mount explicitly in your YAML instead.

### Shared checkpoint volume

Mount it in your YAML:
```yaml
volumes:
  /checkpoints: checkpoints
```

Path convention:
```
/checkpoints/{your-email}/{experiment-tag}/step_000500.pt
```

Example: `/checkpoints/mo@pantheon.inc/vit-lr3e4-aug04/step_001000.pt`

- **PANTHEON_USER** = your @pantheon.inc email (no collision with other researchers)
- **EXPERIMENT_TAG** = unique per experiment. Reusing a tag deliberately = resuming
  that run's checkpoints. Reusing one accidentally = corrupting them. When in
  doubt, mint a new tag and date-stamp it.

⚠️ **`chmod 777` every directory you create on `/checkpoints`, immediately.**
Container images differ in default user — `nvcr.io/nvidia/pytorch` runs as
**root**, SkyPilot's default image runs as **`sky` (uid 1000)**. Whichever image
first creates `/checkpoints/<you>/` owns it, and a later job under the other
image gets `mkdir: Permission denied` *inside its own directory*. The volume
root is already 777; keep your subdirectories the same:

```bash
mkdir -p "$D" && chmod 777 "$D"
```

### Creating your own volume

Only if `/checkpoints` doesn't fit — it's RWX, 15 Ti, and **45% used as of 2026-08-14**, so
there is still room but it is no longer close to empty. Expansion is one-way.

```yaml
name: my-volume
type: k8s-pvc
infra: kubernetes
size: 1Ti                              # 1 TiB MINIMUM, whole-Ti steps only
config:
  namespace: default
  storage_class_name: crusoe-sharedfs  # RWX-capable class
  access_mode: ReadWriteMany           # REQUIRED for multi-node
```

```bash
sky volumes apply my-volume.yaml
sky volumes ls          # confirm READY before submitting anything
```

Three ways this bites if you skip the explicit lines:

- **Omit `access_mode`** → you get `ReadWriteOnce`, and any multi-node task is
  rejected at submit: `Volume X with access mode ReadWriteOnce is not supported
  for multi-node tasks`. **Access mode is fixed at creation** — the only fix is
  a new volume.
- **Omit `storage_class_name`** → you may land on an RWO-only class.
- **Size below 1 TiB, or not a whole-Ti step** → the PVC goes permanently
  `NOT_READY` **silently**, with no error at creation. Every job that mounts it
  then holds GPU quota it can never use. This took the queue down on Aug 5.

Never enable **Auto Mount** — it injects the volume into every new pod
cluster-wide.

### Local scratch (node NVMe)

Nothing to request — **the container filesystem already is the node's NVMe.**
Write to `/tmp` (or anywhere on `/`) and you get roughly **1.5 TiB free**:

```
/dev/nvme0n1   1.8T  170G  1.5T  11%  /
```

**Do not set `disk_size`, and never request `ephemeral-storage` in
`pod_config` either.** `disk_size` maps to Kubernetes *ephemeral-storage*, of which
the node has very little — and the two numbers `kubectl describe node` prints are not the
same number:

| | Value | What it is |
|---|---|---|
| `capacity` | `129886128Ki` = **123.9 GiB** | the boot disk `/dev/vda1` — not the NVMe |
| `allocatable` | `119703055367` = **111.5 GiB** | capacity × 0.9, after the default 10% `nodefs.available` eviction reserve |

**These are two different gates, and a value can pass the first and fail the second
forever.**

* `capacity` gates **submit**. `sky` compares the raw `disk_size` number against
  capacity ÷ 2³⁰ = **123.9**, so anything above that is rejected in about two seconds with
  `FAILED_PRECHECKS` — no pod, no workload, no queue row. That silently cost the team 20+
  failed launches over two days.
* `allocatable` gates **scheduling**. The pod is rendered `ephemeral-storage: <n>G`,
  Kubernetes reads a bare `G` as 10⁹, and kube-scheduler compares that against
  **111.5 GiB**. So the real ceiling for a pod that will ever run is ~119 GB.

Which leaves a trap in between: **`disk_size` 120–123 passes every check and then never
schedules** — no error, no event, the job simply pends. That is the same failure shape as
an `ephemeral-storage` request in a `pod_config` (below), reached through the supported
field.

**But the values that pass are the expensive ones**, and that is the part worth
internalising:

A node's whole ephemeral-storage allocatable is **111.5 GiB shared across all 8 GPUs**, so
one GPU's proportional share is about **14 GiB**. A `disk_size: 100` is a **93.1 GiB**
request — **83.5% of an entire node, for a single-GPU job.** It schedules happily, and then
the node's other **7 GPUs cannot be used** by anything that also requests
ephemeral-storage.

Measured **2026-08-11**, with sixteen 1-GPU jobs at `disk_size: 100` on the cluster:

| node | GPUs in use | eph free of 111.5 GiB | GPUs blocked |
|---|---|---|---|
| node-1 | **1/8** | 18.4 | **7** |
| node-4 | **1/8** | 18.4 | **7** |
| node-9 | 5/8 | 51.9 | 3 |
| node-10 | 7/8 | 22.1 | 1 |
| node-14 | 7/8 | 18.4 | 1 |

**19 GPUs idle**, while 26 workloads sat pending — all of them also requesting ≥60 GiB.
Every node with free GPUs had no disk quota left; every node with disk quota was full on
GPUs. Nothing could move.

⚠️ **Lowering the number does not fix this.** 80G still leaves only 37 GiB, which is not
enough for a second such job — so it is still one job per node. There is no safe non-zero
value at this node size. **Remove the line.**

Beware the platform tooltip here: it reports *"the largest node allocatable is 111.5Gi"*,
which reads as advice to use 110Gi or less. 110Gi is 98% of a node and reproduces exactly
the deadlock above.

Crusoe confirmed on 2026-08-06 that this is expected CMK behaviour with no setting to
repoint it at the NVMe. A node-image change that would have moved kubelet's root-dir onto
the NVMe array — after which these numbers change completely — was targeted for 2026-08-12
and **deferred on 2026-08-11 with no ETA**: it is being reworked to sit behind an explicit
opt-in flag and needs further testing. So this is not a trap that expires. **Do not request
ephemeral-storage at all**, and do not wait for the numbers to get better.

`modal-skypilot` refuses Modal's `ephemeral_disk=` at **every** size for this reason, and
`modal-skypilot submit` scans a hand-written YAML for both spellings and warns.

⚠️ **The `pod_config` form of this is worse than `disk_size`.** An explicit
`resources.requests.ephemeral-storage` in `config.kubernetes.pod_config`
*passes* prechecks, then never schedules: kube-scheduler reports `0 pods fit`
on every node and the job pends indefinitely with no error anywhere. On
2026-08-09 this held 53 jobs for 7+ hours while half the cluster sat idle, and
it presented as a bin-packing problem. Pods are immutable, so the only fix is
to cancel, remove the stanza, `sky down`, and relaunch.

Fast, free, node-local — and wiped on eviction. Stage datasets here, write temp
files here.

### `/mnt/raid0` — 12.2 TiB of free node-local NVMe (new, 2026-08-09)

Each node has 8 NVMe drives. One backs the container filesystem (that's what
`/tmp` is); the other seven are now a RAID0 array mounted at `/mnt/raid0`,
**12.2 TiB per node, free** — the capacity is already paid for in the node.
Live on **all 16 nodes**, ~195 TiB cluster-wide. (It was 13 of 16 for most of 2026-08-09;
the rollout's second phase finished the same afternoon. Keep the `nodeSelector` below
anyway — it selects nothing out today and it is what keeps the `hostPath` honest the day a
node is rebuilt or added without the array.)

Measured on the array (fio, 8 jobs, `direct=1`):

| | `/mnt/raid0` | `/tmp` (1 drive) | `/checkpoints` (NFS) |
|---|---|---|---|
| Sequential write | **20.1 GB/s** | 2.88 GB/s | 5.15 GB/s |
| Sequential read | **39.3 GB/s** | 6.05 GB/s | 21.3 GB/s* |
| **Random read 4k** | **7458 MiB/s** | 4390 MiB/s | 448 MiB/s |

\* warm cache, treat as an upper bound.

**Random read is the number that matters for a dataloader — 16.6× better than
`/checkpoints`.** If your job is I/O bound on small reads, this is the fix.

To use it, add a `nodeSelector` and a `hostPath` mount:

```yaml
config:
  kubernetes:
    pod_config:
      spec:
        nodeSelector:
          pantheon.inc/raid0: "enabled"
        volumes:
          - name: raid0
            hostPath:
              path: /mnt/raid0
              type: Directory        # load-bearing, see below
        containers:
          - volumeMounts:            # no `name:` key on this entry
              - mountPath: /mnt/raid0
                name: raid0
```

`type: Directory` is not optional. Without it, kubelet silently creates an empty
directory on any node lacking the array and your job writes to the boot disk
while looking healthy. With it you get a loud `FailedMount`.

⚠️ **This is a cache, not storage. Everything on it must be reconstructible.**

- **RAID0 across 7 drives** — one drive failure loses the whole node's array.
- **It does not survive a reboot or node replacement.** The array is recreated
  **empty**. A job that finds its data missing must be able to re-fetch it.
- **No quotas, `chmod 777`.** One greedy job fills 12.2 TiB and evicts its
  neighbours' data. Clean up after yourself.
- **The scheduler doesn't know which node holds what.** A pod can land on a node
  with a cold cache. Write for a miss: pull from object storage or `/checkpoints`
  and populate on first use. The object-storage leg runs at **~15 GB/s with a Go
  client** and **~1.4 GB/s from Python threads** (measured 2026-08-11, see the bucket
  section) — so a cold 1 TiB slice is about **70 seconds** done well and **13 minutes**
  done with default boto3. Which client you use is the difference between a miss being
  free and a miss being felt.
- **Same 777 subdirectory trap as `/checkpoints`** — a directory created by a
  root-default image (`nvcr.io/...`) is unwritable by `sky`-default jobs
  (uid 1000). `chmod 777` anything you create for shared use.

Good fits: dataset slices, HF/model caches, embedding shards, checkpoint staging
before a durable write. Bad fits: the only copy of anything. Anything you want to keep goes to `/checkpoints` or a bucket before
the job ends.

⚠️ **Both node-local tiers are unpoliced**, and they are different sizes: `/tmp` has
~1.5 TiB free and `/mnt/raid0` has 12.2 TiB. Neither has a per-job quota, and filling
either triggers disk-pressure evictions for your neighbours as well as yourself. Clean up
large intermediates as you go.

### Reading the encodings bucket (bulk data)

```yaml
run: |
  export AWS_ACCESS_KEY_ID=$CRUSOE_S3_ACCESS_KEY AWS_SECRET_ACCESS_KEY=$CRUSOE_S3_SECRET_KEY
  export AWS_ENDPOINT_URL=https://object.eu-iceland1-a.crusoecloudcompute.com
  aws s3 sync s3://encodings/<prefix>/ /tmp/data/
  python train.py --data /tmp/data
```

Keep the slice you pull under the ~1.5 TiB of free node NVMe. For prefixes with thousands of
objects, `s5cmd` is ~10× faster than `aws s3 sync`.

**How fast this is depends almost entirely on your client.** Measured 2026-08-11, same
GPU node, same bucket, same objects:

| Client | Throughput |
|---|---|
| `warp` (Go), 64 MiB objects, 64 concurrent | **15.19 GB/s** |
| Python `boto3`, 48–384 threads | **1.4 GB/s** |

**A 10× difference, and adding threads made Python worse** — 1.10 GB/s at 48 workers,
1.00 at 256, while the process used only 6.6 of 32 cores. That is the client serialising,
not the store.

⚠️ **An earlier version of this guide said "budget ~1.3 GB/s and do not plan around more",
and dismissed a higher figure as unreproducible. That was wrong** — 1.3 GB/s was a boto3
ceiling mistaken for a storage ceiling. Corrected here rather than deleted, because the
1.3 number is still exactly what you will measure if you benchmark with default boto3
threads, and knowing why is the useful part.

So if your dataloader streams from `s3://encodings`:

- **Use multiple processes**, not a `ThreadPoolExecutor`. One boto3 client per process,
  shard the key list across them.
- Or shell out — `s5cmd`, `mc` and `warp` are Go and saturate the link.
- If you see ~1.4 GB/s, you are looking at the Python ceiling, not the storage.

Two nodes reading concurrently showed no contention (each *above* its own solo rate), so
the ceiling is per client rather than global — more readers move more data.

Instance type matters too: a `c1a.32x` topped out around 2.2 GB/s where the GPU node
reached 15.19. Crusoe confirmed 2026-08-11 that host network bandwidth is allocated
proportionally to VM size, and full-host GPU instances are typically single-tenant, so they
get close to the full NIC.

### Checkpoint code pattern

```python
import os, torch

CKPT_DIR = f"/checkpoints/{os.environ['PANTHEON_USER']}/{os.environ['EXPERIMENT_TAG']}"
os.makedirs(CKPT_DIR, exist_ok=True)

def save_checkpoint(model, optimizer, step):
    path = f"{CKPT_DIR}/step_{step:06d}.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step}, path)

def load_latest_checkpoint(model, optimizer):
    if not os.path.exists(CKPT_DIR):
        return 0
    files = sorted(f for f in os.listdir(CKPT_DIR) if f.startswith("step_"))
    for f in reversed(files):
        try:
            ckpt = torch.load(f"{CKPT_DIR}/{f}", weights_only=False)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            return ckpt["step"]
        except Exception:
            continue
    return 0

# Training loop
start_step = load_latest_checkpoint(model, optimizer)
for step in range(start_step, total_steps):
    train_one_step()
    if step % 500 == 0:
        save_checkpoint(model, optimizer, step)
```

Keep the volume lean: rotate old step files (keep last N + milestones). 15 TiB
is shared across everyone. There is **no automatic drain to hades** yet —
long-term checkpoint retention is manual for now; treat the volume as working
state, not an archive.

### What happens on eviction

1. Higher-priority job submitted → Kueue preempts your workload
2. Your pod is killed → SkyPilot marks the job RECOVERING
3. GPUs free up → SkyPilot re-launches on a new node
4. Your job starts, mounts `/checkpoints`, loads the latest checkpoint, resumes

No re-queuing logic needed from you. Save + load checkpoints. **Verified on this
cluster:** a job preempted three times resumed from step 81, then 122, each time
picking up exactly where the previous life ended.

### You get no warning — do not write a SIGTERM handler

Measured across 47 real preemptions on this cluster (three separate jobs), **a
`SIGTERM` handler in your training script never runs.** Traps on both `TERM` and
`EXIT` produced no output in any case. The user script is not PID 1, the signal
is not forwarded to it, and the process ends at SIGKILL — which no handler can
intercept.

So there is **no save window**. Plan for the process to stop between one
instruction and the next:

- Checkpoint on a **step interval you can afford to lose** (e.g. every 500
  steps), not on a shutdown hook.
- Write atomically — `torch.save` to a temp path, then `os.replace` — so a kill
  mid-write can't leave a truncated checkpoint as `latest`.
- Assume the last partial interval is gone. That is the cost of running at p3,
  and it is normal.

There is a second reason this matters: a job that ends on a **nonzero exit code**
is classified by SkyPilot as a *user program failure* and is **not retried**,
while a job whose pod simply disappears is recovered. A slow or blocking
shutdown path makes the bad outcome more likely, not less.

### wandb run continuity

Add these so eviction doesn't fragment your wandb charts:
```yaml
envs:
  WANDB_RESUME: allow
  WANDB_RUN_ID: my-experiment-tag   # same as EXPERIMENT_TAG
```

### Secrets

Pre-configured and global, so **you do not create these**:
`CRUSOE_S3_ACCESS_KEY`, `CRUSOE_S3_SECRET_KEY`, `WANDB_API_KEY`. Corroborated by a live
job on 2026-08-06, and the list `modal-skypilot`'s `config/secret-map.yaml` is allowed to
name — `tests/test_config.py::PLATFORM_SECRETS` holds it to exactly these three, because a
mapped secret suppresses the "create this first" action an unmapped one would raise, and a
stale entry there puts "Ready to submit" over a launch that fails.

Reference one by name; do not paste its value. That is the `secrets:NAME` form, and it is
what `modal-skypilot` emits:

```yaml
secrets:
  - secrets:WANDB_API_KEY          # a REFERENCE; the API server resolves it at launch
  - secrets:CRUSOE_S3_ACCESS_KEY
  - secrets:CRUSOE_S3_SECRET_KEY
```

Anything else you need, add at
https://pantheon.svc.skypilot.co/dashboard/secrets-manager — "Add Secret" → "Env Var" →
name it → paste the value — and then tell `modal-skypilot` about it by adding a line to
`config/secret-map.yaml`, or `convert` will raise an ACTION REQUIRED for it every time.

---

## 5.9 Retrying transient crashes, and separating teams

### `max_restarts_on_errors` — a crash currently ends your job

**Hardware failures and preemptions are auto-recovered. Your code exiting non-zero is
not** — the job goes FAILED and stops. So a crash from a transient cause (an NCCL timeout,
a driver hiccup, a flaky read) ends a run that a restart would have saved.

Measured 2026-08-10: **32 of 153 jobs in one sweep** ended with `exit code 137` after long
stretches of healthy training, with `Task 0 failed and will not be retried` and 0
recoveries. Every one of those runs would have survived a retry.

> **Updated 2026-08-12 — it was almost certainly OOM, and the test that ruled it out was
> invalid.** This paragraph used to say the cause was never attributed, listing "not OOM
> (zero `OOMKilling` events cluster-wide)" first. **That test does not work.** Kubernetes
> emits `OOMKilled` for a *cgroup* limit breach; until 8/12 no pod had a memory limit, so
> there was no cgroup bound to breach and the kernel's **global** OOM killer fired
> instead — which produces no Kubernetes event at all. Absence of the event was not
> absence of the OOM. The other exclusions still stand: no `Evicted`/`Preempted`
> condition on any of 74 Workloads, no disk eviction, no kubelet restarts.
>
> Exit 137 is SIGKILL. With limits enforced (§4), a container that exceeds its memory
> number is now `OOMKilled` *attributably* — it shows in `kubectl describe pod` and in
> `memory.events`. **If you get a 137 and both of those are clean, that is a different
> cause and worth reporting in #training-infra.**

```yaml
resources:
  accelerators: H200:8
  job_recovery:
    max_restarts_on_errors: 3    # restart up to 3x on ANY non-zero exit
```

Each restart runs on a **newly provisioned cluster**, so this only helps if you
checkpoint — without it a restart begins from step 0.

⚠️ **Never put 137 in `recover_on_exit_codes`.** SkyPilot reserves that code internally and
its docs say including it may interfere with recovery. `recover_on_exit_codes` is for codes
*your own program* returns deliberately:

```yaml
  job_recovery:
    max_restarts_on_errors: 3
    recover_on_exit_codes: [33, 34]   # codes YOUR code exits with. Never 137.
```

Those restarts don't count against `max_restarts_on_errors`. In a multi-node job, recovery
triggers if **any** node exits with a listed code.

3 is a reasonable default: something that fails identically three times is a real bug, and
a job that retries forever burns GPUs silently.

### Separating work between teams or research bets

If different projects shouldn't see each other's jobs, checkpoints or data, **SkyPilot
workspaces already do this** — no infra work needed.

- The dashboard has a **Workspaces** page with a *Create New Workspace* button.
- Clusters and jobs are tagged with their workspace and views filter by it.
- **Private workspaces** restrict access to named users, and can pin the workspace to its
  own queue so quota is separated too:

  ```yaml
  my-bet:
    private: true
    allowed_users: [alice@pantheon.inc, bob@pantheon.inc]
    kubernetes:
      quota:
        queue: my-bet-queue
  ```

- **Routing is automatic** if you have access to exactly one workspace — no
  `sky workspace use` needed, and no per-command flag to remember.
- Each workspace can run under its **own Kubernetes service account**, which is how access
  to per-team storage gets scoped rather than left to convention.
- Volumes are already per-workspace (the `WORKSPACE` column in `sky volumes ls`), so a
  private workspace **plus a per-bet volume** is what actually separates data. Today
  everything shares one `/checkpoints` volume at a common path, so anyone can read anyone's
  checkpoints.

⚠️ **The `quota.queue` block only works inside a workspace definition.** The same block in
a *task* YAML does nothing — measured three times on 2026-08-11, every job landed in
`default-queue` with no error and no warning, running against the wrong quota. Both the
SkyPilot user guide and the installation guide's own verify step present the task-YAML form
as standalone, so this is worth knowing before you trust it:

```yaml
# ❌ DOES NOTHING — silently ignored
config:
  kubernetes:
    quota:
      queue: my-bet-queue
```

`sky workspace info` shows where your next request lands. Ask in #training-infra before
creating one so quota and naming stay coherent.

### Prebuilt images — skip the reinstall

**How much this is worth depends entirely on the shape of your job, and the two measured
cases are two orders of magnitude apart.**

**Measured 2026-08-10, an 8-GPU job doing a from-scratch torch and CUDA-stack install:**
**~10 minutes per launch holding all 8 GPUs at 77 W** — idle. Two pip passes, ~90 packages,
~1.3 GPU-hours per launch, roughly **66 GPU-hours across a 50-job sweep**. That is the case
a prebuilt image is for.

**Measured 2026-08-14, a 63-job sweep of 1-GPU jobs installing nine small wheels:** 92
seconds from container start to `[RESUME]`, ~80 s of it pip, with the image pull 4 s and
warm. Total across the whole sweep: **1.4 GPU-hours.** An image saves almost nothing here.

Both factors differ — GPUs held, and what `setup:` installs — which is why the two diverge
so far. Check which one you are before asking for an image.

Baking the same **pinned** versions into an image removes it: `setup:` becomes a no-op
because pip finds everything satisfied, with no version drift to reason about. Verified on
an H200 — `torch 2.11.0+cu128`, CUDA available, matmul correct, 2m29s cold pull (8.73 GiB,
smaller than NGC's 10.17), and ~0 once pre-warmed.

Images in the private Pantheon registry need an `imagePullSecret`. Without it you get
`ErrImagePull: pull access denied` and the job sits **`STARTING` with no error anywhere** —
the pod is in `ImagePullBackOff` but `sky jobs queue` shows nothing and the controller log
is silent:

```yaml
resources:
  image_id: docker:registry.eu-iceland1-a.ccr.crusoecloudcompute.com/pantheon.dac16446/<image>:<tag>
config:
  kubernetes:
    pod_config:
      spec:
        imagePullSecrets:
          - name: crusoe-registry
```

Leave your `setup:` block as it is. Ask in #training-infra for an image built to your
stack — the recipe is just your own resolved `pip list`.

### zstd-compressed images pull 2.7× faster

**Measured 2026-08-12**, same node, same image content, two fresh containerd namespaces,
sequential: **gzip 128 s → zstd 48 s.** Crusoe reproduced the same ratio independently on a
different image (44.6 s). It follows from the unpack cost in §8 — gzip is single-threaded by
construction, so the fix is a format that decompresses faster, not a faster network.

Worth doing for an image that is **pinned and rebuilt rarely**; a one-time re-encode pays
back within a couple of launches on a sweep. Not worth it for an image you rebuild daily.

```bash
# run this INSIDE the cluster, as a pod on quay.io/skopeo/stable — ~4 min
# registry-to-registry. From a laptop the whole image routes through your machine.
skopeo copy --dest-compress-format zstd --dest-compress-level 3 \
  --dest-force-compress-format \
  --override-os linux --override-arch amd64 \
  docker://<src-repo>/<image>:<tag> docker://<ccr-repo>/<image>:<tag>-zstd
```

`--override-os linux --override-arch amd64` is **required from an Apple Silicon Mac**, or
skopeo looks for a `darwin/arm64` layer and fails. Ask in #training-infra and we will
mirror it for you.

---

## 6. Templates

### Single-GPU experiment (start here)

```yaml
name: my-experiment

resources:
  accelerators: H200:1
  cpus: 20
  memory: 230

volumes:
  /checkpoints: checkpoints

envs:
  PANTHEON_USER: yourname@pantheon.inc
  EXPERIMENT_TAG: my-experiment-aug04
  WANDB_PROJECT: pfm-experiments
  WANDB_RESUME: allow
  WANDB_RUN_ID: my-experiment-aug04

secrets:
  - secrets:WANDB_API_KEY          # a reference, not a value -- see §5

setup: |
  pip install torch wandb --index-url https://download.pytorch.org/whl/cu128

run: |
  python train.py --lr 1e-4 --epochs 10
```

### Multi-GPU single-node

```yaml
name: my-ablation

resources:
  accelerators: H200:4
  cpus: 80
  memory: 920

volumes:
  /checkpoints: checkpoints

envs:
  PANTHEON_USER: yourname@pantheon.inc
  EXPERIMENT_TAG: ablation-lr1e4-bs64
  WANDB_PROJECT: pfm-ablations
  WANDB_RESUME: allow
  WANDB_RUN_ID: ablation-lr1e4-bs64

secrets:
  - secrets:WANDB_API_KEY          # a reference, not a value -- see §5

setup: |
  pip install torch torchvision wandb accelerate --index-url https://download.pytorch.org/whl/cu128

run: |
  torchrun --nproc_per_node=$SKYPILOT_NUM_GPUS_PER_NODE \
    train.py --lr 1e-4 --batch-size 64
```

### Multi-node distributed

⚠️ **A partial-node multi-node job puts every "node" on one machine, and you
cannot stop it.** Measured 2026-08-09 across three separate `2x[H200:1]` jobs:
all three placed *both* pods on a single machine (twice on node-4, once on
node-10). The third carried an explicit
`topologySpreadConstraints` block with `maxSkew: 1` and
`whenUnsatisfiable: DoNotSchedule` on `kubernetes.io/hostname` — **it had no
effect.**

**Why the spread constraint did nothing** (measured 2026-08-10): every admitted Kueue
Workload carries a `status.admission.podSetAssignments[].topologyAssignment` naming an
exact host, and the cluster's `Topology` object is
`{"levels":[{"nodeLabel":"kubernetes.io/hostname"}]}`. Kueue's Topology-Aware Scheduling
picks the specific machine **at admission time, at hostname granularity** — and with a
single level there is nothing for a spread constraint to spread *across*.

**Hard constraints are honoured, though** (measured 2026-08-14). This paragraph used to
end "nothing in a `pod_config` — spread constraints, affinity, anti-affinity — is ever
evaluated", and that was too broad: a `nodeSelector` on `kubernetes.io/hostname` placed a
job on the node it named. So a node label you need for another reason does work. What
still cannot be done is separating the pods of one job, because a `pod_config` applies to
every pod of that job alike — it can pin them all to one machine, not spread them over
several.

Nothing forces separation when GPUs-per-node is below 8 — at 8 the arithmetic
does it, below that nothing does.

**The job cannot detect this.** `socket.gethostname()` returns `node-0` and
`node-1` regardless of physical placement, and `container_ips` are pod IPs, so a
co-located pair looks identical to a distributed one from inside. Consequences:
no real network traffic, so a "works multi-node" test proves nothing; both ranks
share a failure domain, so fault-tolerance tests are meaningless; and any
bandwidth figure measured this way is loopback.

**The only reliable way to get genuine separation is `H200:8` per node**, where
two pods cannot fit on one machine. Affinity and spread rules do not work here.
Verify any multi-node run with `kubectl get pods -o wide | grep <jobname>` and
check the NODE column actually differs before trusting a cross-node result.

Multi-node **requires** an InfiniBand `pod_config` — without it, pods get one
shared NIC with GPUDirect disabled and DDP runs ~40× slower (~11 GB/s per rank,
~2.3% MFU). With it, measured all-reduce busbw through a SkyPilot job is
**437.5 GB/s**, validated **2026-08-09** on a converted 2-node × 8 H200 job
(1.07 GB all_reduce, all 8 HCAs, IB not sockets, GDRDMA on all 16 channels). That is the
date `modal-skypilot` stamps into every multi-node header, and it is the number to quote.

(An older 479 GB/s figure from job 446 on 2026-08-06 is still true and is **not** a
ceiling this regressed from: different message size, a different pod builder, a busier
cluster. Neither number is comparable to the other, which is why only the dated one the
shim emits belongs in a template.)

**Physical placement does not appear to matter at this cluster size.** Measured
2026-08-11: the same 2-node × 8×H200 all-reduce pinned inside one `crusoe.ai/pod.id` group
gave **452.4 GB/s**, against **437.5 GB/s** across two groups — 3.4% apart, inside
run-to-run variance on a shared fabric. So at 16 nodes the IB fabric looks effectively
non-blocking and there is nothing to gain from asking the scheduler for locality. Crusoe
also confirmed 2026-08-11 that rack and switch placement is not exposed at the node level,
so there is nothing reliable to express even if it did matter.

> ### Do not hand-copy the IB block
>
> **`modal-skypilot` emits it for you.** Run your conversion through the shim
> and it produces a correct, dated block plus a node-side assertion that fails
> loudly at second zero if IB is missing — instead of silently training at 2.3%
> MFU for six hours.
>
> If you are hand-writing SkyPilot YAML rather than converting Modal code, copy
> the block from **the shim's output**, not from this document. There is one
> source of truth and this is not it. Every time someone has copied a block from
> a doc it has drifted: one fork added `priorityClassName: system-cluster-critical`
> (a *Kubernetes* pod priority class, unrelated to `--priority`, which can evict
> other people's work), added a `k8s.v1.cni.cncf.io/networks` annotation (inert
> here — Multus runs but there are zero NetworkAttachmentDefinitions), and the
> job never ran.
>
> ⚠️ **Never add a `dshm` volume or a `/dev/shm` mount.** SkyPilot already
> provides `/dev/shm` as a memory-backed emptyDir sized to node RAM (measured
> 1.9 TiB). Adding your own is a duplicate volume name *and* a duplicate
> mountPath — the API server rejects the pod with a `422` and the job sits in
> PENDING with nothing in `sky jobs queue` to say why. It is also ~30× smaller
> than what you already had.

The rest of a multi-node task looks like this. The `config:` block is omitted
deliberately — get it from the shim:

```yaml
name: distributed-run

resources:
  accelerators: H200:8
  cpus: 160
  memory: 1840

num_nodes: 2

# config.kubernetes.pod_config: emitted by modal-skypilot — see the box above.

volumes:
  /checkpoints: checkpoints

envs:
  PANTHEON_USER: yourname@pantheon.inc
  EXPERIMENT_TAG: distributed-aug04
  WANDB_PROJECT: pfm-distributed
  WANDB_RESUME: allow
  WANDB_RUN_ID: distributed-aug04
  NCCL_DEBUG: INFO
  NCCL_TOPO_FILE: /opt/nccl_topo/h200-141gb-sxm-ib-cloud-hypervisor.xml
  UCX_RNDV_SCHEME: get_zcopy
  UCX_TLS: self,sm,cuda_copy
  NCCL_IB_PCI_RELAXED_ORDERING: "1"
  NCCL_IB_SPLIT_DATA_ON_QPS: "0"
  NCCL_IB_QPS_PER_CONNECTION: "2"
  NCCL_IB_MERGE_VFS: "0"
  NCCL_IB_HCA: "^mlx5_0:1"
  NCCL_NVLS_ENABLE: "1"
  NCCL_IB_SL: "1"
  NCCL_IBEXT_DISABLE: "1"

secrets:
  - secrets:WANDB_API_KEY          # a reference, not a value -- see §5

setup: |
  pip install torch wandb --index-url https://download.pytorch.org/whl/cu128

run: |
  MASTER_ADDR=$(echo "$SKYPILOT_NODE_IPS" | head -n1)
  torchrun \
    --nproc_per_node=$SKYPILOT_NUM_GPUS_PER_NODE \
    --nnodes=$SKYPILOT_NUM_NODES \
    --node_rank=$SKYPILOT_NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=12345 \
    train.py --lr 1e-4
```

Note: in distributed, only rank 0 should save checkpoints. All ranks load + `dist.barrier()`.

### Parameter sweep

```bash
for lr in 1e-3 3e-4 1e-4 3e-5 1e-5; do
  sky jobs launch sweep.yaml --env LR=$lr --env EXPERIMENT_TAG=sweep-lr-$lr --env WANDB_RUN_ID=sweep-lr-$lr -y
done
```

```yaml
name: sweep

resources:
  accelerators: H200:1
  cpus: 20
  memory: 230

volumes:
  /checkpoints: checkpoints

envs:
  PANTHEON_USER: yourname@pantheon.inc
  EXPERIMENT_TAG: sweep-lr-1e-4
  LR: "1e-4"
  WANDB_PROJECT: pfm-sweeps
  WANDB_RESUME: allow
  WANDB_RUN_ID: sweep-lr-1e-4

secrets:
  - secrets:WANDB_API_KEY          # a reference, not a value -- see §5

setup: |
  pip install torch wandb --index-url https://download.pytorch.org/whl/cu128

run: |
  python train.py --lr $LR
```

---

## 6.5 My job died — what happened?

Work down this list; each step is cheap and rules out a whole class.

| What you see | Likely cause | What to do |
|---|---|---|
| `FAILED_PRECHECKS` in ~2s, no pod ever created | Resource shape the cluster can't satisfy | `sky jobs queue -v` → DETAILS. Common: `disk_size` above ~123 (don't set it at all), cpus above ~175, more than 8 GPUs on one node |
| Admitted by the queue, never bound to a node, no error, forever | `disk_size` in the **120–123 band** — passes the precheck, fails the scheduler (§5) | Cancel and resubmit with the `disk_size` line deleted. Nothing will ever schedule it |
| `OOMKilled`, or exit 137, at a `memory:` that used to work | **Changed 2026-08-12** — `memory:` is a hard limit now (§4) | Your *anonymous* memory exceeded it. Confirm with a nonzero `oom`/`max` in `memory.events`, then size to anon plus headroom. If both are clean, report it — that is a different cause |
| Job runs but against the wrong queue's quota | You can see more than one workspace, so you landed in `default` | `sky workspace info`. See §5.9 |
| Job ran, then `FAILED` with a nonzero exit | Your script | `sky jobs logs <id>` — the error is usually in the last 20 lines |
| Job `PENDING` for hours, nothing on the cluster | Waiting for capacity, or provisioning backoff | `sky jobs queue -v` → DETAILS says `Job is in backoff` if it's retrying. Normal when the cluster is full |
| `RECOVERING`, then resumes | Preempted by a higher-priority job | Nothing to do — this is expected at p3. Make sure you checkpoint |
| Died mid-run, no error in your logs | Preemption, or a node/GPU event | Check `sky jobs queue -v` for recoveries; ping #training-infra if several jobs died together |
| `CUDA error: an illegal memory address` / XID 31 in cluster logs | **Your code** — a null or out-of-bounds pointer in a kernel | `CUDA_LAUNCH_BLOCKING=1` plus `compute-sanitizer` on a single-GPU repro. Harmless to the hardware, but results after the fault are suspect. **Measured 2026-08-12: 13,156 of these fleet-wide over three days, on 9 of 16 nodes — 94% of every GPU error event on the cluster, thirty times more than everything else combined.** If your job logs one, chase it; this is the most common fault here by a wide margin |
| Multi-node job trains but very slowly | Missing the IB pod_config | Your NCCL log will show one HCA and `GDR Disabled`. Get the block from `modal-skypilot convert` and paste it in — §6's template deliberately does not contain one, and neither does any other document |

**If several people's jobs die around the same time, it's probably not you.**
GPUs can be pulled out of the scheduler by node-level health events without any
notification. Say so in #training-infra rather than debugging alone — that's an
infra check, not a code one.

---

## 6.7 Telemetry contract (`training-v1`) — required

The SkyPilot dashboard expects telemetry from every running GPU training job and
**complains loudly if it's missing**. Contract owner: Sambhav.

Every active GPU training job must:

- Publish a **W&B run URL** discoverable by SkyPilot
- Refresh these W&B **summary** fields every ~10s, **from global rank 0 only**:

```python
wandb.run.summary.update({
    "pantheon_telemetry/contract_version": "training-v1",
    "pantheon_telemetry/heartbeat_at":     time.time(),      # unix ts
    "pantheon_telemetry/total_steps":      total_steps,      # positive int
    "pantheon_telemetry/active_step":      step,
    "pantheon_telemetry/step_time_s":      last_step_seconds,
    "pantheon_telemetry/effective_mfu":    mfu,              # 0..1
    "pantheon_telemetry/wandb_url":        wandb.run.url,
})
```

- Log scalar history under the first metric matching `train/loss*`; log `val/loss`
  when validation begins.

**Heartbeats older than 30s are violations.** Missing links, missing fields,
missing train loss, invalid values, or stale heartbeats all produce dashboard
errors.

**`effective_mfu` must be end-to-end**, not compute-only: summed step times in
the denominator must nearly equal the wall-clock length of the run. A
compute-only MFU hides dataloader stalls, which is exactly what this is meant to
surface. You may log a separate compute-only MFU for ceiling analysis — but not
under `pantheon_telemetry/`.

Non-training jobs opt out explicitly, with a reason:

```yaml
resources:
  labels:
    telemetry-contract: opt-out
    telemetry-opt-out-reason: embedding-extraction
```

### Related job requirements

- **Assume every job is preemptible.** Lose no more than **5 minutes** of
  compute to a preemption — and note there is *no* SIGTERM warning (§6.6), so
  that budget has to come from step-interval checkpointing, not a shutdown hook.
  Dataloader, optimizer and model state all need checkpointing.
- **Survive hardware failure** without corrupting past results. Nothing outside
  non-ephemeral storage should be load-bearing.
- **CPU-only data processing belongs on CPU machines**, not on GPU nodes. See
  `docs/cpu_only_guide.md`.
- **Dataloaders must prefetch and never block the GPU.** If dataloading stalls
  training, resize the model or GPU count rather than accepting the stall.
- **Use producer–consumer queues** for anything that shouldn't serialise —
  checkpoint writes especially. Update append-only data *before* metadata so a
  preemption can't leave a ghost shard.
- **Multiprocessing, not threading**, for parallelism in Python.

---

## 7. Commands

| Action | Command |
|---|---|
| Submit | `sky jobs launch experiment.yaml` |
| With priority | `sky jobs launch experiment.yaml --priority p1` |
| With env override | `sky jobs launch experiment.yaml --env LR=1e-3` |
| View queue | `sky jobs queue` |
| Stream logs | `sky jobs logs <job_id>` |
| Cancel | `sky jobs cancel <job_id>` |
| Dashboard | https://pantheon.svc.skypilot.co |

### Job states

| State | Meaning |
|---|---|
| PENDING | Queued, waiting for GPUs |
| RUNNING | Executing |
| RECOVERING | Evicted, auto re-launching |
| SUCCEEDED | Exit code 0 |
| FAILED | Check `sky jobs logs <id> --tail 100` |

---

## 8. Important notes
- **CUDA:** host driver is 12.8. Use `--index-url https://download.pytorch.org/whl/cu128` for PyTorch — it matches the driver exactly. CUDA 13+ wheels install cleanly and then fail at runtime.
- **CPU/RAM:** always set `cpus`/`memory` per §4 — unset jobs get ~4 CPU / 16 GB and starve. Node allocatable is 175.8 CPU / ~1928 GiB, so 176 CPU is unschedulable. **A node is 241 GiB per GPU; exceeding that on a sub-node job strands your neighbours' GPUs (§4).** SkyPilot renders `memory` as decimal `G` and Kubernetes reads `G` as 10⁹ — `memory: 1200` is **1117 GiB**, not 1200 GiB, so the number you write is about 6.9% larger than the binary figure you probably meant. The `241 GiB per GPU` share above is binary; divide a `memory:` value by 1.074 before comparing it. **Changed 2026-08-12:** that value is now a **hard limit** as well as a request (§4). The rendering did not change with it — `memory: 1840` still renders `1840G`, measured across every post-change pod — so this arithmetic still holds.
- **NGC images pin their own packages.** `nvcr.io/nvidia/pytorch` ships `/etc/pip/constraint.txt` pinning `transformer-engine`; a `pip install` of any other version dies in ~24s with `ResolutionImpossible`. Set `PIP_CONSTRAINT=` (empty) in `setup:` before installing. Note also that TransformerEngine's build queries the GPU driver, so it cannot be compiled on a GPU-less job.
- **Scratch vs durable:** bulk/temp data → `/tmp` (node NVMe, ~1.5 TiB free, no request needed); checkpoints → `/checkpoints`. Never set `disk_size` (see §5) and never dump datasets on the checkpoint volume.
- **A slow job start is mostly unpack, not download.** Measured 2026-08-10 on a 5.14 GiB
  image: **23 s to download, 74 s to unpack**, with containerd's
  `max_concurrent_unpacks = 1`. So the dominant cost is one CPU decompressing layers in
  series, and **a smaller image helps proportionally**.
  The in-region pull-through cache does help, but modestly: measured 2026-08-11 on one
  node with three fresh containerd namespaces, **135 s direct from nvcr.io / 121 s via the
  cache / 117 s with the cache warm**. About 11% against direct, and warming it adds only
  another 3% — which follows from download being ~20% of the total. (An earlier
  measurement here reported no benefit at all; that comparison was run through pods on
  different nodes and was almost certainly noise.)
  What removes the cost entirely is not pulling: images pre-warmed onto every node report
  `already present on machine` and start in seconds. Ask in #training-infra if your image
  should be added.
- **Days 1–3:** 1-GPU and 1-node only. No multi-day runs until queue is proven.
- **Eviction is normal** for p3 jobs. Checkpoint early and often, rotate old checkpoints.
- **Don't use p0** without Mo/Sambhav approval.

---

## 9. For agents: mistakes to avoid

| Mistake | Fix |
|---|---|
| `sky launch` instead of `sky jobs launch` | Always `sky jobs launch` for unattended work |
| Accelerators H100/A100 | `H200:N` only |
| Missing `cpus` / `memory` | Set `20 × N` / `230 × N` GB per GPU |
| `memory` above `230 × N` on a sub-node job | Reduce it. A node is 241 GiB per GPU; the excess strands the GPUs you did not request and no other job can use them. Size to *anonymous* memory (`/sys/fs/cgroup/memory.stat`), not to `memory.current`, which counts reclaimable page cache. If `memory.events` shows `max 0`, the request has never been approached |
| Missing `volumes: /checkpoints: checkpoints` | Always mount for checkpoint survival |
| Missing PANTHEON_USER / EXPERIMENT_TAG | Required for checkpoint isolation |
| Same EXPERIMENT_TAG across different experiments | Checkpoints collide — unique per experiment |
| Datasets/temp files written to `/checkpoints` | Write to `/tmp` (node NVMe); the checkpoint volume is shared and capped |
| Setting `disk_size` at all | Remove it. Above ~123 fails precheck in 2s; **120–123 passes every check and then never schedules, forever**; anything that does run blocks the node's other GPUs. You already get ~1.5 TiB at `/tmp` |
| PyTorch with cu130 | Use `cu128` — host driver is 12.8 |
| `infra:` set in YAML | Remove — SkyPilot handles placement |
| Priority above p3 without approval | Default p3 |
| Writing output to `/workspace`, `~`, or a relative path | Write to `/checkpoints/...` or `/mnt/raid0` — the container filesystem is shared node disk and fills it for everyone |
| `cpus`:`memory` far from 1:11 | Keep the `20 × N` / `230 × N` ratio; skewed requests strand the other resource cluster-wide |
| Long backfill or opportunistic work at p3 | Use `p4` (10) — below default, yields to everything else |
| Bare numeric priority (`--priority 2`) | Tier names only — bare numbers land *below* p3 and get evicted first |
| Large inline payload in `setup:` (e.g. a base64 tarball) | Use `file_mounts:` — a long command line fails with `exec /bin/bash: argument list too long` before anything runs, and the job then retries forever |
| Relaunching after editing `config: kubernetes: pod_config` | `sky down <cluster>` first, or SkyPilot silently reuses the old pod config |
| Multi-node without an IB `pod_config` | Get it from `modal-skypilot` — without IB you get one NIC and ~40× slower collectives. Do not copy a block out of a doc |
| Adding a `dshm` volume or `/dev/shm` mount to `pod_config` | Remove it. SkyPilot already provides `/dev/shm` (memory-backed, ~1.9 TiB). Yours is a duplicate volume name *and* mountPath — the API server returns `422`, no pod is created, and the job sits PENDING with no visible cause |
| Any `ephemeral-storage` request in `pod_config`, or `disk_size` at **any** value | Remove it entirely — do not lower it. A node has 111.5 GiB of ephemeral-storage allocatable **shared across all 8 GPUs** (~14 GiB per GPU). `disk_size: 100` is a 93.1 GiB request: it launches, then blocks the node's other 7 GPUs. Measured 2026-08-11: sixteen such 1-GPU jobs left **19 GPUs idle** with 26 workloads pending. A 1 Ti request is worse still — it passes prechecks and then `0 pods fit` on every node, forever. `/tmp` already gives ~1.5 TiB with no request |
| Treating `memory:` as a request with slack | It is a hard limit as of 2026-08-12 — request and limit are the same number. Anon memory over it `OOMKill`s the container. Page cache is still free |
| Padding `memory:` after an OOMKill "to be safe" | Measure instead: `anon` in `/sys/fs/cgroup/memory.stat` plus headroom. Padding strands your neighbours' GPUs (§4) and does not fix a real leak |
| A long-running job submitted before 2026-08-12 | It has no memory limit and never will — the spec is fixed at submission. Cancel and relaunch |
| `config.kubernetes.quota.queue` in a task YAML | Does nothing, silently. Queue routing belongs in the *workspace* definition (§5.9). Measured three times |
| Mirroring an image to zstd from your laptop | Run skopeo as a pod in-cluster (~4 min). From an Apple Silicon Mac also pass `--override-os linux --override-arch amd64` |
| `pip install` in an NGC image failing with `ResolutionImpossible` | NGC images ship `/etc/pip/constraint.txt` pinning `transformer-engine`. Set `PIP_CONSTRAINT=` (empty) before installing |
| Building TransformerEngine on a GPU-less job | Doesn't work — the build queries the driver and fails with `0 active drivers`. Allocate at least 1 GPU |

---

Questions? #training-infra or DM Mo.
