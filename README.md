# pantheon — bowl experiment

Does V-JEPA 2's final-state embedding distinguish *where things ended up*
from *how they got there*?

Three MuJoCo scenes, identical ball/bowl/|initial speed|, mirror-symmetric
about the camera:

- **A** — ball rolls into the bowl over its right rim, settles at the bottom
- **B** — mirror of A (enters over the left rim), settles at the same spot
- **C** — differs by scenario:
  - **Scenario 1**: starts ON the right rim with outward velocity (same
    speed, lower than s2), rolls a short way, stops on the flat near the bowl
  - **Scenario 2**: starts at the bowl bottom and escapes out over the right
    rim (shallower bowl + higher shared speed to make escape possible)
  - **Scenario 3**: crater bowl (raised rim) + high-friction apron around it;
    C starts on the rim crest, tips over the outer wall, and stops right next
    to the bowl. The apron is required: C's exit path is A's entry path, so C
    recovers any energy A/B need to get in — only dissipation outside the
    bowl can stop it without wrecking A/B's centering.
  - **Scenario 4**: scenario 3's A/B, but C simply RESTS at the bowl center
    for the whole clip — all three scenes end identically; only history
    differs.

High friction, no bounce; every scene ends with the ball at rest for >1 s,
and A/B end in the same position (verified to ~10 mm or better, sub-pixel).

The eval encodes the **whole clip** in one V-JEPA 2 forward pass (uniformly
subsampled to 64 frames), so trajectory history reaches every token through
bidirectional attention — then pools only the temporal token positions
falling in the **last second**. The invariant must hold despite the history:

```
d(A,B) < d(A,C)   and   d(A,B) < d(B,C)
```

reported in both cosine similarity and L1 distance.

## Layout

Each eval family is a generator module under `evals/`; they share one CLI
harness (`evals/common/family.py`) so all behave identically:

```bash
python -m evals.<family> --out-root <dir>   # render all evals -> <dir>/<id>/{A,B,C}.mp4
python -m evals.<family> --check            # validate the contract on CPU, no render
python -m evals.<family> --out-root <dir> --only 3   # a subset of eval ids
```

- `evals/bowl.py` — scenarios 1–4 → `scenario_1..4/` (heightfield bowl —
  MuJoCo convexifies meshes, so a concave mesh bowl wouldn't collide).
  Contract: A/B settle at the same spot, everything at rest >1 s.
- `evals/cube.py` — one eval `arc/`: kinematic (mocap, collision-free) cube
  on semicircular arcs. A: counterclockwise X→Y; B: clockwise X→Y (same
  endpoints, different path); C: A's arc translated (different start/end).
- `evals/basic_counting.py` — evals 1–10 → `eval_01..10/`: A has X prisms,
  B has X prisms elsewhere, C has X+1 prisms at fresh positions (`--c-mode
  union` restores the old 2X superset variant). Static scenes.
- `evals/encoders/` — the model as a **black box**: `Encoder.encode(video)
  -> one 1-D vector` for the clip's last second. Swap models by implementing
  `base.Encoder` and adding a line to `encoders/__init__.py`. Registered:
  - `vjepa2_raw` / `vjepa2_mean` — V-JEPA 2, runs in the default env.
  - `fastwam_raw` / `fastwam_mean` — FastWAM ActionDiT; hooks the middle+1
    action block (`vjepa2`-incompatible env; needs the stereo rig for its
    2-camera 224x448 input).
  - `qwen3vl_raw` / `qwen3vl_mean` — Qwen3-VL-30B-A3B (native video VLM);
    middle+1 decoder layer, pooled over the video tokens' last-second span.
  - `cosmos_raw` / `cosmos_mean` — Cosmos-Predict 2.5 diffusion DiT; hooks
    the middle+1 transformer block on one forward (gated weights, HF_TOKEN).
  - `saved:<tag>` — reads precomputed vectors (see below).

  **Multi-environment workflow.** Only V-JEPA shares the default env; FastWAM,
  Qwen3-VL, and Cosmos each need incompatible deps, so they run in their own
  venvs (`scripts/setup_{fastwam,qwen,cosmos}.sh`). In that venv you run
  `python -m evals.precompute --root <videos> --encoder <name> --tag <tag>`,
  which writes `<A|B|C>__<tag>.npy` next to each clip; then `measure`
  (default env) consumes them with `--encoders saved:<tag> ...` and puts every
  model in the same PASS/FAIL table. V-JEPA can be run either way.
- `evals/measure.py` — model-agnostic runner (`--videos <dir>` or `--root
  <dir>`, `--encoders ...`). It only ever sees vectors, so it works with any
  registered encoder. For each encoder it tests the invariant in cosine and
  L1 and prints a PASS/FAIL matrix. Never gates via exit code: the goal is
  mapping where the invariant fails, not making it pass.
- `evals/sweep_counting.py` — the canonical way to run counting: renders +
  measures counts 1–10 across many seeds (encoder loaded once) and aggregates
  per-count pass rates + mean margins. A single seed is too noisy for a
  cardinality signal, so `task.yaml` always runs counting as this sweep;
  `basic_counting.py` on its own is for spot-checking / the scene builder.
- `evals/common/` — `family.py` (shared CLI/loop), `objects.py` (true-bowl
  panels, ramps, prism meshes), `xml_scene.py` (one camera/lighting
  convention), `cameras.py` (multi-camera rigs), `observations.py` (writes
  the primary camera as `<name>.mp4` plus extra cameras / per-frame state),
  `sim.py` (rollout loops, ball tracker, `render_cams`), `video.py`,
  `rotations.py`.

### Multi-modal observations (`--rig`)

Every generator takes `--rig {mono,stereo,tri}` (default `mono`). The scene
is the source of truth; the rig chooses which observations to render from it:

- `mono` (default) — one camera → `<name>.mp4`. Back-compatible; V-JEPA and
  `measure` are unchanged.
- `stereo` / `tri` — the primary camera plus extra views → `<name>_<cam>.mp4`,
  and per-frame state → `<name>_state.npz`. This is what lets richer models
  (e.g. a robot world-action model expecting multi-camera + proprioception)
  receive the inputs they need without changing the scene definitions.

Currently wired for the cube family; bowl and counting follow once the exact
downstream rig (camera count / resolution / state layout) is fixed.
- `task.yaml` — SkyPilot task: 1× H200, mounts `kenny-data` at `/data`,
  renders all families, measures everything.
- `volume.yaml` — the `kenny-data` persistent volume (1 TiB,
  `crusoe-sharedfs`, RWX).
- `scripts/scene_server.py` + `scripts/builder.html` — interactive 3D
  A/B/C scene builder (FastAPI + three.js).
  JSON scene specs -> MuJoCo render -> in-browser video + one-click eval.
  Supports TRUE bowls standing on the ground (convex box-panel assembly —
  raised walls, wide flat lip, parabolic interior), ramps (sunk low edges so
  balls roll on smoothly), dynamic balls, and kinematic curve-driven cubes.
  Run on the box: `MUJOCO_GL=egl HF_HOME=/data/hf python scripts/scene_server.py`
  then open http://localhost:8020 (VS Code Remote-SSH forwards the port, or
  `ssh -L 8020:localhost:8020 kenny-dev`). Saved scenarios: /data/scenes/.
  NOTE: a background server does not reset the autostop idle timer — keep an
  SSH session open or raise `sky autostop kenny-dev -i 240 --down`.
- `data` → symlink to `/data` on the cluster (dangles locally; lets one
  VS Code Remote-SSH window browse code + volume together).

## Workflow

```bash
# tune physics locally (CPU, seconds per iteration)
python scripts/bowl_scenes.py --check

# run the full experiment on the cluster (starts the box if it's down)
sky launch -c kenny-dev task.yaml -i 60 --down -y

# iterate without re-running setup (box must be up)
sky exec kenny-dev task.yaml

ssh kenny-dev          # interactive; VS Code Remote-SSH -> kenny-dev
sky down kenny-dev     # done for the day (auto-downs after 60 min idle anyway)
```

Outputs on the volume: `/data/videos/bowl/{A,B,C}.mp4`,
`/data/embeddings/bowl_last1s.pt`. HF model cache: `/data/hf`.

## Physics tuning notes

- Tuned values are the script defaults: `--mu-roll 0.001 --damping 0.0005
  --duration 9.5 --speed 0.8 --start-offset 0.9 --bowl-r 0.4 --bowl-d 0.10`.
  Verified via `--check`: A/B final positions 0.4 mm apart, ≥2 s at rest;
  C never re-enters the bowl and rests ≥6 s.
- Free-joint `damping` acts on angular DOFs too; a small ball's rotational
  inertia is tiny, so values much above ~1e-3 freeze the ball on the spot.
- Concave collision: MuJoCo convexifies meshes (verified empirically — a
  ball dropped into a concave bowl mesh rests on the hull's "lid"), so the
  bowl must be a heightfield, which collides exactly and smoothly. Composite
  ramps/boxes are only needed for a free-standing bowl with above-ground
  walls.
- C starting on the rim (rather than inside) means it needs no escape
  energy, which is what allows the deep (10 cm) bowl and the low shared
  speed.

## Cluster rules that shaped this config (see researcher_experiment_guide.md)

- **PyTorch must be cu128** (`--index-url .../whl/cu128`) — host driver is
  CUDA 12.8; newer wheels install fine and then silently fall back to CPU.
- **Always set `cpus: 20` / `memory: 230` per GPU** — unspecified jobs get
  4 CPU / 16 GB. Never go above the table (it strands other people's GPUs);
  `memory:` is a hard limit (OOMKill above it).
- **Never set `disk_size` or any ephemeral-storage request** — `/tmp` already
  has ~1.5 TiB of node NVMe with no request needed.
- **Never write to `~`, `/workspace`, or relative paths** on the cluster —
  that's shared node disk. Durable output → `/data`; bulk scratch → `/tmp`;
  huge scratch / dataset caches → `/mnt/raid0` (guide §5).
- **Unattended work goes through `sky jobs launch`, never `sky launch`** —
  preemptible (p3 default, tier names only), no SIGTERM, ≤5 min compute loss:
  checkpoint on a step interval, write atomically, set
  `job_recovery.max_restarts_on_errors: 3` (never list 137).
- **Non-training GPU jobs opt out of the telemetry contract** via
  `resources.labels` (already in task.yaml).
- Bulk data in/out of the cluster: `s3://encodings` bucket.
- Questions → #training-infra.
