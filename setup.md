# OrcaQD Setup

Complete guide to get the project running, from a fresh clone to a Modal cloud run.

---

## Prerequisites

- **Linux x86_64** (Ubuntu 22.04+, or WSL2 on Windows 11)
- **NVIDIA GPU** with driver ≥ 525 — optional for development, required for training
- **Git** configured with your credentials
- **~10 GB** free disk space

### Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
exec $SHELL
uv --version    # 0.11+
```

### Install system dependencies (Linux)

```bash
sudo apt update
sudo apt install -y build-essential git curl wget libgl1 libosmesa6 ffmpeg
```

---

## Clone and install

```bash
git clone https://github.com/amugoodbad229/orcaqd.git
cd orcaqd
uv venv --python 3.11
uv sync --extra cuda --extra dev
```

**No GPU?** Use `uv sync --extra dev` instead.

---

## Generate the MJX model

The upstream OrcaHand MJCF uses high-poly mesh collisions that MJX-JAX can't handle. We generate a primitive-collision variant (capsules + box) that MJX accepts:

```bash
uv run python scripts/build_mjcf.py
```

Expected output:
```
Reading orcahand/models/mjcf/orcahand_right.mjcf
Reading orcahand/models/mjcf/orcahand_right_body.xml
  demoted 30 mesh geoms to visual class
  inserted 12 primitive collision geoms
Wrote mjx/orcahand_right_mjx.mjcf
Wrote mjx/scene_right_mjx.xml
```

Run this once after cloning. Re-run only if `PRIMITIVE_SPECS` in the script changes.

---

## Verify everything works

### Environment check

```bash
uv run python scripts/check_env.py
```

Expected:
```
jax: 0.10.0, backend=gpu
mujoco: 3.8.1
[expected-fail (correct)] orcahand/scene_right.xml
[OK] mjx/scene_right_mjx.xml
```

The "expected-fail" line is intentional — the upstream MJCF fails on MJX (which is exactly why we generate the primitive variant). The second line must say OK.

### Pytest suite

```bash
uv run pytest -v
```

All 20 tests should pass: descriptor math (10), env lifecycle (4), MJX model properties (6).

### Throughput benchmark (GPU only)

```bash
uv run python scripts/bench.py --batch 64 --steps 100
uv run python scripts/bench.py --batch 256 --steps 100
```

Reference (RTX 3060 6GB):
- batch 64: ~13,000 steps/sec
- batch 256: ~52,000 steps/sec

Expected on A100 80GB at batch 4096: >150,000 steps/sec.

### Interactive viewer

```bash
uv run python scripts/view.py              # MJX variant
uv run python scripts/view.py --upstream   # original mesh model
```

Mouse: left-drag rotate, right-drag pan, scroll zoom. Press **2** to toggle visual mesh, **3** for collision primitives, Esc to close.

Headless? Use `scripts/preview.py` for a PNG preview instead.

---

## Run training (local)

Smoke run (~2 minutes on any GPU):

```bash
uv run python -m src.qd_engine.train --config configs/paper1_smoke.yaml
```

Expected: archive fills, QD-Score improves from ~-0.6 toward 0.

---

## Run training (Modal cloud)

Modal lets you run the training on rented A100/H100 GPUs without managing infrastructure.

### One-time setup

```bash
uv run modal token new                                    # OAuth, links to your workspace
uv run modal secret create wandb WANDB_API_KEY=<your-key> # for logging (optional)
```

### Cost-conscious workflow

Modal pricing (verified May 2026):

| GPU | Per hour |
|---|---|
| L4 | $0.80 |
| A100-40GB | $2.10 |
| **A100-80GB** | **$2.50** |
| H100 | $3.95 |

**With $30 credit, you have:** ~12 hours of A100-80GB time.

Run in this order to minimize cost:

```bash
# 1. Container build + GPU verification (~$0.02 on L4)
uv run modal run src/cloud.py::smoke

# 2. Throughput benchmark on cheap GPU (~$0.10 on L4)
uv run modal run src/cloud.py::bench

# 3. Short A100-80GB run (~$0.50, validates training scales)
uv run modal run src/cloud.py::train_short

# 4. Full headline run (~$8-10, 3-4 hours, runs detached)
uv run modal run --detach src/cloud.py::train
```

`--detach` lets you close your laptop. WandB tracks progress remotely.

### Pull artifacts back

```bash
uv run modal volume ls orcaqd-artifacts /runs
uv run modal volume get orcaqd-artifacts /runs/<TASK_ID>/archive_final.npz ./
```

---

## Project layout

```
orcaqd/
├── orcahand/                       # OrcaHand v2 model files (MJCF, URDF, STL meshes)
│   ├── models/{assets,mjcf,urdf}/
│   └── scene_{right,left,combined}.xml
├── mjx/                            # generated MJX-friendly MJCF
│   ├── orcahand_right_mjx.mjcf
│   └── scene_right_mjx.xml
├── src/
│   ├── jax_env.py                  # XLA performance flags
│   ├── cloud.py                # Modal entrypoint
│   ├── envs/
│   │   ├── dex_env.py              # hand-agnostic MJX env (DexHandEnv)
│   │   ├── hand_config.py          # HandConfig (OrcaHand, future Leap/Shadow/Allegro)
│   │   └── bd_extractors.py        # behavior descriptors (b1, b2)
│   └── qd_engine/
│       ├── train.py                # PGA-MAP-Elites training loop
│       └── pg_emitter.py           # PG emitter (TD3, available for future use)
├── scripts/
│   ├── build_mjcf.py               # generate MJX MJCF from upstream
│   ├── check_env.py                # verify JAX + MJX
│   ├── bench.py                    # throughput benchmark
│   ├── view.py                     # interactive MuJoCo viewer
│   └── preview.py                  # headless PNG preview
├── tests/                          # 20 tests
├── configs/
│   ├── paper1_smoke.yaml           # 1-2 min local test
│   ├── paper1_short.yaml           # 5-10 min A100 validation
│   └── paper1_main.yaml            # 3-4 hour headline run
├── pyproject.toml                  # dependencies (uv-managed)
├── uv.lock
├── setup.md                        # this file
├── README.md
├── paper1.md                       # Paper 1 outline (QD-RL)
└── paper2.md                       # Paper 2 outline (VLM orchestration)
```

---

## Scripts reference

| Script | Purpose | When to run |
|---|---|---|
| `scripts/build_mjcf.py` | Generate MJX MJCF from upstream OrcaHand model | Once after clone |
| `scripts/check_env.py` | Verify JAX device, load both upstream + MJX MJCF | After install / after MJCF changes |
| `scripts/bench.py` | Batched MJX rollout throughput. Flags: `--batch`, `--steps` | Measure GPU performance |
| `scripts/view.py` | Interactive MuJoCo viewer. Flags: `--upstream`, `--left`, `--combined` | Visual inspection |
| `scripts/preview.py` | 3-panel PNG (visual / collision / both) → `scripts/preview.png` | Headless servers |
| `src.qd_engine.train` | PGA-MAP-Elites training loop. Flag: `--config <yaml>` | Run training |

---

## Adding dependencies

Always go through uv:

```bash
uv add <package>                           # runtime dep
uv add --optional dev <package>            # dev-only dep
uv sync --extra cuda --extra dev           # re-sync
```

Commit `pyproject.toml` and `uv.lock` together.

---

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `jax.devices()` shows `[CpuDevice]` | Installed without cuda extra | `uv sync --extra cuda --extra dev` |
| `mjx.put_model()` fails with `plane/mesh margin` error | Loading upstream MJCF directly | Use `mjx/scene_right_mjx.xml` (run `scripts/build_mjcf.py` first) |
| Tests fail with "file not found" | MJX MJCF not generated yet | Run `scripts/build_mjcf.py` |
| Viewer doesn't open (WSL) | DISPLAY not set | `scripts/view.py` auto-sets it; for headless servers use `scripts/preview.py` |
| `uv sync` can't resolve `jax[cuda13]` | Index strategy not set | Confirm `[tool.uv].index-strategy = "unsafe-best-match"` in pyproject.toml |
| Modal container builds every run | `pyproject.toml` or `uv.lock` changed | Avoid editing those between runs; the layer caches when unchanged |

---

## What's done and what's next

| Week | Status |
|---|---|
| 1: MJX-compatible MJCF, smoke test, benchmark | ✅ Done |
| 2: Env class, descriptors, tests | ✅ Done |
| 3: QDax training loop (GA-only working, PG available) | ✅ Done |
| 4: Run on Modal A100-80GB, validate at scale | Next |
| 5: Visualization (archive heatmap, rollout videos) | Planned |
| 6: Baselines (PPO/SAC via rejax) | Planned |
| 7: Paper 2 (VLM router, cluster archive) | Planned |
| 8: Paper drafts, figures | Planned |

See `paper1.md` and `paper2.md` for full research outlines.
