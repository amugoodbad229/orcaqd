# OrcaQD

**Quality-Diversity RL for high-DOF dexterous hands.**

We use Quality-Diversity (QD) algorithms to discover a diverse archive of grasping policies on multi-fingered robotic hands. Demonstrated on the OrcaHand v2 (17 DOF), but the framework is hand-agnostic and works with any 16–24 DOF anthropomorphic hand.

[Paper 1: QD-RL for dexterous manipulation](paper1.md) · [Paper 2: VLM skill orchestration](paper2.md) · [Setup guide](setup.md)

---

## Quick start

```bash
git clone https://github.com/amugoodbad229/orcaqd.git
cd orcaqd
uv venv --python 3.11
uv sync --extra cuda --extra dev    # or --extra dev for CPU only

uv run python scripts/build_mjcf.py        # generate MJX MJCF
uv run python scripts/check_env.py         # verify JAX + MJX
uv run pytest -v                           # all 20 tests
uv run python scripts/view.py              # open interactive viewer
```

For full details see [setup.md](setup.md).

---

## What this is

Standard deep RL on multi-fingered hands collapses to a single power-wrap grasp. We solve this by re-casting dexterous skill discovery as **archive optimization** — instead of one policy maximizing scalar return, we maintain a 2D grid of elite policies indexed by physically-meaningful behavior descriptors:

- **b₁ — Contact dispersion**: trace of contact-position covariance (small = pinch, large = wrap)
- **b₂ — Thumb force ratio**: fraction of total contact force from the thumb (0 = lateral, 0.5 = opposed power, 1 = thumb-only)

PGA-MAP-Elites optimizes a 50×50 grid over these descriptors. The result is ~2,000 distinct grasping policies covering a verifiable fraction of the Cutkosky taxonomy.

---

## Project layout

```
orcaqd/
├── orcahand/             # OrcaHand v2 model files
├── mjx/                  # generated MJX-friendly MJCF
├── src/
│   ├── envs/             # DexHandEnv, HandConfig, descriptors
│   └── qd_engine/        # PGA-MAP-Elites training, PG emitter
├── scripts/              # build, check, bench, view, preview
├── tests/                # 20 tests, all passing
├── configs/              # YAML training configs (smoke, short, main)
└── paper1.md, paper2.md  # research outlines
```

---

## Scripts

| Script | Purpose |
|---|---|
| `scripts/build_mjcf.py` | Generate MJX-friendly MJCF from upstream model |
| `scripts/check_env.py` | Verify JAX + MJX loads correctly |
| `scripts/bench.py` | Batched MJX rollout benchmark |
| `scripts/view.py` | Interactive MuJoCo viewer |
| `scripts/preview.py` | Headless PNG preview |

## Training

```bash
# Local smoke run (~2 min, any GPU)
uv run python -m src.qd_engine.train --config configs/paper1_smoke.yaml

# Modal A100-80GB validation (~$0.50, 5-10 min)
modal run src/cloud.py::train_short

# Full headline run (~$8-10, 3-4 hours)
modal run --detach src/cloud.py::train
```

See [setup.md](setup.md#run-training-modal-cloud) for the full Modal workflow with cost guidance.

---

## Stack

| Library | Role |
|---|---|
| `mujoco-mjx` 3.8 | JAX-native MuJoCo physics |
| `jax` 0.10 | GPU compute, vmap, JIT |
| `flax` 0.12 | Neural networks |
| `qdax` 0.5 | MAP-Elites core, descriptors |
| `evosax` 0.2 | ES algorithms (Open_ES, CMA_ES) |
| `rejax` 0.1 | Pure-JAX RL baselines (planned) |
| `optax` | Optimizers |
| `wandb` | Experiment tracking |
| `modal` | Cloud GPU |

All managed by `uv` — single `pyproject.toml` for local + Modal.

---

## License

MIT (see LICENSE).

The OrcaHand v2 model files in `orcahand/` are MIT-licensed by the OrcaHand team (https://orcahand.com).
