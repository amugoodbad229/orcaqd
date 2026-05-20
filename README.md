# OrcaQD

Quality-Diversity reinforcement learning for discovering diverse dexterous grasping skills on high-DOF anthropomorphic hands.

Standard deep RL on multi-fingered hands collapses to a single power-wrap grasp. We re-cast skill discovery as **archive optimization** using PGA-MAP-Elites over physically-meaningful behavior descriptors — contact dispersion and inter-digit force allocation. The result is a diverse archive of grasping policies covering multiple Cutkosky grasp types, discovered in hours on a single GPU with zero demonstrations.

Demonstrated on the OrcaHand v2 (17 DOF). Framework is hand-agnostic (16–24 DOF).

**[Setup guide](setup.md)** · **[Paper 1](paper1.md)** · **[Paper 2](paper2.md)** · **[WandB](https://wandb.ai/amugoodbad/orcaQD)**

---

## Quick start

```bash
git clone https://github.com/amugoodbad229/orcaqd.git
cd orcaqd

# Install (requires uv: curl -LsSf https://astral.sh/uv/install.sh | sh)
uv venv --python 3.11
uv sync --extra cuda --extra dev

# Build and verify
uv run python scripts/build_mjcf.py
uv run python scripts/check_env.py
uv run pytest -v
```

## Train

```bash
# Local (any GPU, ~2 min)
uv run python -m src.qd_engine.train --config configs/paper1_smoke.yaml

# Cloud A100-80GB (~$0.35, 5 min)
uv run modal run src/cloud.py::train_short

# Full run (~$8, 3-4 hours, detached)
uv run modal run --detach src/cloud.py::train
```

## Results

Verified on Modal A100-80GB (May 2026):

| Config | Iterations | QD-Score | Coverage | Time |
|---|---|---|---|---|
| `paper1_smoke` | 20 | -0.47 | 100% (10×10) | 2 min |
| `paper1_short` | 50 | -0.15 | 16% (25×25) | 5 min |
| `paper1_main` | 100K | TBD | TBD | ~3-4 hr |

Throughput: **100,000 steps/sec** at batch 256 on L4.

---

## Repository structure

```
orcaqd/
├── orcahand/           # OrcaHand v2 model (MJCF, URDF, STL meshes)
├── mjx/                # generated MJX-compatible MJCF (primitive collisions)
├── src/
│   ├── cloud.py        # cloud GPU runner (Modal)
│   ├── envs/           # DexHandEnv, HandConfig, behavior descriptors
│   └── qd_engine/      # MAP-Elites training loop, PG emitter
├── scripts/            # build_mjcf, check_env, bench, view, preview
├── tests/              # 20 tests
├── configs/            # training configs (smoke / short / main)
├── paper1.md           # Paper 1: QD-RL for dexterous manipulation
├── paper2.md           # Paper 2: VLM skill orchestration over QD archives
└── setup.md            # full setup and usage guide
```

## Citation

If you use this work, please cite:

```bibtex
@misc{orcaqd2026,
  title={OrcaQD: Discovering Dexterous Contact Manifolds via Hardware-Accelerated Quality-Diversity RL},
  author={Khan, Ayman},
  year={2026},
  url={https://github.com/amugoodbad229/orcaqd}
}
```

## License

MIT. OrcaHand v2 model files are MIT-licensed by the [OrcaHand team](https://orcahand.com).
