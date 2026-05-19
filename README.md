# OrcaQD

Quality-Diversity RL for discovering diverse dexterous grasping skills on the OrcaHand v2.

## Quick start

```bash
git clone https://github.com/amugoodbad229/orcaqd.git
cd orcaqd
uv venv --python 3.11
uv sync --extra cuda --extra dev    # or --extra dev for CPU only
uv run python scripts/build_mjx_mjcf.py
uv run python scripts/smoke_test.py
uv run pytest -v
uv run python scripts/view.py       # interactive viewer
```

## Documentation

| Doc | Contents |
|---|---|
| [setup.md](setup.md) | Full environment setup guide |
| [paper1.md](paper1.md) | Paper 1: QD-RL for dexterous manipulation |
| [paper2.md](paper2.md) | Paper 2: VLM skill orchestration |

## Repository layout

```
orcaqd/
├── assets/mjcf/mjx/               # generated MJX-friendly MJCF
├── src/envs/                      # MJX env + behavior descriptors
├── src/qd_engine/                 # QDax training (Week 3)
├── src/agentic_layer/             # VLM routing (Week 7)
├── scripts/                       # build, test, bench, view
├── tests/                         # pytest suite
└── configs/                       # training YAML configs
```

## Scripts

| Script | Purpose |
|---|---|
| `scripts/build_mjx_mjcf.py` | Generate MJX-friendly MJCF from upstream model |
| `scripts/smoke_test.py` | Verify JAX + MJX loads correctly |
| `scripts/bench_throughput.py` | Batched MJX rollout benchmark |
| `scripts/view.py` | Interactive MuJoCo viewer |
| `scripts/render_preview.py` | Headless PNG preview |
