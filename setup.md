# Setup

## Clone and install

```bash
git clone https://github.com/amugoodbad229/orcaqd.git
cd orcaqd
uv venv --python 3.11
uv sync --extra cuda --extra dev
```

No GPU? Use `uv sync --extra dev` instead.

## Generate MJX model and verify

```bash
uv run python scripts/build_mjx_mjcf.py
uv run python scripts/smoke_test.py
uv run pytest -v
```

## View the hand

```bash
uv run python scripts/view.py              # MJX variant (collision primitives)
uv run python scripts/view.py --upstream   # original mesh model
```

## Benchmark

```bash
uv run python scripts/bench_throughput.py --batch 256 --steps 100
```

## Project structure

```
orcaqd/
├── orcahand/              # OrcaHand v2 model files (MJCF, URDF, STL meshes)
├── mjx/       # generated MJX-friendly MJCF (primitive collisions)
├── src/envs/              # MJX environment + behavior descriptors
├── src/qd_engine/         # QDax training (coming)
├── scripts/               # build, test, bench, view
├── tests/                 # pytest suite
├── configs/               # training configs (coming)
├── paper1.md              # Paper 1 outline
├── paper2.md              # Paper 2 outline
└── pyproject.toml         # dependencies (uv-managed)
```

## Scripts

| Script | What it does |
|---|---|
| `scripts/build_mjx_mjcf.py` | Generate MJX MJCF from the OrcaHand model |
| `scripts/smoke_test.py` | Verify JAX + MJX loads correctly |
| `scripts/bench_throughput.py` | Measure batched simulation throughput |
| `scripts/view.py` | Interactive MuJoCo viewer |
| `scripts/render_preview.py` | Render PNG preview (headless) |

## Modal (cloud GPU runs)

```bash
modal token new
modal secret create wandb WANDB_API_KEY=<your-key>
modal run -m src.modal_app::smoke           # ~$0.10 test
modal run --detach -m src.modal_app::train  # ~$10 full run
```

## Troubleshooting

| Problem | Fix |
|---|---|
| JAX sees CPU only | `uv sync --extra cuda --extra dev` |
| MJX fails with plane/mesh error | Run `scripts/build_mjx_mjcf.py` first |
| Viewer doesn't open | Set `DISPLAY=:0` or use `scripts/render_preview.py` |
| Tests fail with file not found | Run `scripts/build_mjx_mjcf.py` first |
