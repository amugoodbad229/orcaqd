# OrcaQD Setup Guide

Complete guide to get the project running from a fresh clone.

---

## Prerequisites

- **Linux x86_64** (Ubuntu 22.04+, or WSL2 on Windows 11)
- **NVIDIA GPU** with driver ≥ 525 (optional — CPU-only works for development, GPU needed for training)
- **Git** configured with your credentials
- **~10 GB** free disk space

### Install uv (Python package manager)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
exec $SHELL
uv --version   # should print 0.11+
```

### Install system dependencies

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

No NVIDIA GPU? Skip the cuda extra:

```bash
uv sync --extra dev
```

This installs JAX, MuJoCo, MJX, QDax, Flax, and all other dependencies from `pyproject.toml`.

---

## Generate the MJX model

The OrcaHand v2 upstream MJCF uses high-poly mesh collisions that MJX-JAX can't handle. We generate a primitive-collision variant (capsules + boxes) that MJX accepts:

```bash
uv run python scripts/build_mjx_mjcf.py
```

Output:
```
Reading orcahand/models/mjcf/orcahand_right.mjcf
Reading orcahand/models/mjcf/orcahand_right_body.xml
  demoted 30 mesh geoms to visual class
  inserted 12 primitive collision geoms
Wrote mjx/orcahand_right_mjx.mjcf
Wrote mjx/scene_right_mjx.xml
```

This only needs to be run once (or again if you edit `PRIMITIVE_SPECS` in the script).

---

## Verify everything works

### Smoke test

```bash
uv run python scripts/smoke_test.py
```

Expected output:
```
jax     : 0.10.0, backend=gpu
mujoco  : 3.8.1
devices : [CudaDevice(id=0)]

[expected-fail (correct)] orcahand/scene_right.xml
[OK] mjx/scene_right_mjx.xml
```

The "expected-fail" is intentional — the upstream MJCF fails on MJX (that's why we generate the primitive variant). The second line must say OK.

### Run tests

```bash
uv run pytest -v
```

All tests should pass (33 total: env lifecycle, descriptor math, MJX model properties, upstream MJCF parsing).

### Throughput benchmark (GPU only)

```bash
uv run python scripts/bench_throughput.py --batch 64 --steps 100
uv run python scripts/bench_throughput.py --batch 256 --steps 100
```

Reference numbers on RTX 3060 6GB:
- batch 64: ~13,000 steps/sec
- batch 256: ~52,000 steps/sec

On H100 with batch 4096: expected >200,000 steps/sec.

### Interactive viewer

```bash
uv run python scripts/view.py              # MJX variant (shows collision primitives)
uv run python scripts/view.py --upstream   # original OrcaHand v2 mesh model
uv run python scripts/view.py --combined   # both hands
```

Controls:
- Left-drag: rotate
- Right-drag: pan
- Scroll: zoom
- Press **2**: toggle visual mesh group
- Press **3**: toggle collision primitive group
- Esc: close

If the viewer doesn't open (headless server), use the headless renderer instead:

```bash
uv run python scripts/render_preview.py    # saves scripts/preview.png
```

---

## Project structure

```
orcaqd/
├── orcahand/                          # OrcaHand v2 model files
│   ├── models/assets/right/           #   STL meshes (right hand)
│   ├── models/assets/left/            #   STL meshes (left hand)
│   ├── models/mjcf/                   #   upstream MJCF + body XML
│   ├── models/urdf/                   #   URDF files
│   ├── scene_right.xml                #   scene files
│   ├── scene_left.xml
│   └── scene_combined.xml
│
├── mjx/                               # generated MJX-friendly MJCF
│   ├── orcahand_right_mjx.mjcf        #   primitive-collision model
│   └── scene_right_mjx.xml            #   scene wrapper with floor
│
├── src/                               # research code
│   ├── __init__.py
│   ├── jax_env.py                     #   XLA performance flags
│   └── envs/
│       ├── orcahand_mjx_env.py        #   MJX environment (reset, step, reward)
│       └── bd_extractors.py           #   behavior descriptors (b1, b2)
│
├── scripts/                           # utility scripts
│   ├── build_mjx_mjcf.py             #   generate MJX MJCF from upstream
│   ├── smoke_test.py                  #   verify JAX + MJX loads
│   ├── bench_throughput.py            #   batched throughput benchmark
│   ├── view.py                        #   interactive MuJoCo viewer
│   └── render_preview.py             #   headless PNG preview
│
├── tests/                             # pytest suite
│   ├── test_mjx_model.py             #   MJX model sanity (6 tests)
│   ├── test_bd_extractors.py         #   descriptor math (10 tests)
│   └── test_env.py                    #   env lifecycle (4 tests)
│
├── configs/                           # training configs (coming Week 3)
├── pyproject.toml                     # dependencies (uv-managed)
├── uv.lock                            # lockfile
├── setup.md                           # this file
├── README.md
├── paper1.md                          # Paper 1 research outline
└── paper2.md                          # Paper 2 research outline
```

---

## Scripts reference

| Script | What it does | When to run |
|---|---|---|
| `scripts/build_mjx_mjcf.py` | Reads upstream OrcaHand MJCF, demotes mesh geoms to visual-only, inserts 12 primitive collision shapes, writes `mjx/` files | Once after clone, or after editing `PRIMITIVE_SPECS` |
| `scripts/smoke_test.py` | Checks JAX device, loads both upstream (expected fail) and MJX (must pass) models | After install, after any MJCF changes |
| `scripts/bench_throughput.py` | Batched MJX rollout benchmark. Flags: `--batch`, `--steps`, `--warmup` | To measure GPU performance |
| `scripts/view.py` | Interactive MuJoCo viewer. Flags: `--upstream`, `--left`, `--combined`, `--mjcf <path>` | Visual inspection |
| `scripts/render_preview.py` | Renders 3-panel PNG (visual / collision / both) to `scripts/preview.png` | Headless servers, or quick check |

---

## Adding dependencies

Never `pip install` directly. Always go through uv:

```bash
uv add <package>                           # runtime dependency
uv add --optional dev <package>            # dev-only dependency
uv sync --extra cuda --extra dev           # re-sync after adding
```

Commit both `pyproject.toml` and `uv.lock` together.

---

## Modal (cloud GPU runs)

For production training on H100:

```bash
# One-time setup
modal token new                                    # authenticate
modal secret create wandb WANDB_API_KEY=<key>      # for logging

# Smoke test (~$0.10)
modal run -m src.modal_app::smoke

# Full training run (~$10, 3 hours, runs detached)
modal run --detach -m src.modal_app::train --config configs/paper1_main.yaml

# Pull results
modal volume ls orcaqd-artifacts /runs
modal volume get orcaqd-artifacts /runs/<TASK_ID>/archive.tar.zst ./
```

---

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `jax.devices()` shows `[CpuDevice]` | Installed without cuda extra | `uv sync --extra cuda --extra dev` |
| `mjx.put_model()` fails with `plane/mesh margin` error | Loading upstream MJCF directly | Run `scripts/build_mjx_mjcf.py` and use `mjx/scene_right_mjx.xml` |
| Tests fail with "file not found" | MJX MJCF not generated yet | Run `scripts/build_mjx_mjcf.py` |
| Viewer window doesn't appear | No display (headless or WSLg issue) | Set `DISPLAY=:0` or use `scripts/render_preview.py` |
| `uv sync` can't resolve `jax[cuda13]` | Index strategy not set | Confirm `[tool.uv].index-strategy = "unsafe-best-match"` in `pyproject.toml` |
| Slow `uv sync` (>10 min) | Project on Windows mount (`/mnt/c/...`) | Copy to native Linux filesystem (`~/projects/`) |

---

## What's next

| Week | Deliverable | Status |
|---|---|---|
| 1 | MJX-compatible MJCF, smoke test, benchmark | ✅ Done |
| 2 | Env class, behavior descriptors, tests | ✅ Done |
| 3 | Add graspable object, wire reward/descriptors, QDax training loop | Next |
| 4–5 | Full PGA-MAP-Elites run, archive visualization | Planned |
| 6 | Ablations (vanilla ME, DCRL-ME, descriptor swap) | Planned |
| 7 | Paper 2: cluster archive, VLM router | Planned |
| 8 | Paper drafts, figures, submission | Planned |
