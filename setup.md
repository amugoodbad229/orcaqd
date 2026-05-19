# OrcaQD — Complete Setup Guide

> **What this is:** A step-by-step guide to create the `orcaqd` research repo from scratch, including the OrcaHand v2 model as a git submodule. Follow this on any Linux machine (bare-metal or WSL2) and you'll end up with a working, pushable GitHub repo.
>
> **Architecture:** Your research code lives in `orcaqd/`. The upstream OrcaHand description lives at `orcaqd/vendor/orcahand_description/` as a git submodule. You never modify the submodule — you only read model files from it.

---

## Prerequisites

- Linux x86_64 (Ubuntu 22.04+, or WSL2 on Windows 11)
- NVIDIA GPU + driver ≥ 525 (optional for CPU-only dev, required for training)
- Git configured with your GitHub credentials
- ~10 GB free disk

---

## Step 1: Create the repo on GitHub

Go to https://github.com/new and create a repo called `orcaqd` (or whatever you want). Private or public. Don't initialize with README (we'll push our own).

---

## Step 2: Set up the local repo

```bash
mkdir ~/projects/orcaqd && cd ~/projects/orcaqd
git init
```

---

## Step 3: Add OrcaHand description as a submodule

```bash
git submodule add https://github.com/orcahand/orcahand_description.git vendor/orcahand_description
# (not needed - model files are included directly)
```

This clones the upstream hand model into `vendor/orcahand_description/`. You can now reference `vendor/orcahand_description/v2/scene_right.xml` etc.

---

## Step 4: Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
exec $SHELL
uv --version   # should print 0.11+
```

---

## Step 5: Install system dependencies

```bash
sudo apt update
sudo apt install -y build-essential git curl wget libgl1 libosmesa6 ffmpeg
```

---

## Step 6: Create project files

### 6.1 `pyproject.toml`

```bash
cat > pyproject.toml << 'EOF'
[project]
name = "orcaqd"
version = "0.1.0"
description = "Quality-Diversity RL for the OrcaHand v2 dexterous manipulator"
readme = "README.md"
requires-python = ">=3.11,<3.13"
dependencies = [
    "mujoco>=3.8.1,<4.0",
    "mujoco-mjx>=3.8.1,<4.0",
    "jax>=0.4.28",
    "flax>=0.8.5",
    "optax>=0.2.4",
    "chex>=0.1.86",
    "qdax>=0.5.0",
    "numpy>=1.26",
    "matplotlib>=3.8",
    "wandb>=0.18",
    "pyyaml>=6.0",
    "pydantic>=2.7",
    "tqdm>=4.66",
]

[project.optional-dependencies]
cuda = ["jax[cuda13]>=0.4.28"]
cuda12 = ["jax[cuda12]>=0.4.28"]
agentic = ["openai>=1.40", "anthropic>=0.34", "pillow>=10.0"]
dev = ["pytest>=8.0", "ruff>=0.5", "modal>=1.0"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src"]

[tool.uv]
index-strategy = "unsafe-best-match"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra --ignore=tests/test_urdf.py"

[tool.ruff]
line-length = 100
target-version = "py311"
EOF
```

### 6.2 `.gitignore`

```bash
cat > .gitignore << 'EOF'
__pycache__/
*.pyc
.venv/
*.egg-info/
dist/
build/
.pytest_cache/
.ruff_cache/
wandb/
*.tar.zst
scripts/preview.png
uv.lock
EOF
```

> Note: I include `uv.lock` in `.gitignore` here for simplicity. If you want reproducible builds across machines, **remove** `uv.lock` from `.gitignore` and commit it.

### 6.3 `README.md`

```bash
cat > README.md << 'EOF'
# OrcaQD

Quality-Diversity RL for discovering diverse dexterous grasping skills on the OrcaHand v2.

## Quick start

```bash
# Prerequisites: Linux, uv installed, NVIDIA GPU (optional)
git clone <your-repo-url>
cd orcaqd
uv venv --python 3.11
uv sync --extra cuda --extra dev    # or just --extra dev for CPU
uv run python scripts/build_mjx_mjcf.py
uv run python scripts/smoke_test.py
uv run pytest -v
```

## Documentation

- [setup.md](setup.md) — full environment setup guide
- [paper1.md](paper1.md) — Paper 1 research outline (QD-RL)
- [paper2.md](paper2.md) — Paper 2 research outline (VLM orchestration)

## Repository layout

```
orcaqd/
├── vendor/orcahand_description/   # upstream hand model (git submodule)
├── assets/mjcf/mjx/               # generated MJX-friendly MJCF
├── src/
│   ├── envs/                      # MJX env + behavior descriptors
│   ├── qd_engine/                 # QDax training scripts
│   └── agentic_layer/             # Paper 2: VLM routing
├── scripts/                       # build, smoke test, bench, viewer
├── tests/                         # pytest suite
├── configs/                       # training YAML configs
└── paper1.md, paper2.md           # research outlines
```
```
EOF
```

### 6.4 Directory structure

```bash
mkdir -p src/envs src/qd_engine src/agentic_layer
mkdir -p scripts tests configs assets/mjcf/mjx
```

### 6.5 `src/__init__.py`

```bash
cat > src/__init__.py << 'EOF'
"""OrcaQD: Quality-Diversity RL for the OrcaHand v2."""
__version__ = "0.1.0"
EOF
```

### 6.6 `src/jax_env.py`

```bash
cat > src/jax_env.py << 'EOF'
"""Set JAX/XLA environment variables for MJX performance.
Import this module *before* importing jax.
"""
import os

def configure_xla_for_mjx() -> None:
    desired = "--xla_gpu_triton_gemm_any=true"
    existing = os.environ.get("XLA_FLAGS", "")
    if desired not in existing:
        os.environ["XLA_FLAGS"] = (existing + " " + desired).strip()
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

configure_xla_for_mjx()
EOF
```

### 6.7 `src/envs/__init__.py`

```bash
cat > src/envs/__init__.py << 'EOF'
"""OrcaQD environments."""
EOF
```

---

## Step 7: Copy the research files

The following files were developed during the research setup phase. Copy them into the new repo. They reference the hand model via `vendor/orcahand_description/v2/...`.

**Important:** In all scripts, the path to the upstream model changes from `v2/...` to `vendor/orcahand_description/v2/...`. The generated MJX MJCF stays at `assets/mjcf/mjx/`.

The key files to copy from the current workspace:

| Source (current workspace) | Destination in `orcaqd/` |
|---|---|
| `scripts/build_mjx_mjcf.py` | `scripts/build_mjx_mjcf.py` |
| `scripts/smoke_test.py` | `scripts/smoke_test.py` |
| `scripts/bench_throughput.py` | `scripts/bench_throughput.py` |
| `scripts/render_preview.py` | `scripts/render_preview.py` |
| `scripts/view.py` | `scripts/view.py` |
| `src/envs/orcahand_mjx_env.py` | `src/envs/orcahand_mjx_env.py` |
| `src/envs/bd_extractors.py` | `src/envs/bd_extractors.py` |
| `tests/test_mjx_model.py` | `tests/test_mjx_model.py` |
| `tests/test_bd_extractors.py` | `tests/test_bd_extractors.py` |
| `tests/test_env.py` | `tests/test_env.py` |
| `paper1.md` | `paper1.md` |
| `paper2.md` | `paper2.md` |
| `setup.md` | `setup.md` |

**After copying, update all path references** from:
- `v2/` → `vendor/orcahand_description/v2/`
- `v1/` → `vendor/orcahand_description/v1/`

The files that need this path update:
- `scripts/build_mjx_mjcf.py` (UPSTREAM_MJCF, UPSTREAM_BODY paths)
- `scripts/smoke_test.py` (the upstream scene path)
- `scripts/view.py` (PRESETS dict)
- `src/envs/orcahand_mjx_env.py` (DEFAULT_SCENE — this one already points to `assets/mjcf/mjx/` so it's fine)
- `tests/test_mjx_model.py` (SCENE path — already points to `assets/mjcf/mjx/` so it's fine)

---

## Step 8: Create the venv and install

```bash
cd ~/projects/orcaqd
uv venv --python 3.11
uv sync --extra cuda --extra dev    # GPU
# uv sync --extra dev               # CPU only
```

---

## Step 9: Generate the MJX MJCF

```bash
uv run python scripts/build_mjx_mjcf.py
```

Expected output:
```
Reading vendor/orcahand_description/v2/models/mjcf/orcahand_right.mjcf
Reading vendor/orcahand_description/v2/models/mjcf/orcahand_right_body.xml
  demoted 30 mesh geoms to visual class
  inserted 12 primitive collision geoms
Wrote assets/mjcf/mjx/orcahand_right_mjx.mjcf
Wrote assets/mjcf/mjx/scene_right_mjx.xml
```

---

## Step 10: Verify

```bash
uv run python scripts/smoke_test.py
# Expected: [OK] assets/mjcf/mjx/scene_right_mjx.xml

uv run pytest -v
# Expected: 33 passed (or similar count, all green)

uv run python scripts/view.py
# Opens interactive viewer — Esc to close

uv run python scripts/bench_throughput.py --batch 256 --steps 100
# Expected: ~50K steps/sec on RTX 3060, much higher on H100
```

---

## Step 11: Push to GitHub

```bash
git add -A
git commit -m "Initial OrcaQD setup: env, descriptors, MJX MJCF, tests"
git remote add origin git@github.com:<your-username>/orcaqd.git
git branch -M main
git push -u origin main
```

---

## Step 12: Modal setup (for production GPU runs)

```bash
# Install modal (already in dev deps)
modal token new              # links to your Modal workspace

# Create WandB secret
modal secret create wandb WANDB_API_KEY=<your-key>

# Smoke test on Modal H100 (~$0.10)
modal run -m src.modal_app::smoke

# Full training run (~$10, 3 hours)
modal run --detach -m src.modal_app::train --config configs/paper1_main.yaml
```

---

## Day-to-day commands

| Task | Command |
|---|---|
| Run all tests | `uv run pytest -v` |
| Lint | `uv run ruff check src/` |
| Regenerate MJX MJCF | `uv run python scripts/build_mjx_mjcf.py` |
| Smoke test | `uv run python scripts/smoke_test.py` |
| Benchmark | `uv run python scripts/bench_throughput.py --batch 256 --steps 100` |
| View hand (MJX variant) | `uv run python scripts/view.py` |
| View hand (upstream) | `uv run python scripts/view.py --upstream` |
| Add a dependency | `uv add <package>` then `uv sync --extra cuda --extra dev` |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `jax.devices()` shows CPU only | Re-run `uv sync --extra cuda --extra dev` |
| `mjx.put_model()` fails with plane/mesh error | You're loading the upstream MJCF. Use `assets/mjcf/mjx/scene_right_mjx.xml` |
| `uv sync` can't resolve `jax[cuda13]` | Confirm `[tool.uv].index-strategy = "unsafe-best-match"` in pyproject.toml |
| Viewer doesn't open | Set `DISPLAY=:0` (WSLg) or use `scripts/render_preview.py` for headless PNG |
| Tests fail with "file not found" for MJCF | Run `uv run python scripts/build_mjx_mjcf.py` first |
