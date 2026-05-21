"""Modal entrypoint for OrcaQD training runs.

Usage:
    modal run src/cloud.py --action smoke              # Verify container + GPU
    modal run --detach src/cloud.py --action train_budget    # PGA-ME (paper results)
    modal run --detach src/cloud.py --action train_ga_baseline  # GA-only baseline

Cost reference (Modal pricing, verified May 2026):
    A100-80GB: $2.50/hr ($0.000694/sec)
"""
import modal


# ---------------------------------------------------------------------------
# Container image
# ---------------------------------------------------------------------------

image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.0-cudnn-runtime-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("libgl1", "libosmesa6", "ffmpeg", "git", "curl")
    .run_commands(
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        "ln -sf /root/.local/bin/uv /usr/local/bin/uv",
    )
    .add_local_file("pyproject.toml", "/root/pyproject.toml", copy=True)
    .add_local_file("uv.lock", "/root/uv.lock", copy=True)
    .workdir("/root")
    .run_commands(
        "uv sync --frozen --extra cuda --no-install-project"
    )
    .env({
        "PATH": "/root/.venv/bin:${PATH}",
        "VIRTUAL_ENV": "/root/.venv",
        "XLA_FLAGS": "--xla_gpu_triton_gemm_any=true",
    })
    .add_local_dir("orcahand", "/root/orcahand")
    .add_local_dir("mjx", "/root/mjx")
    .add_local_dir("configs", "/root/configs")
    .add_local_python_source("src")
)

app = modal.App("orcaqd", image=image)

# Persistent volume for archive checkpoints.
volume = modal.Volume.from_name("orcaqd-artifacts", create_if_missing=True)

# Wandb secret — create in Modal dashboard with key WANDB_API_KEY.
wandb_secret = modal.Secret.from_name("wandb")


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

@app.function(gpu="L4", timeout=5 * 60)
def smoke():
    """Cheapest verification (~$0.02 on L4). Confirms container + GPU + MJX work."""
    import jax
    import mujoco
    import mujoco.mjx as mjx

    print(f"=== Smoke test on Modal ===")
    print(f"JAX: {jax.__version__}, devices: {jax.devices()}")
    print(f"MuJoCo: {mujoco.__version__}")

    m = mujoco.MjModel.from_xml_path("/root/mjx/scene_right_mjx.xml")
    mx = mjx.put_model(m)
    d = mjx.make_data(m)
    d = jax.jit(mjx.step)(mx, d)
    d.qpos.block_until_ready()
    print(f"Model: nq={m.nq}, nv={m.nv}, nu={m.nu}, ngeom={m.ngeom}")
    print(f"Step OK. qpos.shape={d.qpos.shape}")
    print("Smoke test passed.")


@app.function(
    gpu="A100-80GB",
    timeout=8 * 60 * 60,
    volumes={"/artifacts": volume},
    secrets=[wandb_secret],
)
def train_budget():
    """PGA-ME run — identical to GA baseline except pg_batch_size=32.
    500 iterations, ~22 min, ~$0.90 on A100-80GB.
    """
    import os
    import sys
    sys.path.insert(0, "/root")
    from src.qd_engine.train import main as train_main

    out_dir = f"/artifacts/runs/pga_me_500_{os.environ.get('MODAL_TASK_ID', 'local')}"
    os.makedirs(out_dir, exist_ok=True)

    train_main(config_path="/root/configs/paper1_budget.yaml", out_dir=out_dir)
    volume.commit()


@app.function(
    gpu="A100-80GB",
    timeout=8 * 60 * 60,
    volumes={"/artifacts": volume},
    secrets=[wandb_secret],
)
def train_ga_baseline():
    """GA-only baseline — identical to PGA-ME except pg_batch_size=0.
    500 iterations, ~22 min, ~$0.90 on A100-80GB.
    """
    import os
    import sys
    sys.path.insert(0, "/root")
    from src.qd_engine.train import main as train_main

    out_dir = f"/artifacts/runs/ga_baseline_500_{os.environ.get('MODAL_TASK_ID', 'local')}"
    os.makedirs(out_dir, exist_ok=True)

    train_main(config_path="/root/configs/paper1_ga_baseline.yaml", out_dir=out_dir)
    volume.commit()


@app.function(
    gpu="A100-80GB",
    timeout=4 * 60 * 60,
    volumes={"/artifacts": volume},
    secrets=[wandb_secret],
)
def train(config: str = "configs/paper1_main.yaml"):
    """Full headline training run on A100-80GB."""
    import os
    import sys
    sys.path.insert(0, "/root")
    from src.qd_engine.train import main as train_main

    out_dir = f"/artifacts/runs/{os.environ.get('MODAL_TASK_ID', 'local')}"
    os.makedirs(out_dir, exist_ok=True)
    train_main(config_path=f"/root/{config}", out_dir=out_dir)
    volume.commit()


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(action: str = "smoke"):
    """Run an action: smoke, train_budget, train_ga_baseline, or train."""
    if action == "smoke":
        smoke.remote()
    elif action == "train_budget":
        train_budget.spawn()
    elif action == "train_ga_baseline":
        train_ga_baseline.spawn()
    elif action == "train":
        train.spawn()
    else:
        raise ValueError(f"Unknown action: {action}. Choose: smoke, train_budget, train_ga_baseline, train")
