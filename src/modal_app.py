"""Modal entrypoint for OrcaQD training runs.

Cost-conscious workflow (your $30 credit budget):
  1. modal run -m src.modal_app::smoke           # ~$0.02 on L4 - container build + GPU check
  2. modal run -m src.modal_app::bench           # ~$0.05 on L4 - throughput at small batch
  3. modal run -m src.modal_app::train_short     # ~$0.50 on A100-80GB - 5 min training
  4. modal run --detach -m src.modal_app::train  # ~$8 on A100-80GB - full headline run

Cost reference (Modal pricing, verified May 2026):
  L4:           $0.80/hr   ($0.000222/sec)
  A100-40GB:    $2.10/hr   ($0.000583/sec)
  A100-80GB:    $2.50/hr   ($0.000694/sec)
  H100:         $3.95/hr   ($0.001097/sec)
"""
import modal


# ---------------------------------------------------------------------------
# Container image
# ---------------------------------------------------------------------------
# Build the image with uv, using the same pyproject.toml + uv.lock as locally.
# This guarantees the container has the exact same dependency tree.

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
    # Copy lockfile + project metadata first so the dep layer caches.
    .add_local_file("pyproject.toml", "/root/pyproject.toml", copy=True)
    .add_local_file("uv.lock", "/root/uv.lock", copy=True)
    .workdir("/root")
    .run_commands(
        "uv sync --frozen --extra cuda --no-install-project"
    )
    # Then add assets (changes here don't bust the dep cache).
    .add_local_dir("orcahand", "/root/orcahand")
    .add_local_dir("mjx", "/root/mjx")
    .add_local_dir("configs", "/root/configs")
    .add_local_python_source("src")
    .env({
        "PATH": "/root/.venv/bin:${PATH}",
        "VIRTUAL_ENV": "/root/.venv",
        # MJX recommended XLA flag.
        "XLA_FLAGS": "--xla_gpu_triton_gemm_any=true",
    })
)

app = modal.App("orcaqd", image=image)

# Persistent volume for archive checkpoints.
volume = modal.Volume.from_name("orcaqd-artifacts", create_if_missing=True)


# ---------------------------------------------------------------------------
# Functions (ordered cheapest → most expensive)
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
    print("✅ Smoke test passed.")


@app.function(gpu="L4", timeout=15 * 60)
def bench():
    """Throughput benchmark on L4 (~$0.10). Measures steps/sec at small batch."""
    import sys
    sys.path.insert(0, "/root")
    import subprocess
    print("=== Throughput benchmark on Modal L4 ===")
    subprocess.run(
        ["python", "/root/scripts/bench.py", "--batch", "256", "--steps", "100"],
        check=True,
    )


@app.function(
    gpu="A100-80GB",
    timeout=15 * 60,
    volumes={"/artifacts": volume},
)
def train_short():
    """Short A100-80GB run (~$0.50, 5-10 minutes).

    Validates that the training loop scales and produces a reasonable archive
    in a small time budget before committing to the full run.
    """
    import os
    import sys
    sys.path.insert(0, "/root")
    from src.qd_engine.train import main as train_main

    out_dir = f"/artifacts/runs/short_{os.environ.get('MODAL_TASK_ID', 'local')}"
    os.makedirs(out_dir, exist_ok=True)

    # Use a smaller config for the short run.
    train_main(config_path="/root/configs/paper1_short.yaml", out_dir=out_dir)
    volume.commit()


@app.function(
    gpu="A100-80GB",
    timeout=4 * 60 * 60,
    volumes={"/artifacts": volume},
    secrets=[modal.Secret.from_name("wandb")],
)
def train(config: str = "configs/paper1_main.yaml"):
    """Full headline training run on A100-80GB (~$8-10 for 3-4 hours)."""
    import os
    import sys
    sys.path.insert(0, "/root")
    from src.qd_engine.train import main as train_main

    out_dir = f"/artifacts/runs/{os.environ.get('MODAL_TASK_ID', 'local')}"
    os.makedirs(out_dir, exist_ok=True)
    train_main(config_path=f"/root/{config}", out_dir=out_dir)
    volume.commit()


# ---------------------------------------------------------------------------
# Local entrypoint (for `modal run` without explicit function name)
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(action: str = "smoke"):
    """Run an action: smoke, bench, train_short, or train.

    Usage:
        modal run src/modal_app.py --action smoke
        modal run src/modal_app.py --action bench
        modal run src/modal_app.py --action train_short
        modal run --detach src/modal_app.py --action train
    """
    if action == "smoke":
        smoke.remote()
    elif action == "bench":
        bench.remote()
    elif action == "train_short":
        train_short.remote()
    elif action == "train":
        train.remote()
    else:
        raise ValueError(f"Unknown action: {action}. Choose: smoke, bench, train_short, train")
