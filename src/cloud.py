"""Modal entrypoint for OrcaQD training runs.

Cost-conscious workflow (your $21 credit budget):
  1. modal run -m src.cloud::smoke              # ~$0.02 on L4 - container build + GPU check
  2. modal run -m src.cloud::bench              # ~$0.05 on L4 - throughput at small batch
  3. modal run -m src.cloud::train_short        # ~$0.50 on A100-80GB - 5 min training
  4. modal run --detach -m src.cloud::train_budget  # ~$8-12 on A100-80GB - budget paper run
  5. modal run --detach -m src.cloud::train     # ~$10+ on A100-80GB - full headline run

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
#
# Modal rule: ALL `run_commands` must come BEFORE any `add_local_*` calls,
# unless you pass copy=True to add_local_*. We use copy=True for pyproject
# and uv.lock so we can run `uv sync` after them.

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
    # Copy lockfile + project metadata with copy=True so we can run_commands after.
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
    # Local source and assets last (no run_commands after these).
    # These get added to the container at startup, not baked into the image.
    .add_local_dir("orcahand", "/root/orcahand")
    .add_local_dir("mjx", "/root/mjx")
    .add_local_dir("configs", "/root/configs")
    .add_local_python_source("src")
)

app = modal.App("orcaqd", image=image)

# Persistent volume for archive checkpoints.
volume = modal.Volume.from_name("orcaqd-artifacts", create_if_missing=True)

# Wandb secret — create in Modal dashboard with key WANDB_API_KEY.
# Name it "wandb" at https://modal.com/secrets/amugoodbad229/main
wandb_secret = modal.Secret.from_name("wandb")


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
    """Throughput benchmark on L4 (~$0.10). Measures steps/sec at batch 256."""
    import sys
    import time
    sys.path.insert(0, "/root")

    import jax
    import mujoco
    import mujoco.mjx as mjx

    print("=== Throughput benchmark on Modal L4 ===")
    print(f"JAX: {jax.__version__}, devices: {jax.devices()}")

    m = mujoco.MjModel.from_xml_path("/root/mjx/scene_right_mjx.xml")
    mx = mjx.put_model(m)
    print(f"Model: nq={m.nq}, nv={m.nv}, nu={m.nu}")

    batch_size = 256
    n_steps = 100

    @jax.jit
    def vmapped_step(data):
        return jax.vmap(mjx.step, in_axes=(None, 0))(mx, data)

    @jax.jit
    def init_batch(seed):
        import jax.numpy as jnp
        key = jax.random.PRNGKey(seed)
        keys = jax.random.split(key, batch_size)
        def make_one(k):
            d = mjx.make_data(mx)
            noise = jax.random.uniform(k, shape=d.qpos.shape, minval=-0.01, maxval=0.01)
            return d.replace(qpos=d.qpos + noise)
        return jax.vmap(make_one)(keys)

    print(f"batch={batch_size}, steps={n_steps}")
    print("Warming up (JIT compile)...")
    t0 = time.time()
    d = init_batch(0)
    for _ in range(n_steps):
        d = vmapped_step(d)
    d.qpos.block_until_ready()
    print(f"First call (compile + run): {time.time() - t0:.1f}s")

    # Timed runs
    timings = []
    for i in range(5):
        t0 = time.time()
        for _ in range(n_steps):
            d = vmapped_step(d)
        d.qpos.block_until_ready()
        dt = time.time() - t0
        timings.append(dt)
        sps = batch_size * n_steps / dt
        print(f"  iter {i}: {dt*1000:.0f}ms, {sps:,.0f} steps/sec")

    avg = sum(timings) / len(timings)
    sps = batch_size * n_steps / avg
    print(f"\nAverage: {avg*1000:.0f}ms, {sps:,.0f} steps/sec total")
    print("✅ Benchmark complete.")


@app.function(
    gpu="A100-80GB",
    timeout=30 * 60,
    volumes={"/artifacts": volume},
    secrets=[wandb_secret],
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
    timeout=8 * 60 * 60,
    volumes={"/artifacts": volume},
    secrets=[wandb_secret],
)
def train_budget():
    """Budget-conscious run (~$15-17, 6-7 hours on A100-80GB).

    GA-only (PG emitter is dead code), env_batch=64, 25x25 archive, 700 iterations.
    Produces a meaningful archive for paper figures within a $21 credit budget.
    Total policy evaluations: 89,600 (143x the 625-cell archive).
    """
    import os
    import sys
    sys.path.insert(0, "/root")
    from src.qd_engine.train import main as train_main

    out_dir = f"/artifacts/runs/budget_{os.environ.get('MODAL_TASK_ID', 'local')}"
    os.makedirs(out_dir, exist_ok=True)

    train_main(config_path="/root/configs/paper1_budget.yaml", out_dir=out_dir)
    volume.commit()


@app.function(
    gpu="A100-80GB",
    timeout=4 * 60 * 60,
    volumes={"/artifacts": volume},
    secrets=[wandb_secret],
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
        modal run src/cloud.py --action smoke
        modal run src/cloud.py --action bench
        modal run src/cloud.py --action train_short
        modal run --detach src/cloud.py --action train
    """
    if action == "smoke":
        smoke.remote()
    elif action == "bench":
        bench.remote()
    elif action == "train_short":
        train_short.spawn()
    elif action == "train_budget":
        train_budget.spawn()
    elif action == "train":
        train.spawn()
    else:
        raise ValueError(f"Unknown action: {action}. Choose: smoke, bench, train_short, train_budget, train")
