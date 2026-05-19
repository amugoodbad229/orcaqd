"""Modal entrypoint for OrcaQD training runs.

Usage:
    modal run -m src.modal_app::smoke           # GPU verification (~$0.10)
    modal run -m src.modal_app::train           # full training (~$10)
    modal run --detach -m src.modal_app::train  # detached (close laptop)
"""
import modal

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
    .run_commands("uv sync --frozen --extra cuda --no-install-project")
    .add_local_dir("orcahand", "/root/orcahand")
    .add_local_dir("mjx", "/root/mjx")
    .add_local_dir("configs", "/root/configs")
    .add_local_python_source("src")
    .env({"PATH": "/root/.venv/bin:${PATH}", "VIRTUAL_ENV": "/root/.venv"})
)

app = modal.App("orcaqd", image=image)
volume = modal.Volume.from_name("orcaqd-artifacts", create_if_missing=True)


@app.function(gpu="H100", timeout=4 * 60 * 60, volumes={"/artifacts": volume},
              secrets=[modal.Secret.from_name("wandb")])
def train(config: str = "configs/paper1_main.yaml"):
    """Run full PGA-MAP-Elites training."""
    import os, sys
    sys.path.insert(0, "/root")
    from src.qd_engine.train_pga_map_elites import main as train_main
    out_dir = f"/artifacts/runs/{os.environ.get('MODAL_TASK_ID', 'local')}"
    os.makedirs(out_dir, exist_ok=True)
    train_main(config_path=f"/root/{config}", out_dir=out_dir)
    volume.commit()


@app.function(gpu="H100", timeout=10 * 60)
def smoke():
    """30-second smoke test: GPU + MJX + env."""
    import jax, mujoco, mujoco.mjx as mjx
    print(f"JAX: {jax.__version__}, devices: {jax.devices()}")
    print(f"MuJoCo: {mujoco.__version__}")
    m = mujoco.MjModel.from_xml_path("/root/mjx/scene_right_mjx.xml")
    mx = mjx.put_model(m)
    d = mjx.make_data(m)
    d = jax.jit(mjx.step)(mx, d)
    print(f"Model: nq={m.nq}, nv={m.nv}, nu={m.nu}")
    print("MJX step OK.")
