"""Throughput benchmark for the MJX-friendly OrcaHand v2 model.

Measures steps/sec for a batch of parallel environments stepping for `--steps`
each. Reports both raw throughput (steps_per_sec) and per-env throughput.

Usage:
    uv run python scripts/bench_throughput.py
    uv run python scripts/bench_throughput.py --batch 256 --steps 100
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

# Apply MJX-recommended XLA flags before jax is imported.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import src.jax_env  # noqa: F401

import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx

ROOT = Path(__file__).resolve().parent.parent
SCENE = str(ROOT / "mjx" / "scene_right_mjx.xml")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=64,
                   help="number of parallel environments")
    p.add_argument("--steps", type=int, default=100,
                   help="number of physics steps per env")
    p.add_argument("--warmup", type=int, default=5,
                   help="warmup iterations (excluded from timing)")
    args = p.parse_args()

    print(f"jax: {jax.__version__}, devices: {jax.devices()}")
    print(f"loading {SCENE}")

    m = mujoco.MjModel.from_xml_path(SCENE)
    mx = mjx.put_model(m)
    print(f"model: nq={m.nq}, nv={m.nv}, nu={m.nu}, ngeom={m.ngeom}")

    # Build a vmapped step function.
    @jax.jit
    def vmapped_step(data):
        return jax.vmap(mjx.step, in_axes=(None, 0))(mx, data)

    def batched_rollout(data, n_steps: int):
        for _ in range(n_steps):
            data = vmapped_step(data)
        return data

    # Initialize batch.
    @jax.jit
    def init_batch(seed):
        key = jax.random.PRNGKey(seed)
        keys = jax.random.split(key, args.batch)

        def make_one(k):
            d = mjx.make_data(mx)
            qpos_noise = jax.random.uniform(
                k, shape=d.qpos.shape, minval=-0.01, maxval=0.01
            )
            return d.replace(qpos=d.qpos + qpos_noise)

        return jax.vmap(make_one)(keys)

    # Warmup
    print(f"\nbatch={args.batch}, steps_per_call={args.steps}")
    print("warming up (JIT compile)...")
    t0 = time.time()
    d = init_batch(0)
    d = batched_rollout(d, args.steps)
    d.qpos.block_until_ready()
    print(f"first call (compile + run): {time.time() - t0:.1f} s")

    # Timed runs
    print(f"benching {args.warmup} warmup + 5 timed iters...")
    for i in range(args.warmup):
        d = batched_rollout(d, args.steps)
        d.qpos.block_until_ready()

    timings = []
    for i in range(5):
        t0 = time.time()
        d = batched_rollout(d, args.steps)
        d.qpos.block_until_ready()
        dt = time.time() - t0
        timings.append(dt)
        sps = args.batch * args.steps / dt
        print(f"  iter {i}: {dt*1000:.1f} ms, {sps:,.0f} steps/sec")

    avg = sum(timings) / len(timings)
    sps = args.batch * args.steps / avg
    per_env = args.steps / avg
    print(f"\naverage: {avg*1000:.1f} ms")
    print(f"throughput: {sps:,.0f} steps/sec total, "
          f"{per_env:.0f} steps/sec/env")


if __name__ == "__main__":
    main()
