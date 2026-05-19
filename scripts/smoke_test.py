"""End-to-end environment smoke test.

Run with:
    uv run python scripts/smoke_test.py

Reports JAX device, MuJoCo / MJX versions, and tries to load the OrcaHand v2
in three configurations:
  1. Upstream v2 scene (native mujoco only — fails on MJX-JAX as expected)
  2. Generated MJX-friendly scene (must work on both native + MJX-JAX)
"""
from __future__ import annotations
import os
import sys
import time

# Apply MJX-recommended XLA flags before jax is imported anywhere.
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import src.jax_env  # noqa: F401  (side-effect import)

import jax
import mujoco
import mujoco.mjx as mjx


def banner(msg: str) -> None:
    print()
    print(msg)
    print("-" * len(msg))


def try_load(path: str, expect_mjx: bool) -> bool:
    print(f"\n[{path}] (mjx expected: {'YES' if expect_mjx else 'no'})")
    if not os.path.exists(path):
        print("  skip: file not found")
        return False
    try:
        m = mujoco.MjModel.from_xml_path(path)
        print(f"  native mujoco: ok (nq={m.nq}, nv={m.nv}, nu={m.nu}, ngeom={m.ngeom})")
    except Exception as e:
        print(f"  native mujoco FAILED: {e}")
        return False
    try:
        t0 = time.time()
        mx = mjx.put_model(m)
        d = mjx.make_data(m)
        d = jax.jit(mjx.step)(mx, d)
        d.qpos.block_until_ready()
        dt = time.time() - t0
        print(f"  mjx-jax: ok in {dt*1000:.0f} ms (qpos.shape={d.qpos.shape})")
        return True
    except Exception as e:
        print(f"  mjx-jax FAILED: {type(e).__name__}: {e}")
        return False


def main() -> int:
    banner("Versions and devices")
    print(f"jax     : {jax.__version__}, backend={jax.default_backend()}")
    print(f"mujoco  : {mujoco.__version__}")
    print(f"devices : {jax.devices()}")

    banner("OrcaHand v2 load attempts")
    cases = [
        ("vendor/orcahand_description/v2/scene_right.xml", False),
        ("assets/mjcf/mjx/scene_right_mjx.xml", True),
    ]
    results = []
    for path, expect in cases:
        results.append((path, expect, try_load(path, expect)))

    banner("Summary")
    fail = False
    for path, expected, ok in results:
        if expected:
            verdict = "OK" if ok else "FAIL"
            if not ok:
                fail = True
        else:
            verdict = "expected-fail (correct)" if not ok else "unexpected pass"
        print(f"  [{verdict}] {path}")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
