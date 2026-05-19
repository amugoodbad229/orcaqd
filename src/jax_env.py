"""Set JAX/XLA environment variables for performance.

Import this module *before* importing jax. The recommended flags follow
https://mujoco.readthedocs.io/en/stable/mjx.html#gpu-performance .
"""
from __future__ import annotations
import os


def configure_xla_for_mjx() -> None:
    """Apply XLA flags recommended by the MJX docs.

    Idempotent: if XLA_FLAGS is already set, we extend it rather than overwrite.
    """
    desired = "--xla_gpu_triton_gemm_any=true"
    existing = os.environ.get("XLA_FLAGS", "")
    if desired not in existing:
        os.environ["XLA_FLAGS"] = (existing + " " + desired).strip()

    # Avoid TF allocator messages from JAX dependencies.
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")


# Apply on import.
configure_xla_for_mjx()
