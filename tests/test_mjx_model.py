"""Tests for the MJX-friendly OrcaHand v2 MJCF.

Run with:
    uv run pytest tests/test_mjx_model.py -v
"""
from __future__ import annotations
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCENE = ROOT / "assets" / "mjcf" / "mjx" / "scene_right_mjx.xml"


@pytest.fixture(scope="module")
def model():
    import mujoco
    return mujoco.MjModel.from_xml_path(str(SCENE))


def test_mjx_scene_exists():
    assert SCENE.exists(), (
        f"MJX scene missing at {SCENE}. "
        "Run: uv run python scripts/build_mjx_mjcf.py"
    )


def test_dof_count(model):
    assert model.nq == 17, f"expected 17 qpos, got {model.nq}"
    assert model.nv == 17, f"expected 17 qvel, got {model.nv}"
    assert model.nu == 17, f"expected 17 actuators, got {model.nu}"


def test_collision_geom_count(model):
    """Expect 12 primitive collision geoms (palm + 11 phalanges) plus the floor."""
    n_collision = sum(1 for i in range(model.ngeom) if model.geom_contype[i] != 0)
    assert n_collision == 13, (
        f"expected 13 collision geoms (12 hand primitives + 1 floor), got {n_collision}"
    )


def test_no_mesh_collisions(model):
    """No mesh geom should be collision-enabled (MJX-JAX requirement)."""
    import mujoco
    for i in range(model.ngeom):
        if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH:
            assert model.geom_contype[i] == 0, (
                f"mesh geom {i} ({mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)}) "
                "has contype != 0"
            )


def test_zero_margin(model):
    """All geoms should have margin=0 (MJX-JAX cannot handle margin>0 for plane/mesh)."""
    import numpy as np
    margins = np.array(model.geom_margin)
    assert np.allclose(margins, 0.0), f"expected all margins 0, got max {margins.max()}"


def test_mjx_loads_and_steps(model):
    """End-to-end: model must load on MJX-JAX and step without raising."""
    import jax
    import mujoco.mjx as mjx

    mx = mjx.put_model(model)
    d = mjx.make_data(model)
    step_jit = jax.jit(mjx.step)
    d = step_jit(mx, d)
    d.qpos.block_until_ready()
    assert d.qpos.shape == (17,)
