"""Render a preview of the MJX-friendly OrcaHand v2 with collision primitives visible.

Saves a PNG side-by-side: visual mesh only, then collision primitives only.

Usage:
    uv run python scripts/render_preview.py
"""
from __future__ import annotations
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SCENE = str(ROOT / "mjx" / "scene_right_mjx.xml")
OUT = ROOT / "scripts" / "preview.png"


def render_with_groups(model, data, group_visual, group_collision, w=640, h=480):
    opt = mujoco.MjvOption()
    # Default: all groups enabled. We'll toggle via the geomgroup array.
    for g in range(6):
        opt.geomgroup[g] = 0
    if group_visual:
        opt.geomgroup[2] = 1   # visual class -> group 2
    if group_collision:
        opt.geomgroup[3] = 1   # collision class -> group 3
    # Floor lives in group 0 by default; show it always.
    opt.geomgroup[0] = 1

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance = 0.40
    cam.azimuth = 200
    cam.elevation = -25
    cam.lookat[:] = [0.0, 0.0, 0.15]

    renderer = mujoco.Renderer(model, height=h, width=w)
    renderer.update_scene(data, camera=cam, scene_option=opt)
    return renderer.render()


def main() -> None:
    print(f"loading {SCENE}")
    m = mujoco.MjModel.from_xml_path(SCENE)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    img_visual = render_with_groups(m, d, group_visual=True, group_collision=False)
    img_collision = render_with_groups(m, d, group_visual=False, group_collision=True)
    img_both = render_with_groups(m, d, group_visual=True, group_collision=True)

    # Side-by-side composite.
    composite = np.concatenate([img_visual, img_collision, img_both], axis=1)
    try:
        from PIL import Image
        Image.fromarray(composite).save(OUT)
        print(f"wrote {OUT}")
    except ImportError:
        # Fall back to matplotlib if pillow isn't installed.
        import matplotlib.pyplot as plt
        plt.figure(figsize=(15, 5))
        plt.imshow(composite)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(OUT, dpi=100, bbox_inches="tight")
        print(f"wrote {OUT} (via matplotlib)")


if __name__ == "__main__":
    main()
