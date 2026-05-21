"""Interactive viewer for the OrcaHand v2.

Pops a MuJoCo viewer window so you can rotate/zoom/inspect the model. Works
on:
  * Linux desktop (X11)
  * WSL2 + WSLg (Windows 11) — sets DISPLAY=:0 if unset
  * Linux server with X forwarding (set DISPLAY before launching)

Usage:
    uv run python scripts/view.py                     # MJX variant (default)
    uv run python scripts/view.py --upstream          # upstream v2 model
    uv run python scripts/view.py --left              # upstream v2 left hand
    uv run python scripts/view.py --combined          # upstream v2 both hands

Mouse:
  * Left drag       rotate
  * Right drag      pan
  * Scroll          zoom
  * Double-click    select body
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PRESETS = {
    "mjx": ROOT / "mjx" / "scene_right_mjx.xml",
    "upstream": ROOT / "orcahand" / "scene_right.xml",
    "left": ROOT / "orcahand" / "scene_left.xml",
    "combined": ROOT / "orcahand" / "scene_combined.xml",
}


def ensure_display() -> bool:
    """Make sure DISPLAY is set for the WSLg X server."""
    if os.environ.get("DISPLAY"):
        return True
    # WSLg exposes X11 at :0
    if Path("/tmp/.X11-unix/X0").exists():
        os.environ["DISPLAY"] = ":0"
        print(f"[view] auto-set DISPLAY=:0 (detected WSLg)")
        return True
    print("[view] WARNING: no DISPLAY and no /tmp/.X11-unix/X0 — viewer may fail")
    return False


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--upstream", action="store_true",
                   help="open upstream v2 right-hand scene")
    p.add_argument("--left", action="store_true",
                   help="open upstream v2 left-hand scene")
    p.add_argument("--combined", action="store_true",
                   help="open upstream v2 combined (both hands) scene")
    p.add_argument("--mjcf", type=str, default=None,
                   help="explicit MJCF path (overrides preset flags)")
    args = p.parse_args()

    if args.mjcf:
        path = Path(args.mjcf)
    elif args.combined:
        path = PRESETS["combined"]
    elif args.left:
        path = PRESETS["left"]
    elif args.upstream:
        path = PRESETS["upstream"]
    else:
        path = PRESETS["mjx"]

    if not path.exists():
        print(f"[view] ERROR: file not found: {path}")
        return 1

    ensure_display()

    print(f"[view] loading: {path}")
    import mujoco
    import mujoco.viewer

    m = mujoco.MjModel.from_xml_path(str(path))
    print(f"[view] model: nq={m.nq}, nv={m.nv}, nu={m.nu}, ngeom={m.ngeom}")
    print(f"[view] starting interactive viewer...")
    print(f"[view] close the window or press Esc to exit")

    d = mujoco.MjData(m)
    mujoco.viewer.launch(m, d)
    return 0


if __name__ == "__main__":
    sys.exit(main())
