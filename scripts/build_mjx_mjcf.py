"""Generate an MJX-JAX-compatible MJCF for the OrcaHand v2.

The upstream OrcaHand v2 description (in v2/) uses high-poly mesh collision geoms
with margin="0.0005" everywhere. MJX-JAX cannot handle:
  * (plane, mesh) margin/gap collisions
  * convex-convex mesh collisions for the >32-vertex hulls in this model

This script reads the upstream MJCF + body XML and emits a self-contained
MJX-friendly MJCF where:
  1. Every original mesh geom is demoted to visual-only (contype=0 conaffinity=0)
     and class="visual" (margin=0).
  2. Per-phalanx primitive collision capsules and a palm box are inserted, all
     with margin=0 and class="collision".
  3. Asset (mesh) declarations and actuator definitions are copied from upstream
     so the MJX MJCF is a single self-contained file.
  4. Mesh file paths in the asset block are rewritten relative to the new file
     location.
  5. Contact-exclude pairs are copied from upstream.

Outputs:
    assets/mjcf/orcahand_right_mjx.mjcf      (single self-contained MJCF)
    assets/mjcf/scene_right_mjx.xml          (scene wrapper with floor)

Usage:
    uv run python scripts/build_mjx_mjcf.py
"""
from __future__ import annotations
import sys
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UPSTREAM_MJCF = ROOT / "orcahand" / "models" / "mjcf" / "orcahand_right.mjcf"
UPSTREAM_BODY = ROOT / "orcahand" / "models" / "mjcf" / "orcahand_right_body.xml"
OUT_MJCF = ROOT / "assets" / "mjcf" / "mjx" / "orcahand_right_mjx.mjcf"
OUT_SCENE = ROOT / "assets" / "mjcf" / "mjx" / "scene_right_mjx.xml"

# Mesh STLs are referenced from upstream relative to v2/. The upstream MJCF at
# v2/models/mjcf/orcahand_right.mjcf has paths like "models/assets/right/X.stl"
# which resolve to v2/models/assets/right/X.stl. Our MJX MJCF lives at
# assets/mjcf/mjx/, so the rewrite is three levels up to repo root, then into v2.
MESH_PATH_REWRITE = "../../../orcahand/{}"


# --- Per-body collision-primitive specifications.
#
# Capsule fromto coordinates are in the body's local frame, where +Z is the
# segment's long axis. Sizes are conservative cylinder/sphere approximations
# of each phalanx's dimensions (read from the visual STL bounding extents in
# the upstream MJCF).
PRIMITIVE_SPECS: dict[str, dict] = {
    # Palm (carpals) — central rectangular box.
    "right_R-Carpals_": {
        "type": "box",
        "size": "0.040 0.025 0.012",
        "pos": "0 0 0",
    },
    # Index proximal phalanx
    "right_I-PP_": {
        "type": "capsule",
        "fromto": "0 0 0  0 0 0.037",
        "size": "0.008",
    },
    # Index distal (tip)
    "right_I-FingerTipAssembly_": {
        "type": "capsule",
        "fromto": "0 0 0  0 0 0.025",
        "size": "0.007",
    },
    # Middle proximal phalanx (M-PP under M-AP_e04a96f2)
    "right_M-PP_08efa608": {
        "type": "capsule",
        "fromto": "0 0 0  0 0 0.040",
        "size": "0.009",
    },
    "right_M-FingerTipAssembly_34afb748": {
        "type": "capsule",
        "fromto": "0 0 0  0 0 0.025",
        "size": "0.007",
    },
    # Ring proximal phalanx (M-PP under M-AP_6ec59111)
    "right_M-PP_8660a1eb": {
        "type": "capsule",
        "fromto": "0 0 0  0 0 0.040",
        "size": "0.008",
    },
    "right_M-FingerTipAssembly_424a8e75": {
        "type": "capsule",
        "fromto": "0 0 0  0 0 0.025",
        "size": "0.007",
    },
    # Pinky proximal phalanx
    "right_P-PP_": {
        "type": "capsule",
        "fromto": "0 0 0  0 0 0.031",
        "size": "0.007",
    },
    "right_P-FingerTipAssembly_": {
        "type": "capsule",
        "fromto": "0 0 0  0 0 0.020",
        "size": "0.006",
    },
    # Thumb trapezium link
    "right_T-TP-R_": {
        "type": "capsule",
        "fromto": "0 0 0  0 0 0.018",
        "size": "0.007",
    },
    # Thumb proximal phalanx
    "right_T-PP_": {
        "type": "capsule",
        "fromto": "0 0 0  0 0 0.032",
        "size": "0.008",
    },
    # Thumb distal (tip)
    "right_T-DP_": {
        "type": "capsule",
        "fromto": "0 0 0  0 0 0.022",
        "size": "0.007",
    },
}
EXPECTED_PRIMITIVE_COUNT = 12


def find_spec(body_name: str) -> dict | None:
    """Longest-prefix match against PRIMITIVE_SPECS keys."""
    matches = sorted(
        (k for k in PRIMITIVE_SPECS if body_name.startswith(k)),
        key=len,
        reverse=True,
    )
    return PRIMITIVE_SPECS[matches[0]] if matches else None


def demote_mesh_geoms_to_visual(root: ET.Element) -> int:
    n = 0
    for geom in root.iter("geom"):
        if "mesh" in geom.attrib:
            geom.set("contype", "0")
            geom.set("conaffinity", "0")
            geom.set("class", "visual")
            for attr in ("margin", "condim", "friction"):
                geom.attrib.pop(attr, None)
            n += 1
    return n


def insert_primitive_collisions(root: ET.Element) -> int:
    n = 0
    for body in root.iter("body"):
        spec = find_spec(body.get("name", ""))
        if spec is None:
            continue
        attribs = {
            "name": f"{body.get('name')}_collision",
            "class": "collision",
            "type": spec["type"],
            "rgba": "0.5 0.5 0.5 0.25",
        }
        for k in ("fromto", "pos", "size"):
            if k in spec:
                attribs[k] = spec[k]
        geom = ET.SubElement(body, "geom", attribs)
        body.remove(geom)
        body.insert(0, geom)
        n += 1
    return n


def rewrite_mesh_paths(asset_block: ET.Element) -> None:
    """Rewrite mesh file= attributes to be reachable from assets/mjcf/."""
    for mesh in asset_block.findall("mesh"):
        f = mesh.get("file")
        if f is not None:
            mesh.set("file", MESH_PATH_REWRITE.format(f))


def prettify(elem: ET.Element, level: int = 0) -> None:
    indent = "  "
    i = "\n" + level * indent
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + indent
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for child in elem:
            prettify(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = i
    elif level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i


def build_mjx_mjcf() -> tuple[int, int]:
    upstream = ET.parse(UPSTREAM_MJCF).getroot()
    body_tree = ET.parse(UPSTREAM_BODY).getroot()

    # Modify the body in-place.
    visual_count = demote_mesh_geoms_to_visual(body_tree)
    primitive_count = insert_primitive_collisions(body_tree)

    # Build the new MJCF.
    mj = ET.Element("mujoco", {"model": "orcahand_right_mjx"})
    mj.append(ET.Comment(
        " Auto-generated by scripts/build_mjx_mjcf.py from v2/models/mjcf/orcahand_right(_body).xml. "
        "Do not edit by hand; re-run the script. "
    ))

    # Compiler options.
    ET.SubElement(mj, "compiler", {"angle": "radian", "eulerseq": "XYZ"})

    # MJX-JAX performance tuning. Per
    # https://mujoco.readthedocs.io/en/stable/mjx.html#performance-tuning :
    #   * NEWTON solver converges in very few iterations on GPU.
    #   * iterations / ls_iterations down to "just stable".
    #   * eulerdamp off for performance (re-enable if instability appears).
    #   * dense Jacobian on GPU when nv < 60 (we have nv=17).
    option_attrs = {
        "timestep": "0.002",
        "iterations": "1",
        "ls_iterations": "5",
        "solver": "Newton",
        "integrator": "Euler",
        "jacobian": "dense",
        "cone": "elliptic",
    }
    option = ET.SubElement(mj, "option", option_attrs)
    ET.SubElement(option, "flag", {"eulerdamp": "disable"})

    # Broadphase budgets for MJX-JAX. These cap the size of the contact buffer
    # that MJX allocates per environment. Tune up if you see "too many contacts"
    # warnings; tune down for memory if needed.
    custom = ET.SubElement(mj, "custom")
    ET.SubElement(custom, "numeric", {
        "name": "max_contact_points", "data": "32",
    })
    ET.SubElement(custom, "numeric", {
        "name": "max_geom_pairs", "data": "64",
    })

    # Defaults: visual class has margin=0 and contact disabled; collision class
    # has margin=0 (MJX-JAX requirement) and standard friction.
    default = ET.SubElement(mj, "default")
    # Top-level joint defaults (mirror upstream).
    ET.SubElement(default, "joint", {
        "type": "hinge", "limited": "true", "damping": "0.1",
        "armature": "0.001", "margin": "0.01", "frictionloss": "0.001",
    })
    ET.SubElement(default, "position", {
        "ctrllimited": "true", "forcelimited": "true",
        "forcerange": "-3 3", "kp": "5.0",
    })
    ET.SubElement(default, "mesh", {
        "scale": "0.001 0.001 0.001",
        # MJX-JAX performance tuning: cap convex-hull vertex count.
        # The docs recommend <=64 for general convex-mesh collisions
        # and <=32 for convex-convex; we go with 64 since our meshes
        # are visual-only by default.
        "maxhullvert": "64",
    })

    visual_d = ET.SubElement(default, "default", {"class": "visual"})
    ET.SubElement(visual_d, "geom", {
        "type": "mesh", "contype": "0", "conaffinity": "0",
        "margin": "0", "group": "2",
    })

    collision_d = ET.SubElement(default, "default", {"class": "collision"})
    ET.SubElement(collision_d, "geom", {
        "contype": "1", "conaffinity": "1", "condim": "3",
        "margin": "0", "friction": "1.0 0.005 0.001", "group": "3",
    })

    # Materials (we generate our own minimal set rather than rely on options.xml).
    asset = ET.SubElement(mj, "asset")
    ET.SubElement(asset, "material", {"name": "white", "rgba": "1 1 1 1"})
    ET.SubElement(asset, "material", {"name": "black", "rgba": "0.16 0.16 0.16 1"})
    # Copy mesh declarations from upstream and rewrite paths.
    upstream_asset = upstream.find("asset")
    if upstream_asset is not None:
        for mesh in upstream_asset.findall("mesh"):
            asset.append(deepcopy(mesh))
        rewrite_mesh_paths(asset)

    # Worldbody: include the modified hand body tree.
    worldbody = ET.SubElement(mj, "worldbody")
    # Insert each top-level body from the modified body XML.
    for body in body_tree.findall("body"):
        worldbody.append(body)

    # Actuators: copy verbatim from upstream.
    upstream_actuator = upstream.find("actuator")
    if upstream_actuator is not None:
        mj.append(deepcopy(upstream_actuator))

    # Contact-exclude pairs: copy verbatim from upstream.
    upstream_contact = upstream.find("contact")
    if upstream_contact is not None:
        mj.append(deepcopy(upstream_contact))

    OUT_MJCF.parent.mkdir(parents=True, exist_ok=True)
    prettify(mj)
    ET.ElementTree(mj).write(OUT_MJCF, encoding="utf-8", xml_declaration=True)

    return visual_count, primitive_count


def write_scene() -> None:
    scene = ET.Element("mujoco", {"model": "orcahand_right_mjx_scene"})
    scene.append(ET.Comment(" Auto-generated by scripts/build_mjx_mjcf.py "))
    ET.SubElement(scene, "include", {"file": "orcahand_right_mjx.mjcf"})
    ET.SubElement(scene, "statistic", {"extent": "0.3", "center": "0 0 0.15"})
    visual = ET.SubElement(scene, "visual")
    ET.SubElement(visual, "rgba", {"haze": "0.15 0.25 0.35 1"})
    ET.SubElement(visual, "global", {"azimuth": "220", "elevation": "-30"})
    asset = ET.SubElement(scene, "asset")
    ET.SubElement(asset, "texture", {
        "type": "skybox", "builtin": "gradient",
        "rgb1": "0.3 0.5 0.7", "rgb2": "0 0 0",
        "width": "512", "height": "3072",
    })
    ET.SubElement(asset, "texture", {
        "type": "2d", "name": "groundplane", "builtin": "checker",
        "mark": "edge", "rgb1": "0.2 0.3 0.4", "rgb2": "0.1 0.2 0.3",
        "markrgb": "0.8 0.8 0.8", "width": "300", "height": "300",
    })
    ET.SubElement(asset, "material", {
        "name": "groundplane", "texture": "groundplane",
        "texuniform": "true", "texrepeat": "5 5", "reflectance": "0.2",
    })

    worldbody = ET.SubElement(scene, "worldbody")
    ET.SubElement(worldbody, "light", {
        "pos": "0 0.5 1", "dir": "0 -0.3 -1",
        "diffuse": "0.8 0.8 0.8", "specular": "0.2 0.2 0.2",
    })
    ET.SubElement(worldbody, "geom", {
        "name": "floor", "type": "plane", "size": "0 0 0.05",
        "pos": "0 0 0", "material": "groundplane",
        "contype": "1", "conaffinity": "1", "condim": "3",
        "margin": "0",
    })

    OUT_SCENE.parent.mkdir(parents=True, exist_ok=True)
    prettify(scene)
    ET.ElementTree(scene).write(OUT_SCENE, encoding="utf-8", xml_declaration=True)


def main() -> int:
    if not UPSTREAM_MJCF.exists() or not UPSTREAM_BODY.exists():
        print("FATAL: upstream OrcaHand v2 files missing.", file=sys.stderr)
        return 1

    print(f"Reading {UPSTREAM_MJCF.relative_to(ROOT)}")
    print(f"Reading {UPSTREAM_BODY.relative_to(ROOT)}")
    visual_count, primitive_count = build_mjx_mjcf()
    print(f"  demoted {visual_count} mesh geoms to visual class")
    print(f"  inserted {primitive_count} primitive collision geoms")

    write_scene()
    print(f"Wrote {OUT_MJCF.relative_to(ROOT)}")
    print(f"Wrote {OUT_SCENE.relative_to(ROOT)}")

    if primitive_count != EXPECTED_PRIMITIVE_COUNT:
        print(
            f"WARNING: inserted {primitive_count} primitives; "
            f"expected {EXPECTED_PRIMITIVE_COUNT}.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
