"""Hand configuration abstraction.

Defines the interface between the QD framework and any specific hand model.
The framework needs to know:
  - How many DOF the hand has
  - Which geoms belong to which digit (for descriptor computation)
  - Which digit is the thumb (for b2)
  - Where to place the object relative to the palm

To add a new hand: create a HandConfig instance with the appropriate mappings.
See ORCAHAND_RIGHT for the reference implementation.

Usage:
    from src.envs.hand_config import ORCAHAND_RIGHT
    env = OrcaHandEnv(hand_config=ORCAHAND_RIGHT)
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DigitConfig:
    """Configuration for a single digit (finger)."""
    name: str                          # human-readable: "thumb", "index", etc.
    joint_names: tuple[str, ...]       # ordered proximal → distal
    distal_geom_substring: str         # substring to match the distal collision geom name
    is_thumb: bool = False


@dataclass(frozen=True)
class HandConfig:
    """Hand-agnostic configuration for the QD-RL framework.

    Any anthropomorphic hand with 16-24 DOF and an opposable thumb can be
    described by this config. The framework uses it to:
      - Determine observation/action dimensions
      - Identify thumb contacts for b2 descriptor
      - Identify hand-vs-object contacts for b1 descriptor
      - Place the object within reach
    """
    name: str                                      # e.g. "orcahand_right", "leap_hand"
    n_actuators: int                               # total actuated DOF
    digits: tuple[DigitConfig, ...]                # all digits including thumb
    palm_geom_substring: str                       # substring to match palm collision geom
    hand_geom_prefix: str                          # prefix shared by all hand collision geoms
    object_init_pos: tuple[float, float, float]    # (x, y, z) where to spawn the object
    wrist_body_name: str                           # body to mount the hand on

    @property
    def thumb(self) -> DigitConfig:
        for d in self.digits:
            if d.is_thumb:
                return d
        raise ValueError(f"No thumb digit in {self.name}")

    @property
    def n_digits(self) -> int:
        return len(self.digits)


# ---------------------------------------------------------------------------
# OrcaHand v2 Right — the primary experimental platform
# ---------------------------------------------------------------------------

ORCAHAND_RIGHT = HandConfig(
    name="orcahand_v2_right",
    n_actuators=17,
    digits=(
        DigitConfig(
            name="thumb",
            joint_names=("right_t-cmc", "right_t-abd", "right_t-mcp", "right_t-pip"),
            distal_geom_substring="T-DP_",
            is_thumb=True,
        ),
        DigitConfig(
            name="index",
            joint_names=("right_i-abd", "right_i-mcp", "right_i-pip"),
            distal_geom_substring="I-FingerTipAssembly_",
        ),
        DigitConfig(
            name="middle",
            joint_names=("right_m-abd", "right_m-mcp", "right_m-pip"),
            distal_geom_substring="M-FingerTipAssembly_34afb748",
        ),
        DigitConfig(
            name="ring",
            joint_names=("right_r-abd", "right_r-mcp", "right_r-pip"),
            distal_geom_substring="M-FingerTipAssembly_424a8e75",
        ),
        DigitConfig(
            name="pinky",
            joint_names=("right_p-abd", "right_p-mcp", "right_p-pip"),
            distal_geom_substring="P-FingerTipAssembly_",
        ),
    ),
    palm_geom_substring="R-Carpals_",
    hand_geom_prefix="right_",
    object_init_pos=(0.0, 0.0, 0.08),  # slightly above the palm
    wrist_body_name="right_R-Carpals_8d1f1041",
)


# ---------------------------------------------------------------------------
# Placeholder configs for other hands (to be filled when needed)
# ---------------------------------------------------------------------------

# LEAP_HAND = HandConfig(
#     name="leap_hand",
#     n_actuators=16,
#     digits=(...),
#     ...
# )

# SHADOW_HAND = HandConfig(
#     name="shadow_hand",
#     n_actuators=20,
#     digits=(...),
#     ...
# )
