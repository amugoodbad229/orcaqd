"""Behavior descriptor extractors for the OrcaHand v2 QD archive.

Two descriptors are computed from a trajectory's contact data during the
"lift window" (the period after the object is grasped and being held):

  b1 — Contact Dispersion: trace of the spatial covariance of active
       hand-object contact points. Units: m². Small = fingertip-clustered
       (pinch), large = spread across palm (power wrap).

  b2 — Thumb Force Ratio: fraction of total distal-phalanx contact force
       contributed by the thumb. Range [0, 1]. ~0.5 = opposed-thumb power
       grasp, ~0 = lateral pinch without thumb, ~1 = thumb-dominant.

Both are computed from mjx.Data.contact fields and are fully JIT-compatible.

Usage:
    from src.envs.bd_extractors import compute_descriptors

    # Inside a rollout loop, accumulate contact data, then:
    b1, b2 = compute_descriptors(contact_positions, contact_forces, digit_ids)
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


# Digit identifiers. These map to the collision geom names in the generated MJCF.
# The geom names follow the pattern: right_<body>_collision
# We identify digits by their body-name prefix.
DIGIT_NAMES = ["thumb", "index", "middle", "ring", "pinky"]

# Geom name substrings that identify each digit's distal phalanx collision geom.
# These are the geoms whose contact forces contribute to b2.
DISTAL_GEOM_SUBSTRINGS = {
    "thumb": "T-DP_",
    "index": "I-FingerTipAssembly_",
    "middle": "M-FingerTipAssembly_34afb748",   # middle finger
    "ring": "M-FingerTipAssembly_424a8e75",     # ring finger (shares M- prefix)
    "pinky": "P-FingerTipAssembly_",
}


def compute_contact_dispersion(
    contact_pos: jax.Array,
    contact_active: jax.Array,
) -> jax.Array:
    """Compute b1: trace of the contact-position covariance matrix.

    Args:
        contact_pos: (max_contacts, 3) world-frame contact positions.
        contact_active: (max_contacts,) boolean mask of active contacts.

    Returns:
        b1: scalar, trace(Cov(active contact positions)). Units: m².
             Returns 0 if fewer than 2 active contacts.
    """
    n_active = jnp.sum(contact_active)

    # Mask inactive contacts by zeroing their positions.
    mask = contact_active[:, None].astype(jnp.float32)  # (N, 1)
    masked_pos = contact_pos * mask  # (N, 3)

    # Mean of active contacts.
    mean_pos = jnp.sum(masked_pos, axis=0) / jnp.maximum(n_active, 1.0)  # (3,)

    # Centered positions (inactive ones contribute 0).
    centered = (contact_pos - mean_pos) * mask  # (N, 3)

    # Covariance trace = sum of variances along x, y, z.
    cov_trace = jnp.sum(centered ** 2) / jnp.maximum(n_active, 1.0)

    # Return 0 if fewer than 2 contacts (covariance undefined).
    return jnp.where(n_active >= 2, cov_trace, 0.0)


def compute_thumb_force_ratio(
    contact_force: jax.Array,
    contact_active: jax.Array,
    is_thumb_contact: jax.Array,
) -> jax.Array:
    """Compute b2: fraction of total distal contact force from the thumb.

    Args:
        contact_force: (max_contacts,) normal force magnitude per contact.
        contact_active: (max_contacts,) boolean mask of active contacts.
        is_thumb_contact: (max_contacts,) boolean mask identifying thumb-distal contacts.

    Returns:
        b2: scalar in [0, 1]. Returns 0.5 if total force is 0 (no contacts).
    """
    active_force = contact_force * contact_active.astype(jnp.float32)
    total_force = jnp.sum(active_force)
    thumb_force = jnp.sum(active_force * is_thumb_contact.astype(jnp.float32))

    # Avoid division by zero; default to 0.5 (neutral) if no force.
    return jnp.where(total_force > 1e-8, thumb_force / total_force, 0.5)


def compute_descriptors(
    contact_pos: jax.Array,
    contact_force: jax.Array,
    contact_active: jax.Array,
    is_thumb_contact: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Compute both behavior descriptors from a single timestep's contact data.

    Args:
        contact_pos: (max_contacts, 3) world-frame contact positions.
        contact_force: (max_contacts,) normal force magnitudes.
        contact_active: (max_contacts,) boolean mask.
        is_thumb_contact: (max_contacts,) boolean mask for thumb-distal contacts.

    Returns:
        (b1, b2): tuple of scalars.
    """
    b1 = compute_contact_dispersion(contact_pos, contact_active)
    b2 = compute_thumb_force_ratio(contact_force, contact_active, is_thumb_contact)
    return b1, b2


def accumulate_descriptors_over_window(
    contact_pos_seq: jax.Array,
    contact_force_seq: jax.Array,
    contact_active_seq: jax.Array,
    is_thumb_contact: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Average descriptors over a time window (the lift window).

    Args:
        contact_pos_seq: (T, max_contacts, 3)
        contact_force_seq: (T, max_contacts)
        contact_active_seq: (T, max_contacts)
        is_thumb_contact: (max_contacts,) — static mask, same across timesteps.

    Returns:
        (b1_mean, b2_mean): time-averaged descriptors over the window.
    """
    def per_step(carry, inputs):
        b1_sum, b2_sum, count = carry
        pos, force, active = inputs
        b1, b2 = compute_descriptors(pos, force, active, is_thumb_contact)
        # Only accumulate if there are active contacts.
        has_any = jnp.any(active)
        b1_sum = b1_sum + b1 * has_any
        b2_sum = b2_sum + b2 * has_any
        count = count + has_any.astype(jnp.float32)
        return (b1_sum, b2_sum, count), None

    init = (jnp.float32(0.0), jnp.float32(0.0), jnp.float32(0.0))
    (b1_sum, b2_sum, count), _ = jax.lax.scan(
        per_step,
        init,
        (contact_pos_seq, contact_force_seq, contact_active_seq),
    )

    # Average; default to (0, 0.5) if no contacts in the window.
    b1_mean = jnp.where(count > 0, b1_sum / count, 0.0)
    b2_mean = jnp.where(count > 0, b2_sum / count, 0.5)
    return b1_mean, b2_mean
