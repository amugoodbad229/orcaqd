"""MJX environment for QD-RL on high-DOF dexterous hands.

Hand-agnostic: parameterized by a HandConfig. Works with any anthropomorphic
hand that has position-controlled actuators and an opposable thumb.

Demonstrated on OrcaHand v2 (17 DOF) but designed for 16-24 DOF hands.

Usage:
    from src.envs.dex_env import DexHandEnv, EnvConfig
    from src.envs.hand_config import ORCAHAND_RIGHT

    cfg = EnvConfig(hand=ORCAHAND_RIGHT)
    env = DexHandEnv(cfg)
    state = env.reset(jax.random.PRNGKey(0))
    state = env.step(state, action)
"""
from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx
import numpy as np

from src.envs.hand_config import HandConfig, ORCAHAND_RIGHT

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_SCENE = str(ROOT / "mjx" / "scene_right_mjx.xml")

# Object geom name (must match what build_mjx_mjcf.py injects)
OBJECT_GEOM_NAME = "object_geom"
OBJECT_BODY_NAME = "object"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class EnvConfig:
    """Environment configuration."""

    hand: HandConfig = ORCAHAND_RIGHT
    scene_path: str = DEFAULT_SCENE
    episode_length: int = 250          # control steps
    dt: float = 0.002                  # physics timestep
    ctrl_dt: float = 0.02             # control timestep (50 Hz)
    lift_threshold: float = 0.05       # meters above init to count as lifted

    # Reward weights
    w_contact: float = 0.5
    w_lift: float = 10.0
    w_success: float = 5.0
    w_action_rate: float = 0.001

    # Domain randomization
    dr_mass_range: tuple[float, float] = (0.05, 0.5)
    dr_friction_range: tuple[float, float] = (0.4, 1.2)
    dr_kp_scale_range: tuple[float, float] = (0.9, 1.1)
    dr_obj_pos_range: float = 0.02
    dr_obj_yaw_range: float = 0.26


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class EnvState(NamedTuple):
    """Per-environment state carried across steps."""
    mjx_data: mjx.Data
    obs: jax.Array
    reward: jax.Array
    done: jax.Array
    step_count: jax.Array
    prev_action: jax.Array
    obj_init_z: jax.Array
    has_contact: jax.Array
    # Descriptor accumulators (running sums over the lift window)
    bd_b1_sum: jax.Array
    bd_b2_sum: jax.Array
    bd_count: jax.Array


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class DexHandEnv:
    """JAX-native dexterous hand environment for QD-RL.

    Hand-agnostic: parameterized by HandConfig.
    Designed to be used inside jax.vmap for batched rollouts.
    """

    def __init__(self, config: EnvConfig | None = None):
        self.cfg = config or EnvConfig()
        hand = self.cfg.hand

        # Load model.
        self.mj_model = mujoco.MjModel.from_xml_path(self.cfg.scene_path)
        self.mjx_model = mjx.put_model(self.mj_model)

        # Pre-compute default MJX data template (avoids reallocation per reset).
        self._default_data = mjx.make_data(self.mjx_model)

        self.nq = self.mj_model.nq
        self.nv = self.mj_model.nv
        self.nu = self.mj_model.nu
        self.n_substeps = max(1, int(self.cfg.ctrl_dt / self.cfg.dt))

        # Determine if there's a free-body object in the scene.
        self._obj_body_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, OBJECT_BODY_NAME
        )
        self.has_object = self._obj_body_id >= 0

        if self.has_object:
            # Object's qpos starts at joint_qposadr for its free joint.
            obj_jnt_id = self.mj_model.body_jntadr[self._obj_body_id]
            self._obj_qpos_start = self.mj_model.jnt_qposadr[obj_jnt_id]
            # obs = hand joints + hand vels + obj pose(7) + obj vel(6)
            self.obs_dim = hand.n_actuators * 2 + 13
        else:
            self._obj_qpos_start = -1
            # obs = hand joints + hand vels only
            self.obs_dim = hand.n_actuators * 2

        self.action_dim = hand.n_actuators

        # Build static geom-ID masks for descriptor computation.
        self._build_geom_masks(hand)

    def _build_geom_masks(self, hand: HandConfig) -> None:
        """Pre-compute boolean masks mapping geom IDs to digits."""
        n_geom = self.mj_model.ngeom

        # Mask: is this geom part of the hand?
        self._hand_geom_mask = np.zeros(n_geom, dtype=bool)
        # Mask: is this geom the thumb distal?
        self._thumb_geom_mask = np.zeros(n_geom, dtype=bool)
        # Mask: is this geom the object?
        self._object_geom_mask = np.zeros(n_geom, dtype=bool)

        for i in range(n_geom):
            name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if name is None:
                continue
            if hand.hand_geom_prefix in name and "_collision" in name:
                self._hand_geom_mask[i] = True
                if hand.thumb.distal_geom_substring in name:
                    self._thumb_geom_mask[i] = True
            if name == OBJECT_GEOM_NAME:
                self._object_geom_mask[i] = True

        # Convert to JAX arrays for use in JIT'd functions.
        self._hand_geom_ids = jnp.array(np.where(self._hand_geom_mask)[0], dtype=jnp.int32)
        self._thumb_geom_ids = jnp.array(np.where(self._thumb_geom_mask)[0], dtype=jnp.int32)
        self._object_geom_ids = jnp.array(np.where(self._object_geom_mask)[0], dtype=jnp.int32)

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(self, key: jax.Array) -> EnvState:
        """Reset to a new initial state."""
        # Small random perturbation to hand joints.
        key, subkey = jax.random.split(key)
        n_hand_q = self.cfg.hand.n_actuators
        qpos_noise = jax.random.uniform(
            subkey, shape=(n_hand_q,), minval=-0.01, maxval=0.01
        )
        # Use pre-computed default data template (avoids mjx.make_data allocation).
        new_qpos = self._default_data.qpos.at[:n_hand_q].add(qpos_noise)
        data = self._default_data.replace(qpos=new_qpos)

        # Forward to compute derived quantities.
        data = mjx.forward(self.mjx_model, data)

        obs = self._get_obs(data)
        obj_init_z = jnp.where(
            self.has_object,
            data.qpos[self._obj_qpos_start + 2],  # z component of object pos
            jnp.float32(0.0),
        )

        return EnvState(
            mjx_data=data,
            obs=obs,
            reward=jnp.float32(0.0),
            done=jnp.bool_(False),
            step_count=jnp.int32(0),
            prev_action=jnp.zeros(self.action_dim),
            obj_init_z=obj_init_z,
            has_contact=jnp.bool_(False),
            bd_b1_sum=jnp.float32(0.0),
            bd_b2_sum=jnp.float32(0.0),
            bd_count=jnp.float32(0.0),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(self, state: EnvState, action: jax.Array) -> EnvState:
        """Take one control step (multiple physics substeps)."""
        action = jnp.clip(action, -1.0, 1.0)
        data = state.mjx_data.replace(ctrl=action)

        # Substep physics.
        def substep(d, _):
            return mjx.step(self.mjx_model, d), None
        data, _ = jax.lax.scan(substep, data, jnp.arange(self.n_substeps))

        obs = self._get_obs(data)

        # Contact detection and descriptor accumulation.
        has_contact, b1_step, b2_step, n_contacts = self._extract_contacts(data)
        has_contact = jnp.logical_or(state.has_contact, has_contact)

        # Accumulate descriptors (only when there are hand-object contacts).
        has_valid = n_contacts >= 2
        bd_b1_sum = state.bd_b1_sum + b1_step * has_valid
        bd_b2_sum = state.bd_b2_sum + b2_step * has_valid
        bd_count = state.bd_count + has_valid.astype(jnp.float32)

        # Reward.
        reward = self._compute_reward(state, data, action, has_contact)

        step_count = state.step_count + 1
        done = step_count >= self.cfg.episode_length

        return EnvState(
            mjx_data=data,
            obs=obs,
            reward=reward,
            done=done,
            step_count=step_count,
            prev_action=action,
            obj_init_z=state.obj_init_z,
            has_contact=has_contact,
            bd_b1_sum=bd_b1_sum,
            bd_b2_sum=bd_b2_sum,
            bd_count=bd_count,
        )

    def get_descriptors(self, state: EnvState) -> jax.Array:
        """Extract final (b1, b2) descriptors from accumulated state."""
        b1 = jnp.where(state.bd_count > 0, state.bd_b1_sum / state.bd_count, 0.0)
        b2 = jnp.where(state.bd_count > 0, state.bd_b2_sum / state.bd_count, 0.5)
        return jnp.array([b1, b2])

    def _get_obs(self, data: mjx.Data) -> jax.Array:
        """Extract observation vector."""
        n_hand = self.cfg.hand.n_actuators
        hand_qpos = data.qpos[:n_hand]
        hand_qvel = data.qvel[:n_hand]

        if self.has_object:
            obj_pos = data.qpos[self._obj_qpos_start:self._obj_qpos_start + 7]
            obj_vel = data.qvel[self._obj_qpos_start - 1:self._obj_qpos_start + 5]
            # Note: free joint qvel is 6D (3 linear + 3 angular), starts at qveladr
            return jnp.concatenate([hand_qpos, hand_qvel, obj_pos, obj_vel])
        return jnp.concatenate([hand_qpos, hand_qvel])

    def _extract_contacts(self, data: mjx.Data) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Extract hand-object contacts and compute per-step descriptors.

        Returns:
            has_contact: bool, any hand-object contact exists
            b1: contact dispersion (trace of covariance)
            b2: thumb force ratio
            n_contacts: number of active hand-object contacts
        """
        contact = data._impl.contact if hasattr(data, '_impl') else data.contact
        active = contact.dist < 0  # penetration = active contact

        # Identify hand-object contact pairs.
        # contact.geom is (max_contacts, 2) — the two geom IDs in each pair.
        geom1 = contact.geom[:, 0]
        geom2 = contact.geom[:, 1]

        # A contact is hand-object if one geom is hand and the other is object.
        g1_hand = jnp.isin(geom1, self._hand_geom_ids)
        g2_hand = jnp.isin(geom2, self._hand_geom_ids)
        g1_obj = jnp.isin(geom1, self._object_geom_ids)
        g2_obj = jnp.isin(geom2, self._object_geom_ids)

        is_hand_obj = (g1_hand & g2_obj) | (g2_hand & g1_obj)
        valid = active & is_hand_obj

        n_contacts = jnp.sum(valid)
        has_contact = n_contacts > 0

        # Contact positions for b1.
        contact_pos = contact.pos  # (max_contacts, 3)
        mask = valid[:, None].astype(jnp.float32)
        masked_pos = contact_pos * mask
        mean_pos = jnp.sum(masked_pos, axis=0) / jnp.maximum(n_contacts, 1.0)
        centered = (contact_pos - mean_pos) * mask
        b1 = jnp.sum(centered ** 2) / jnp.maximum(n_contacts, 1.0)

        # Thumb force ratio for b2.
        # Identify which contacts involve the thumb distal geom.
        g1_thumb = jnp.isin(geom1, self._thumb_geom_ids)
        g2_thumb = jnp.isin(geom2, self._thumb_geom_ids)
        is_thumb = (g1_thumb | g2_thumb) & valid

        # Force magnitude: use the penetration depth as a proxy (dist is negative).
        force_proxy = jnp.abs(contact.dist) * valid.astype(jnp.float32)
        total_force = jnp.sum(force_proxy)
        thumb_force = jnp.sum(force_proxy * is_thumb.astype(jnp.float32))
        b2 = jnp.where(total_force > 1e-8, thumb_force / total_force, 0.5)

        return has_contact, b1, b2, n_contacts

    def _compute_reward(
        self, state: EnvState, data: mjx.Data, action: jax.Array, has_contact: jax.Array
    ) -> jax.Array:
        """Staged reward: contact + lift + success - action rate."""
        cfg = self.cfg

        # Contact bonus.
        r_contact = cfg.w_contact * has_contact.astype(jnp.float32)

        # Lift reward (only if object exists).
        if self.has_object:
            obj_z = data.qpos[self._obj_qpos_start + 2]
            delta_z = obj_z - state.obj_init_z
            # Only reward upward motion after contact.
            r_lift = cfg.w_lift * jnp.maximum(delta_z, 0.0) * has_contact.astype(jnp.float32)
            r_success = cfg.w_success * (delta_z > cfg.lift_threshold).astype(jnp.float32)
        else:
            r_lift = jnp.float32(0.0)
            r_success = jnp.float32(0.0)

        # Action-rate penalty.
        action_diff = action - state.prev_action
        r_action_rate = -cfg.w_action_rate * jnp.sum(action_diff ** 2)

        return r_contact + r_lift + r_success + r_action_rate
