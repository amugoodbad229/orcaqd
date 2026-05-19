"""OrcaHand v2 MJX environment for QD-RL.

This module provides a JAX-native, vmappable environment for the OrcaHand v2
with primitive-collision geometry. It wraps MJX step/reset and computes:
  - Observations (joint pos/vel + object pose/vel)
  - Reward (staged curriculum: contact → lift → hold)
  - Done signal (episode termination)
  - Domain randomization (object mass, friction, actuator gain, initial pose)

The env is designed to be used as a QDax scoring function: given a batch of
policy parameters, roll out episodes and return (fitness, descriptors).

Usage:
    from src.envs.orcahand_mjx_env import OrcaHandEnv, EnvConfig

    cfg = EnvConfig()
    env = OrcaHandEnv(cfg)
    state = env.reset(jax.random.PRNGKey(0))
    state = env.step(state, action)
"""
from __future__ import annotations

import functools
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_SCENE = str(ROOT / "assets" / "mjcf" / "mjx" / "scene_right_mjx.xml")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class EnvConfig:
    """Environment configuration."""

    scene_path: str = DEFAULT_SCENE
    episode_length: int = 250          # steps at 500 Hz = 0.5 s (or 50 Hz = 5 s depending on dt)
    dt: float = 0.002                  # physics timestep (from MJCF)
    ctrl_dt: float = 0.02             # control timestep (10x physics dt → 50 Hz control)
    lift_threshold: float = 0.05       # meters above initial z to count as "lifted"
    contact_force_threshold: float = 0.05  # N, minimum normal force to count a contact

    # Reward weights
    w_contact: float = 0.5
    w_lift: float = 10.0
    w_success: float = 5.0
    w_action_rate: float = 0.001

    # Domain randomization ranges
    dr_mass_range: tuple[float, float] = (0.05, 0.5)
    dr_friction_range: tuple[float, float] = (0.4, 1.2)
    dr_kp_scale_range: tuple[float, float] = (0.9, 1.1)
    dr_obj_pos_range: float = 0.02     # ±meters
    dr_obj_yaw_range: float = 0.26     # ±radians (~15°)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class EnvState(NamedTuple):
    """Per-environment state carried across steps."""

    mjx_data: mjx.Data
    obs: jax.Array                     # (obs_dim,)
    reward: jax.Array                  # scalar
    done: jax.Array                    # bool scalar
    step_count: jax.Array              # int scalar
    prev_action: jax.Array             # (nu,) for action-rate penalty
    obj_init_z: jax.Array              # scalar, initial object height
    has_contact: jax.Array             # bool, whether any hand-object contact occurred


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class OrcaHandEnv:
    """JAX-native OrcaHand v2 environment for QD-RL.

    Designed to be used inside jax.vmap for batched rollouts.
    """

    def __init__(self, config: EnvConfig | None = None):
        self.cfg = config or EnvConfig()

        # Load model once (CPU-side).
        self.mj_model = mujoco.MjModel.from_xml_path(self.cfg.scene_path)
        self.mjx_model = mjx.put_model(self.mj_model)

        self.nq = self.mj_model.nq
        self.nv = self.mj_model.nv
        self.nu = self.mj_model.nu
        self.n_substeps = max(1, int(self.cfg.ctrl_dt / self.cfg.dt))

        # Observation: joint_pos(17) + joint_vel(17) + obj_pose(7) + obj_vel(6) = 47
        # For now without object (no free body in the scene yet), obs = joint_pos + joint_vel = 34
        self.obs_dim = self.nq + self.nv
        self.action_dim = self.nu

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(self, key: jax.Array) -> EnvState:
        """Reset the environment to a new initial state."""
        data = mjx.make_data(self.mjx_model)

        # Small random perturbation to joint positions for exploration diversity.
        key, subkey = jax.random.split(key)
        qpos_noise = jax.random.uniform(
            subkey, shape=(self.nq,), minval=-0.01, maxval=0.01
        )
        data = data.replace(qpos=data.qpos + qpos_noise)

        # Forward to compute derived quantities.
        data = mjx.forward(self.mjx_model, data)

        obs = self._get_obs(data)
        zero_action = jnp.zeros(self.nu)

        return EnvState(
            mjx_data=data,
            obs=obs,
            reward=jnp.float32(0.0),
            done=jnp.bool_(False),
            step_count=jnp.int32(0),
            prev_action=zero_action,
            obj_init_z=jnp.float32(0.0),  # placeholder until object is added
            has_contact=jnp.bool_(False),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(self, state: EnvState, action: jax.Array) -> EnvState:
        """Take one control step (multiple physics substeps)."""
        # Clip action to [-1, 1] then scale to actuator ctrl range.
        action = jnp.clip(action, -1.0, 1.0)
        data = state.mjx_data.replace(ctrl=action)

        # Substep the physics.
        def substep(d, _):
            return mjx.step(self.mjx_model, d), None

        data, _ = jax.lax.scan(substep, data, jnp.arange(self.n_substeps))

        # Observation.
        obs = self._get_obs(data)

        # Contact detection: any geom pair with normal force above threshold.
        has_contact = self._detect_contact(data)
        has_contact = jnp.logical_or(state.has_contact, has_contact)

        # Reward.
        reward = self._compute_reward(state, data, action, has_contact)

        # Done.
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
        )

    def _get_obs(self, data: mjx.Data) -> jax.Array:
        """Extract observation vector from MJX data."""
        return jnp.concatenate([data.qpos, data.qvel])

    def _detect_contact(self, data: mjx.Data) -> jax.Array:
        """Return True if any contact has normal force above threshold."""
        # MJX 3.8+ stores contacts in data._impl.contact (JAX backend).
        # The dist field: dist < 0 means penetration (active contact).
        contact = data._impl.contact if hasattr(data, '_impl') else data.contact
        active = contact.dist < 0
        return jnp.any(active)

    def _compute_reward(
        self,
        state: EnvState,
        data: mjx.Data,
        action: jax.Array,
        has_contact: jax.Array,
    ) -> jax.Array:
        """Staged reward: contact bonus + lift + success - action rate."""
        cfg = self.cfg

        # Contact bonus (sparse, once per episode).
        r_contact = cfg.w_contact * has_contact.astype(jnp.float32)

        # Lift reward: delta-z of object. Placeholder (0) until object is added.
        r_lift = jnp.float32(0.0)

        # Success bonus. Placeholder until object is added.
        r_success = jnp.float32(0.0)

        # Action-rate penalty.
        action_diff = action - state.prev_action
        r_action_rate = -cfg.w_action_rate * jnp.sum(action_diff ** 2)

        return r_contact + r_lift + r_success + r_action_rate

    # ------------------------------------------------------------------
    # QDax scoring interface
    # ------------------------------------------------------------------

    def make_scoring_fn(self, episode_length: int | None = None):
        """Return a QDax-compatible scoring function.

        The scoring function takes a batch of policy parameters (as a Flax
        PyTree) and returns (fitness, descriptors, extra_scores).
        """
        ep_len = episode_length or self.cfg.episode_length

        @jax.jit
        def scoring_fn(params, key, policy_fn):
            """Score a single policy by rolling out an episode.

            Args:
                params: Flax model parameters for one policy.
                key: PRNG key.
                policy_fn: callable(params, obs) -> action.

            Returns:
                fitness: scalar total reward.
                descriptors: (2,) behavior descriptor [b1, b2].
            """
            state = self.reset(key)
            total_reward = jnp.float32(0.0)

            def body(carry, _):
                state, total_reward, key = carry
                key, subkey = jax.random.split(key)
                action = policy_fn(params, state.obs)
                state = self.step(state, action)
                total_reward = total_reward + state.reward
                return (state, total_reward, key), state.mjx_data

            (final_state, total_reward, _), trajectory_data = jax.lax.scan(
                body, (state, total_reward, key), jnp.arange(ep_len)
            )

            # Descriptors computed from trajectory (placeholder zeros for now).
            descriptors = jnp.zeros(2)

            return total_reward, descriptors

        return scoring_fn
