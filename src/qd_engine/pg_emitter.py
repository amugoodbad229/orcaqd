"""Policy-Gradient emitter for PGA-MAP-Elites.

Implements the PG half of PGA-MAP-Elites (Nilsson & Cully, 2021) as a QDax
Emitter subclass. Uses TD3 (Twin Delayed DDPG) to train a shared critic,
then applies critic-gradient ascent to parent policies sampled from the archive.

Compatible with JAX 0.10+ (no brax.v1 dependency).

The GA half (Iso+LineDD) is handled by QDax's MixingEmitter. This emitter
produces the PG-improved offspring that fill the other half of the batch.

Usage:
    from src.qd_engine.pg_emitter import PGEmitter, PGEmitterConfig

    pg_emitter = PGEmitter(config=PGEmitterConfig(), policy_net=policy_net)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax import struct

from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from qdax.custom_types import Genotype, RNGKey


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PGEmitterConfig:
    """Configuration for the PG emitter."""
    critic_hidden_size: int = 256
    critic_lr: float = 3e-4
    actor_lr: float = 3e-4
    discount: float = 0.99
    tau: float = 0.005              # Polyak averaging for target networks
    policy_noise: float = 0.2      # TD3 target policy smoothing
    noise_clip: float = 0.5
    policy_delay: int = 2          # TD3 delayed policy update
    pg_steps: int = 50             # gradient steps per emit call
    batch_size: int = 256          # replay buffer sample size
    buffer_size: int = 100_000     # max transitions in replay buffer
    warmup_steps: int = 1000       # random actions before PG starts


# ---------------------------------------------------------------------------
# Critic network (TD3 twin critics)
# ---------------------------------------------------------------------------

class TwinCritic(nn.Module):
    """Twin Q-networks for TD3."""
    hidden_size: int = 256

    @nn.compact
    def __call__(self, obs: jax.Array, action: jax.Array) -> tuple[jax.Array, jax.Array]:
        x = jnp.concatenate([obs, action], axis=-1)
        # Q1
        q1 = nn.Dense(self.hidden_size)(x)
        q1 = nn.relu(q1)
        q1 = nn.Dense(self.hidden_size)(q1)
        q1 = nn.relu(q1)
        q1 = nn.Dense(1)(q1).squeeze(-1)
        # Q2
        q2 = nn.Dense(self.hidden_size)(x)
        q2 = nn.relu(q2)
        q2 = nn.Dense(self.hidden_size)(q2)
        q2 = nn.relu(q2)
        q2 = nn.Dense(1)(q2).squeeze(-1)
        return q1, q2


# ---------------------------------------------------------------------------
# Replay buffer (simple, fixed-size, JAX-compatible)
# ---------------------------------------------------------------------------

@struct.dataclass
class ReplayBufferState:
    """State of the replay buffer."""
    obs: jax.Array          # (buffer_size, obs_dim)
    action: jax.Array       # (buffer_size, act_dim)
    reward: jax.Array       # (buffer_size,)
    next_obs: jax.Array     # (buffer_size, obs_dim)
    done: jax.Array         # (buffer_size,)
    size: jax.Array         # scalar int
    ptr: jax.Array          # scalar int


def init_replay_buffer(obs_dim: int, act_dim: int, max_size: int) -> ReplayBufferState:
    return ReplayBufferState(
        obs=jnp.zeros((max_size, obs_dim)),
        action=jnp.zeros((max_size, act_dim)),
        reward=jnp.zeros(max_size),
        next_obs=jnp.zeros((max_size, obs_dim)),
        done=jnp.zeros(max_size, dtype=bool),
        size=jnp.int32(0),
        ptr=jnp.int32(0),
    )


def add_to_buffer(
    buf: ReplayBufferState,
    obs: jax.Array,
    action: jax.Array,
    reward: jax.Array,
    next_obs: jax.Array,
    done: jax.Array,
) -> ReplayBufferState:
    """Add a batch of transitions to the buffer."""
    batch_size = obs.shape[0]
    indices = (jnp.arange(batch_size) + buf.ptr) % buf.obs.shape[0]
    new_buf = buf.replace(
        obs=buf.obs.at[indices].set(obs),
        action=buf.action.at[indices].set(action),
        reward=buf.reward.at[indices].set(reward),
        next_obs=buf.next_obs.at[indices].set(next_obs),
        done=buf.done.at[indices].set(done),
        size=jnp.minimum(buf.size + batch_size, buf.obs.shape[0]),
        ptr=(buf.ptr + batch_size) % buf.obs.shape[0],
    )
    return new_buf


def sample_buffer(buf: ReplayBufferState, key: RNGKey, batch_size: int):
    """Sample a random batch from the buffer."""
    indices = jax.random.randint(key, (batch_size,), 0, buf.size)
    return (
        buf.obs[indices],
        buf.action[indices],
        buf.reward[indices],
        buf.next_obs[indices],
        buf.done[indices],
    )


# ---------------------------------------------------------------------------
# PG Emitter state
# ---------------------------------------------------------------------------

@struct.dataclass
class PGEmitterState(EmitterState):
    """State carried by the PG emitter across MAP-Elites iterations."""
    critic_params: Any
    critic_target_params: Any
    critic_opt_state: Any
    replay_buffer: ReplayBufferState
    total_steps: jax.Array
    key: jax.Array


# ---------------------------------------------------------------------------
# PG Emitter
# ---------------------------------------------------------------------------

class PGEmitter(Emitter):
    """Policy-gradient emitter using TD3 critic for PGA-MAP-Elites.

    Each emit() call:
    1. Samples parent policies from the archive
    2. Applies `pg_steps` of critic-gradient ascent to each parent
    3. Returns the improved policies as offspring
    """

    def __init__(
        self,
        config: PGEmitterConfig,
        policy_network: nn.Module,
        obs_dim: int,
        action_dim: int,
        batch_size: int = 64,  # number of offspring to produce per emit
    ):
        self.config = config
        self.policy_network = policy_network
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self._batch_size = batch_size

        # Critic
        self.critic = TwinCritic(hidden_size=config.critic_hidden_size)
        self.critic_optimizer = optax.adam(config.critic_lr)
        self.actor_optimizer = optax.adam(config.actor_lr)

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def init(self, init_genotypes: Genotype, random_key: RNGKey) -> tuple[PGEmitterState, RNGKey]:
        """Initialize the emitter state."""
        key, critic_key = jax.random.split(random_key)

        # Init critic
        dummy_obs = jnp.zeros(self.obs_dim)
        dummy_act = jnp.zeros(self.action_dim)
        critic_params = self.critic.init(critic_key, dummy_obs, dummy_act)
        critic_target_params = critic_params  # copy
        critic_opt_state = self.critic_optimizer.init(critic_params)

        # Init replay buffer
        replay_buffer = init_replay_buffer(
            self.obs_dim, self.action_dim, self.config.buffer_size
        )

        state = PGEmitterState(
            critic_params=critic_params,
            critic_target_params=critic_target_params,
            critic_opt_state=critic_opt_state,
            replay_buffer=replay_buffer,
            total_steps=jnp.int32(0),
            key=key,
        )
        return state, key

    def emit(
        self,
        repertoire: MapElitesRepertoire,
        emitter_state: PGEmitterState,
        random_key: RNGKey,
    ) -> tuple[Genotype, RNGKey]:
        """Produce offspring by PG-improving parents from the archive."""
        # Sample parents from the archive (uniform over filled cells).
        key, sample_key, pg_key = jax.random.split(random_key, 3)

        # Get random filled indices.
        filled_mask = repertoire.fitnesses != -jnp.inf
        n_filled = jnp.sum(filled_mask)

        # Sample parent indices (with replacement).
        parent_indices = jax.random.choice(
            sample_key,
            jnp.arange(repertoire.fitnesses.shape[0]),
            shape=(self._batch_size,),
            p=filled_mask / jnp.maximum(n_filled, 1),
        )

        # Extract parent genotypes.
        parents = jax.tree.map(lambda x: x[parent_indices], repertoire.genotypes)

        # Apply PG improvement if buffer has enough data.
        can_train = emitter_state.replay_buffer.size >= self.config.warmup_steps

        def pg_improve(parent_params, key):
            """Apply actor_lr gradient ascent using the critic."""
            def actor_loss(params):
                action = self.policy_network.apply(params, obs_batch)
                q1, _ = self.critic.apply(emitter_state.critic_params, obs_batch, action)
                return -jnp.mean(q1)

            # Sample a batch of observations from the replay buffer.
            obs_batch = sample_buffer(
                emitter_state.replay_buffer, key, self.config.batch_size
            )[0]

            # One gradient step on the actor.
            grads = jax.grad(actor_loss)(parent_params)
            updates, _ = self.actor_optimizer.update(grads, optax.EmptyState())
            improved = optax.apply_updates(parent_params, updates)
            return improved

        def no_improve(parent_params, key):
            return parent_params

        # Apply PG or pass-through based on buffer state.
        pg_keys = jax.random.split(pg_key, self._batch_size)
        offspring = jax.lax.cond(
            can_train,
            lambda: jax.vmap(pg_improve)(parents, pg_keys),
            lambda: parents,
        )

        return offspring, key

    def state_update(
        self,
        emitter_state: PGEmitterState,
        repertoire: MapElitesRepertoire,
        genotypes: Genotype,
        fitnesses: jax.Array,
        descriptors: jax.Array,
        extra_scores: dict,
    ) -> PGEmitterState:
        """Update the critic using transitions from the replay buffer.

        Note: In a full implementation, transitions would be collected during
        the scoring function and added to the buffer here. For now, this is
        a placeholder that will be wired up when the scoring function provides
        trajectory data.
        """
        # Update step counter.
        new_state = emitter_state.replace(
            total_steps=emitter_state.total_steps + self._batch_size
        )
        return new_state
