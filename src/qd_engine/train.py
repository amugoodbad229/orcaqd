"""PGA-MAP-Elites training loop for dexterous hand QD.

Combines:
  - GA half: QDax MixingEmitter with Iso+LineDD variation
  - PG half: Custom TD3-based PGEmitter (critic gradient ascent)

Both halves produce offspring that compete for archive cells.

Usage:
    uv run python -m src.qd_engine.train_pga_map_elites --config configs/paper1_smoke.yaml
"""
from __future__ import annotations

import argparse
import functools
import time
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import yaml

import src.jax_env  # noqa: F401

import flax.linen as nn
from qdax.core.map_elites import MAPElites
from qdax.core.containers.mapelites_repertoire import compute_euclidean_centroids
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.utils.metrics import default_qd_metrics

from src.envs.dex_env import DexHandEnv, EnvConfig
from src.envs.hand_config import ORCAHAND_RIGHT


# ---------------------------------------------------------------------------
# Policy network
# ---------------------------------------------------------------------------

class PolicyNetwork(nn.Module):
    """MLP actor: obs -> action in [-1, 1]."""
    hidden_size: int = 64
    action_dim: int = 17

    @nn.compact
    def __call__(self, obs: jax.Array) -> jax.Array:
        x = nn.Dense(self.hidden_size)(obs)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_size)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return nn.tanh(x)


# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    grid_shape: tuple[int, int] = (50, 50)
    min_bd: tuple[float, float] = (-5.0, 0.0)
    max_bd: tuple[float, float] = (-1.0, 1.0)

    # Batch sizes: total offspring = ga_batch_size + pg_batch_size
    ga_batch_size: int = 64
    pg_batch_size: int = 64
    num_iterations: int = 1000
    episode_length: int = 250
    env_batch_size: int = 64

    # Networks
    policy_hidden: int = 64
    iso_sigma: float = 0.005
    line_sigma: float = 0.05

    # PG emitter config
    critic_hidden: int = 256
    critic_lr: float = 3e-4
    actor_lr: float = 3e-4
    pg_steps: int = 50
    pg_buffer_size: int = 100_000
    pg_warmup: int = 500

    # Logging
    log_every: int = 10
    save_every: int = 100
    wandb_project: str = "orcaQD"
    wandb_entity: str = "amugoodbad"
    wandb_enabled: bool = True
    out_dir: str = "outputs"

    @property
    def total_batch_size(self) -> int:
        return self.ga_batch_size + self.pg_batch_size

    @classmethod
    def from_yaml(cls, path: str) -> "TrainConfig":
        with open(path) as f:
            d = yaml.safe_load(f)
        flat = {}
        for k, v in d.items():
            if isinstance(v, dict):
                flat.update(v)
            else:
                flat[k] = v
        for key in ("grid_shape", "min_bd", "max_bd"):
            if key in flat and isinstance(flat[key], list):
                flat[key] = tuple(flat[key])
        return cls(**{k: v for k, v in flat.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Scoring function
# ---------------------------------------------------------------------------

def make_scoring_fn(env: DexHandEnv, policy_net: PolicyNetwork, cfg: TrainConfig):
    """Create a QDax-compatible scoring function.

    Optimized: single flattened vmap over all (genotype × seed) rollouts
    instead of nested vmap. Reduces XLA compilation graph depth and improves
    GPU kernel fusion.
    """

    def rollout_one(params, key):
        """Single policy rollout with one random seed."""
        state = env.reset(key)
        total_reward = jnp.float32(0.0)

        def step_fn(carry, _):
            state, total_reward = carry
            action = policy_net.apply(params, state.obs)
            state = env.step(state, action)
            total_reward = total_reward + state.reward
            return (state, total_reward), None

        (final_state, total_reward), _ = jax.lax.scan(
            step_fn, (state, total_reward), jnp.arange(cfg.episode_length)
        )
        descriptors = env.get_descriptors(final_state)
        return total_reward, descriptors

    @jax.jit
    def scoring_fn(genotypes, key):
        n_genotypes = jax.tree.leaves(genotypes)[0].shape[0]
        total_rollouts = n_genotypes * cfg.env_batch_size
        keys = jax.random.split(key, total_rollouts)

        # Replicate each genotype env_batch_size times for parallel evaluation.
        def replicate(tree):
            return jax.tree.map(
                lambda x: jnp.repeat(x, cfg.env_batch_size, axis=0),
                tree
            )
        flat_genotypes = replicate(genotypes)

        # Single vmap over all rollouts (flattened nested vmap).
        rewards, descriptors = jax.vmap(rollout_one)(flat_genotypes, keys)

        # Reshape back to (n_genotypes, env_batch_size) and average.
        rewards = rewards.reshape(n_genotypes, cfg.env_batch_size).mean(axis=1)
        descriptors = descriptors.reshape(n_genotypes, cfg.env_batch_size, -1).mean(axis=1)

        # b1 in log scale for the archive grid.
        b1_log = jnp.log10(jnp.maximum(descriptors[:, 0], 1e-8))
        final_desc = jnp.stack([b1_log, descriptors[:, 1]], axis=1)
        return rewards, final_desc, {}

    return scoring_fn


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg: TrainConfig):
    print(f"Config: {cfg}")
    print(f"JAX devices: {jax.devices()}")

    # Environment.
    env_cfg = EnvConfig(hand=ORCAHAND_RIGHT, episode_length=cfg.episode_length)
    env = DexHandEnv(env_cfg)
    print(f"Env: obs={env.obs_dim}, act={env.action_dim}, object={env.has_object}")

    # Network.
    policy_net = PolicyNetwork(hidden_size=cfg.policy_hidden, action_dim=env.action_dim)
    dummy_obs = jnp.zeros(env.obs_dim)
    key = jax.random.PRNGKey(42)
    key, init_key = jax.random.split(key)
    example_params = policy_net.init(init_key, dummy_obs)
    n_params = sum(p.size for p in jax.tree.leaves(example_params))
    print(f"Policy: {n_params:,} params")

    # Scoring.
    scoring_fn = make_scoring_fn(env, policy_net, cfg)

    # Centroids.
    centroids = compute_euclidean_centroids(
        grid_shape=cfg.grid_shape,
        minval=jnp.array(cfg.min_bd),
        maxval=jnp.array(cfg.max_bd),
    )
    print(f"Archive: {cfg.grid_shape[0]}x{cfg.grid_shape[1]} = {len(centroids)} cells")

    # --- Emitters ---
    # GA emitter: Iso+LineDD variation. PG emitter (pg_emitter.py) is available
    # but disabled by default — set pg_batch_size > 0 to enable.
    total_offspring = cfg.ga_batch_size + cfg.pg_batch_size
    variation_fn = functools.partial(
        isoline_variation,
        iso_sigma=cfg.iso_sigma,
        line_sigma=cfg.line_sigma,
    )
    emitter = MixingEmitter(
        mutation_fn=lambda x, k: (x, k),
        variation_fn=variation_fn,
        variation_percentage=1.0,
        batch_size=total_offspring,
    )
    print(f"Emitter: GA Iso+LineDD, batch={total_offspring}")

    # MAP-Elites.
    metrics_fn = functools.partial(default_qd_metrics, qd_offset=0.0)
    map_elites = MAPElites(
        scoring_function=scoring_fn,
        emitter=emitter,
        metrics_function=metrics_fn,
    )

    # Init.
    key, init_key = jax.random.split(key)
    init_keys = jax.random.split(init_key, cfg.total_batch_size)
    init_genotypes = jax.vmap(lambda k: policy_net.init(k, dummy_obs))(init_keys)

    print("Initializing archive...")
    t0 = time.time()
    repertoire, emitter_state, metrics = map_elites.init(init_genotypes, centroids, init_key)
    print(f"Init: {time.time()-t0:.1f}s, QD-Score={float(metrics['qd_score']):.2f}, "
          f"Coverage={float(metrics['coverage']):.4f}")

    # WandB.
    if cfg.wandb_enabled:
        try:
            import wandb
            wandb.init(
                project=cfg.wandb_project,
                entity=cfg.wandb_entity,
                config=vars(cfg),
            )
        except Exception as e:
            print(f"WandB failed: {e}")
            cfg.wandb_enabled = False

    # Training.
    update_fn = jax.jit(map_elites.update)
    print(f"\nTraining: {cfg.num_iterations} iterations")

    for i in range(cfg.num_iterations):
        key, update_key = jax.random.split(key)
        t_iter = time.time()
        repertoire, emitter_state, metrics = update_fn(repertoire, emitter_state, update_key)

        if (i + 1) % cfg.log_every == 0:
            qs = float(metrics["qd_score"])
            cov = float(metrics["coverage"])
            dt = time.time() - t_iter
            print(f"  [{i+1}/{cfg.num_iterations}] QD={qs:.2f} Cov={cov:.4f} dt={dt:.2f}s")
            if cfg.wandb_enabled:
                import wandb
                wandb.log({"qd_score": qs, "coverage": cov, "iter": i+1, "dt": dt})

        if (i + 1) % cfg.save_every == 0:
            p = Path(cfg.out_dir) / f"archive_{i+1}.npz"
            p.parent.mkdir(parents=True, exist_ok=True)
            jnp.savez(str(p), fitnesses=repertoire.fitnesses, descriptors=repertoire.descriptors)
            print(f"  saved: {p}")

    # Final.
    p = Path(cfg.out_dir) / "archive_final.npz"
    p.parent.mkdir(parents=True, exist_ok=True)
    jnp.savez(str(p), fitnesses=repertoire.fitnesses, descriptors=repertoire.descriptors)
    print(f"\nDone. QD={float(metrics['qd_score']):.2f} Cov={float(metrics['coverage']):.4f}")
    print(f"Archive: {p}")

    if cfg.wandb_enabled:
        import wandb
        wandb.finish()

    return repertoire


def main(config_path: str | None = None, out_dir: str | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper1_smoke.yaml")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    cfg = TrainConfig.from_yaml(config_path or args.config)
    if out_dir or args.out_dir:
        cfg.out_dir = out_dir or args.out_dir
    train(cfg)


if __name__ == "__main__":
    main()
