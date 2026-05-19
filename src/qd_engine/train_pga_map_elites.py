"""PGA-MAP-Elites training loop for dexterous hand QD.

Wires the DexHandEnv into QDax's PGAMEEmitter to discover a diverse archive
of grasping policies. Hand-agnostic: works with any HandConfig.

Usage:
    # Local smoke run (small batch, few iterations):
    uv run python -m src.qd_engine.train_pga_map_elites --config configs/paper1_smoke.yaml

    # Full run (use Modal for GPU):
    modal run --detach -m src.modal_app::train --config configs/paper1_main.yaml
"""
from __future__ import annotations

import argparse
import functools
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import yaml

# Apply XLA flags before any JAX computation.
import src.jax_env  # noqa: F401

import flax.linen as nn
import optax
from qdax.core.map_elites import MAPElites
from qdax.core.containers.mapelites_repertoire import (
    MapElitesRepertoire,
    compute_euclidean_centroids,
)
from qdax.core.emitters.pga_me_emitter import PGAMEEmitter, PGAMEConfig
from qdax.utils.metrics import default_qd_metrics

from src.envs.orcahand_mjx_env import DexHandEnv, EnvConfig
from src.envs.hand_config import ORCAHAND_RIGHT


# ---------------------------------------------------------------------------
# Policy and Critic networks
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


class CriticNetwork(nn.Module):
    """MLP critic: (obs, action) -> Q-value."""
    hidden_size: int = 256

    @nn.compact
    def __call__(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        x = jnp.concatenate([obs, action], axis=-1)
        x = nn.Dense(self.hidden_size)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_size)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x.squeeze(-1)


# ---------------------------------------------------------------------------
# Training config (loaded from YAML)
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """Training hyperparameters."""
    # Archive
    grid_shape: tuple[int, int] = (50, 50)
    min_bd: tuple[float, float] = (-5.0, 0.0)   # log10(b1) min, b2 min
    max_bd: tuple[float, float] = (-1.0, 1.0)   # log10(b1) max, b2 max

    # QD
    batch_size: int = 128          # offspring per iteration
    num_iterations: int = 1000
    pg_steps: int = 100            # PG improvement steps per offspring

    # Environment
    episode_length: int = 250
    env_batch_size: int = 64       # parallel envs for scoring one policy

    # Networks
    policy_hidden: int = 64
    critic_hidden: int = 256
    policy_lr: float = 3e-4
    critic_lr: float = 3e-4

    # Logging
    log_every: int = 10
    save_every: int = 100
    wandb_project: str = "orcaqd"
    wandb_enabled: bool = True

    # Output
    out_dir: str = "outputs"

    @classmethod
    def from_yaml(cls, path: str) -> "TrainConfig":
        with open(path) as f:
            d = yaml.safe_load(f)
        # Flatten nested dicts if present.
        flat = {}
        for k, v in d.items():
            if isinstance(v, dict):
                flat.update(v)
            else:
                flat[k] = v
        # Convert lists to tuples for dataclass fields.
        for key in ("grid_shape", "min_bd", "max_bd"):
            if key in flat and isinstance(flat[key], list):
                flat[key] = tuple(flat[key])
        return cls(**{k: v for k, v in flat.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Scoring function
# ---------------------------------------------------------------------------

def make_scoring_fn(env: DexHandEnv, policy_net: PolicyNetwork, cfg: TrainConfig):
    """Create a QDax-compatible scoring function.

    scoring_fn(genotypes, key) -> (fitnesses, descriptors, extra_scores, key)

    Where genotypes is a batch of policy parameter PyTrees.
    """

    def score_one_policy(params, key):
        """Roll out one policy across env_batch_size DR seeds."""
        keys = jax.random.split(key, cfg.env_batch_size)

        def rollout_one_seed(seed_key):
            state = env.reset(seed_key)
            total_reward = jnp.float32(0.0)

            def step_fn(carry, _):
                state, total_reward = carry
                obs = state.obs
                action = policy_net.apply(params, obs)
                state = env.step(state, action)
                total_reward = total_reward + state.reward
                return (state, total_reward), None

            (final_state, total_reward), _ = jax.lax.scan(
                step_fn, (state, total_reward), jnp.arange(cfg.episode_length)
            )

            descriptors = env.get_descriptors(final_state)
            return total_reward, descriptors

        # Vmap over DR seeds.
        rewards, descriptors = jax.vmap(rollout_one_seed)(keys)

        # Average fitness and descriptors across seeds.
        fitness = jnp.mean(rewards)
        mean_desc = jnp.mean(descriptors, axis=0)

        # Transform b1 to log scale for the archive grid.
        b1_log = jnp.log10(jnp.maximum(mean_desc[0], 1e-8))
        final_desc = jnp.array([b1_log, mean_desc[1]])

        return fitness, final_desc

    @jax.jit
    def scoring_fn(genotypes, key):
        """Score a batch of policies."""
        keys = jax.random.split(key, jax.tree.leaves(genotypes)[0].shape[0] + 1)
        scoring_keys = keys[:-1]
        next_key = keys[-1]

        fitnesses, descriptors = jax.vmap(score_one_policy)(genotypes, scoring_keys)

        return fitnesses, descriptors, {}, next_key

    return scoring_fn


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(cfg: TrainConfig) -> MapElitesRepertoire:
    """Run PGA-MAP-Elites and return the final archive."""
    print(f"Training config: {cfg}")
    print(f"JAX devices: {jax.devices()}")

    # --- Environment ---
    env_cfg = EnvConfig(
        hand=ORCAHAND_RIGHT,
        episode_length=cfg.episode_length,
    )
    env = DexHandEnv(env_cfg)
    print(f"Env: obs_dim={env.obs_dim}, action_dim={env.action_dim}, has_object={env.has_object}")

    # --- Networks ---
    policy_net = PolicyNetwork(hidden_size=cfg.policy_hidden, action_dim=env.action_dim)
    critic_net = CriticNetwork(hidden_size=cfg.critic_hidden)

    # Init params for shape inference.
    dummy_obs = jnp.zeros(env.obs_dim)
    dummy_action = jnp.zeros(env.action_dim)
    key = jax.random.PRNGKey(0)
    key, p_key, c_key = jax.random.split(key, 3)
    policy_params = policy_net.init(p_key, dummy_obs)
    critic_params = critic_net.init(c_key, dummy_obs, dummy_action)

    print(f"Policy params: {sum(p.size for p in jax.tree.leaves(policy_params)):,}")
    print(f"Critic params: {sum(p.size for p in jax.tree.leaves(critic_params)):,}")

    # --- Scoring function ---
    scoring_fn = make_scoring_fn(env, policy_net, cfg)

    # --- Archive centroids ---
    centroids = compute_euclidean_centroids(
        grid_shape=cfg.grid_shape,
        minval=jnp.array(cfg.min_bd),
        maxval=jnp.array(cfg.max_bd),
    )
    print(f"Archive: {cfg.grid_shape[0]}x{cfg.grid_shape[1]} = {len(centroids)} cells")

    # --- Emitter ---
    pga_config = PGAMEConfig(
        num_critic_training_steps=cfg.pg_steps,
        num_pg_training_steps=cfg.pg_steps,
    )

    emitter = PGAMEEmitter(
        config=pga_config,
        policy_network=policy_net,
        env_batch_size=cfg.batch_size,
        scoring_fn=scoring_fn,
    )

    # --- MAP-Elites ---
    metrics_fn = functools.partial(default_qd_metrics, qd_offset=0.0)

    map_elites = MAPElites(
        scoring_function=scoring_fn,
        emitter=emitter,
        metrics_function=metrics_fn,
    )

    # --- Init ---
    key, init_key = jax.random.split(key)
    # Create initial batch of random policies.
    init_keys = jax.random.split(init_key, cfg.batch_size)
    init_genotypes = jax.vmap(lambda k: policy_net.init(k, dummy_obs))(init_keys)

    print("Initializing archive...")
    t0 = time.time()
    repertoire, emitter_state, init_metrics = map_elites.init(
        init_genotypes, centroids, init_key
    )
    print(f"Init done in {time.time() - t0:.1f}s")
    print(f"  QD-Score: {init_metrics['qd_score']:.2f}")
    print(f"  Coverage: {init_metrics['coverage']:.4f}")

    # --- WandB ---
    if cfg.wandb_enabled:
        try:
            import wandb
            wandb.init(project=cfg.wandb_project, config=vars(cfg))
        except Exception as e:
            print(f"WandB init failed: {e}. Continuing without logging.")
            cfg.wandb_enabled = False

    # --- Training loop ---
    update_fn = jax.jit(map_elites.update)

    print(f"\nStarting training: {cfg.num_iterations} iterations")
    for i in range(cfg.num_iterations):
        key, update_key = jax.random.split(key)
        t_iter = time.time()

        repertoire, emitter_state, metrics = update_fn(
            repertoire, emitter_state, update_key
        )

        if (i + 1) % cfg.log_every == 0:
            qd_score = float(metrics["qd_score"])
            coverage = float(metrics["coverage"])
            dt = time.time() - t_iter
            print(f"  iter {i+1}/{cfg.num_iterations}: "
                  f"QD-Score={qd_score:.2f}, Coverage={coverage:.4f}, "
                  f"dt={dt:.2f}s")

            if cfg.wandb_enabled:
                import wandb
                wandb.log({
                    "qd_score": qd_score,
                    "coverage": coverage,
                    "iteration": i + 1,
                    "iter_time_s": dt,
                })

        if (i + 1) % cfg.save_every == 0:
            out_path = Path(cfg.out_dir) / f"archive_iter_{i+1}.npz"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            # Save fitnesses and descriptors (lightweight).
            jnp.savez(
                str(out_path),
                fitnesses=repertoire.fitnesses,
                descriptors=repertoire.descriptors,
            )
            print(f"  saved checkpoint: {out_path}")

    # --- Final save ---
    out_path = Path(cfg.out_dir) / "archive_final.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    jnp.savez(
        str(out_path),
        fitnesses=repertoire.fitnesses,
        descriptors=repertoire.descriptors,
    )
    print(f"\nTraining complete. Final archive: {out_path}")
    print(f"  QD-Score: {float(metrics['qd_score']):.2f}")
    print(f"  Coverage: {float(metrics['coverage']):.4f}")

    if cfg.wandb_enabled:
        import wandb
        wandb.finish()

    return repertoire


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(config_path: str | None = None, out_dir: str | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/paper1_smoke.yaml")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    path = config_path or args.config
    cfg = TrainConfig.from_yaml(path)

    if out_dir or args.out_dir:
        cfg.out_dir = out_dir or args.out_dir

    train(cfg)


if __name__ == "__main__":
    main()
