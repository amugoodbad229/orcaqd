"""Tests for the OrcaHand MJX environment.

Run with:
    uv run pytest tests/test_env.py -v
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from src.envs.orcahand_mjx_env import OrcaHandEnv, EnvConfig


@pytest.fixture(scope="module")
def env():
    return OrcaHandEnv(EnvConfig())


def test_env_creates(env):
    assert env.nq == 17
    assert env.nv == 17
    assert env.nu == 17
    assert env.obs_dim == 34  # 17 qpos + 17 qvel (no object yet)
    assert env.action_dim == 17


def test_reset(env):
    key = jax.random.PRNGKey(42)
    state = env.reset(key)
    assert state.obs.shape == (env.obs_dim,)
    assert state.done == False
    assert state.step_count == 0


def test_step(env):
    key = jax.random.PRNGKey(0)
    state = env.reset(key)
    action = jnp.zeros(env.action_dim)
    state = env.step(state, action)
    assert state.obs.shape == (env.obs_dim,)
    assert state.step_count == 1
    assert state.reward.shape == ()


def test_episode_terminates(env):
    key = jax.random.PRNGKey(1)
    state = env.reset(key)
    action = jnp.zeros(env.action_dim)
    for _ in range(env.cfg.episode_length):
        state = env.step(state, action)
    assert state.done == True
    assert state.step_count == env.cfg.episode_length
