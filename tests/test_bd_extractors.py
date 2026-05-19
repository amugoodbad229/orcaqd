"""Tests for behavior descriptor extractors.

Run with:
    uv run pytest tests/test_bd_extractors.py -v
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from src.envs.bd_extractors import (
    compute_contact_dispersion,
    compute_descriptors,
    compute_thumb_force_ratio,
    accumulate_descriptors_over_window,
)


class TestContactDispersion:
    def test_no_contacts_returns_zero(self):
        pos = jnp.zeros((10, 3))
        active = jnp.zeros(10, dtype=bool)
        b1 = compute_contact_dispersion(pos, active)
        assert float(b1) == 0.0

    def test_single_contact_returns_zero(self):
        pos = jnp.array([[0.1, 0.2, 0.3]] + [[0.0, 0.0, 0.0]] * 9)
        active = jnp.array([True] + [False] * 9)
        b1 = compute_contact_dispersion(pos, active)
        assert float(b1) == 0.0

    def test_two_contacts_positive(self):
        pos = jnp.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]] + [[0.0, 0.0, 0.0]] * 8)
        active = jnp.array([True, True] + [False] * 8)
        b1 = compute_contact_dispersion(pos, active)
        # Two points at (0,0,0) and (0.1,0,0): mean=(0.05,0,0),
        # variance = 0.05^2 + 0 + 0 = 0.0025 per point, trace = 0.0025
        assert float(b1) == pytest.approx(0.0025, abs=1e-6)

    def test_jit_compiles(self):
        pos = jnp.zeros((10, 3))
        active = jnp.zeros(10, dtype=bool)
        jit_fn = jax.jit(compute_contact_dispersion)
        b1 = jit_fn(pos, active)
        assert b1.shape == ()


class TestThumbForceRatio:
    def test_no_contacts_returns_half(self):
        force = jnp.zeros(10)
        active = jnp.zeros(10, dtype=bool)
        is_thumb = jnp.zeros(10, dtype=bool)
        b2 = compute_thumb_force_ratio(force, active, is_thumb)
        assert float(b2) == pytest.approx(0.5)

    def test_all_thumb_returns_one(self):
        force = jnp.array([1.0, 2.0] + [0.0] * 8)
        active = jnp.array([True, True] + [False] * 8)
        is_thumb = jnp.array([True, True] + [False] * 8)
        b2 = compute_thumb_force_ratio(force, active, is_thumb)
        assert float(b2) == pytest.approx(1.0)

    def test_half_thumb(self):
        force = jnp.array([1.0, 1.0] + [0.0] * 8)
        active = jnp.array([True, True] + [False] * 8)
        is_thumb = jnp.array([True, False] + [False] * 8)
        b2 = compute_thumb_force_ratio(force, active, is_thumb)
        assert float(b2) == pytest.approx(0.5)

    def test_jit_compiles(self):
        force = jnp.zeros(10)
        active = jnp.zeros(10, dtype=bool)
        is_thumb = jnp.zeros(10, dtype=bool)
        jit_fn = jax.jit(compute_thumb_force_ratio)
        b2 = jit_fn(force, active, is_thumb)
        assert b2.shape == ()


class TestAccumulate:
    def test_window_of_zeros(self):
        T, N = 5, 10
        pos = jnp.zeros((T, N, 3))
        force = jnp.zeros((T, N))
        active = jnp.zeros((T, N), dtype=bool)
        is_thumb = jnp.zeros(N, dtype=bool)
        b1, b2 = accumulate_descriptors_over_window(pos, force, active, is_thumb)
        assert float(b1) == 0.0
        assert float(b2) == pytest.approx(0.5)

    def test_jit_compiles(self):
        T, N = 5, 10
        pos = jnp.zeros((T, N, 3))
        force = jnp.zeros((T, N))
        active = jnp.zeros((T, N), dtype=bool)
        is_thumb = jnp.zeros(N, dtype=bool)
        jit_fn = jax.jit(accumulate_descriptors_over_window)
        b1, b2 = jit_fn(pos, force, active, is_thumb)
        assert b1.shape == ()
        assert b2.shape == ()
