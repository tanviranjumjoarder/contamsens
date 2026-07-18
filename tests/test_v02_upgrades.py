"""Tests for the v0.2 red-team upgrades (THEORY.md SS9)."""

import math

import numpy as np
import pytest

from contamsens import (
    gamma_star,
    identified_interval,
    identified_interval_twosided,
    max_bias,
    max_bias_grouped,
    max_inflation,
)

RNG = np.random.default_rng(42)


# ---------- U1: two-sided CSM ----------

def test_twosided_reduces_to_onesided_at_zero():
    y = RNG.uniform(0, 1, 300)
    lo1, hi1 = identified_interval(y, lam=0.3, pi=0.2)
    lo2, hi2 = identified_interval_twosided(y, lam_plus=0.3, lam_minus=0.0, pi=0.2)
    assert lo2 == pytest.approx(lo1)
    assert hi2 == pytest.approx(hi1)


def test_twosided_upper_moves_only_with_lam_minus():
    y = RNG.uniform(0.2, 0.8, 300)
    lo, hi = identified_interval_twosided(y, lam_plus=0.3, lam_minus=0.2, pi=0.2)
    _, hi_one = identified_interval(y, lam=0.3, pi=0.2)
    assert hi > hi_one
    assert hi <= 1.0 and lo >= 0.0


def test_inflation_hand_computed():
    """u_i = min(1 - y, y * lam-/(1 - lam-)); lam-=0.5 -> u = min(1-y, y)."""
    y = np.array([0.2, 0.5, 0.9, 1.0])
    # n=4, pi=0.5 -> k=2; u = [0.2, 0.5, 0.1, 0.0] -> top-2 = 0.5+0.2 -> 0.175
    assert max_inflation(y, lam_minus=0.5, pi=0.5) == pytest.approx(0.175)


def test_gamma_star_twosided_is_smaller():
    """Allowing B-deflation makes claims easier to overturn: Gamma* decreases."""
    y_a = RNG.uniform(0.3, 0.95, 400)
    y_b = np.clip(y_a - 0.05, 0, 1)
    g0 = gamma_star(y_a, y_b, pi=0.2, rho=0.0)
    g1 = gamma_star(y_a, y_b, pi=0.2, rho=0.5)
    if math.isfinite(g0):
        assert g1 < g0
    # simple two-sided frontier: Delta / (pi * (1 + rho))
    g_simple = gamma_star(np.full(10, 0.62), np.full(10, 0.60), pi=0.1,
                          simple=True, rho=1.0)
    assert g_simple == pytest.approx(0.1)  # 0.02 / (0.1 * 2)


# ---------- U2: group-level budgets ----------

def test_grouped_never_exceeds_itemlevel():
    """Group constraints can only weaken the adversary."""
    y = RNG.uniform(0, 1, 400)
    groups = RNG.integers(0, 20, 400)
    for lam, pi in ((0.2, 0.1), (0.4, 0.3), (0.8, 0.5)):
        assert max_bias_grouped(y, lam, pi, groups) <= max_bias(y, lam, pi) + 1e-12


def test_grouped_equals_itemlevel_when_singleton_groups():
    y = RNG.uniform(0, 1, 200)
    groups = np.arange(200)  # every item its own group
    assert max_bias_grouped(y, 0.3, 0.2, groups) == pytest.approx(
        max_bias(y, 0.3, 0.2), abs=1e-9
    )


def test_grouped_hand_computed():
    """Two groups; budget forces the denser group (fractional greedy)."""
    # group 0: items d = [0.3, 0.3] (density 0.3); group 1: d = [0.5, 0.1] (0.3)
    # lam=1 -> d = y. budget pi=0.5 -> k=2. densities equal; greedy takes one
    # group fully: total deflation 0.6 either way -> B = 0.6/4 = 0.15
    y = np.array([0.3, 0.3, 0.5, 0.1])
    groups = np.array([0, 0, 1, 1])
    assert max_bias_grouped(y, lam=1.0, pi=0.5, groups=groups) == pytest.approx(0.15)


def test_grouped_validates_shape():
    with pytest.raises(ValueError):
        max_bias_grouped(np.array([0.5, 0.5]), 0.3, 0.5, np.array([0]))


# ---------- U4: spillover channel ----------

def test_spillover_widens_both_ends():
    y = RNG.uniform(0.2, 0.8, 200)
    lo0, hi0 = identified_interval_twosided(y, 0.3, 0.1, 0.2)
    lo1, hi1 = identified_interval_twosided(y, 0.3, 0.1, 0.2, spillover=0.02)
    assert lo1 == pytest.approx(lo0 - 0.02)
    assert hi1 == pytest.approx(hi0 + 0.02)


def test_spillover_zero_is_default_and_negative_rejected():
    y = RNG.uniform(0, 1, 100)
    assert identified_interval_twosided(y, 0.3, 0.0, 0.2) == \
        identified_interval_twosided(y, 0.3, 0.0, 0.2, spillover=0.0)
    with pytest.raises(ValueError):
        identified_interval_twosided(y, 0.3, 0.0, 0.2, spillover=-0.1)


# ---------- U3: stratified budgets ----------

def test_stratified_all_eligible_equals_itemlevel():
    from contamsens import max_bias_stratified

    y = RNG.uniform(0, 1, 300)
    full = np.ones(300, dtype=bool)
    assert max_bias_stratified(y, 0.3, 0.2, full) == pytest.approx(
        max_bias(y, 0.3, 0.2)
    )


def test_stratified_none_eligible_is_zero():
    from contamsens import max_bias_stratified

    y = RNG.uniform(0, 1, 100)
    assert max_bias_stratified(y, 0.5, 0.5, np.zeros(100, dtype=bool)) == 0.0


def test_stratified_never_exceeds_itemlevel():
    from contamsens import max_bias_stratified

    y = RNG.uniform(0, 1, 400)
    for frac in (0.1, 0.5, 0.9):
        elig = RNG.uniform(size=400) < frac
        b_s = max_bias_stratified(y, 0.4, 0.2, elig)
        assert b_s <= max_bias(y, 0.4, 0.2) + 1e-12
