"""Unit tests: hand-computed cases and the structural invariants of THEORY.md."""

import math

import numpy as np
import pytest

from contamsens import (
    deflation,
    gamma_star,
    identified_interval,
    is_robust,
    item_floor,
    margin,
    max_bias,
    max_bias_simple,
)
from contamsens.csm import CSM

RNG = np.random.default_rng(42)


# ---------- deflation and floors: hand-computed cases ----------

def test_deflation_hand_computed():
    y = np.array([0.0, 0.1, 0.2, 0.5, 0.9, 1.0])
    lam = 0.2
    # y <= lam: d = y ; y > lam: d = lam(1-y)/(1-lam) = 0.25*(1-y)
    expected = np.array([0.0, 0.1, 0.2, 0.25 * 0.5, 0.25 * 0.1, 0.0])
    np.testing.assert_allclose(deflation(y, lam), expected, atol=1e-12)


def test_deflation_peak_at_lambda():
    """d is maximized at y = lam with value lam (Corollary 1)."""
    y = np.linspace(0, 1, 10001)
    for lam in (0.1, 0.3, 0.7):
        d = deflation(y, lam)
        assert d.max() == pytest.approx(lam, abs=1e-3)
        assert y[d.argmax()] == pytest.approx(lam, abs=1e-3)


def test_ceiling_effect():
    """Observed y = 1 forces y* = 1 for lam < 1 (THEORY.md Remark)."""
    assert item_floor(np.array([1.0]), 0.5)[0] == pytest.approx(1.0)
    assert deflation(np.array([1.0]), 0.99)[0] == pytest.approx(0.0, abs=1e-9)


def test_deflation_extremes():
    y = np.array([0.3, 0.8])
    np.testing.assert_allclose(deflation(y, 0.0), [0.0, 0.0])
    np.testing.assert_allclose(deflation(y, 1.0), y)  # full memorization


# ---------- Proposition 1: the sharp interval ----------

def test_max_bias_hand_computed():
    """n=4, lam=0.5, pi=0.5 -> k=2. d = [y if y<=.5 else (1-y)]"""
    y = np.array([0.2, 0.4, 0.6, 1.0])
    # d = [0.2, 0.4, 0.5*(0.4)/0.5=0.4, 0.0] -> top-2 = 0.4+0.4 -> B = 0.2
    assert max_bias(y, lam=0.5, pi=0.5) == pytest.approx(0.2)


def test_interval_contains_mu_and_is_ordered():
    y = RNG.uniform(0, 1, 500)
    lo, hi = identified_interval(y, lam=0.3, pi=0.2)
    assert lo <= hi
    assert hi == pytest.approx(y.mean())
    assert lo >= 0.0


def test_reduction_at_zero():
    """Corollary 2: lam=0 or pi=0 collapses the set to {mu}."""
    y = RNG.uniform(0, 1, 200)
    for lam, pi in ((0.0, 0.3), (0.3, 0.0)):
        lo, hi = identified_interval(y, lam=lam, pi=pi)
        assert lo == pytest.approx(hi)


def test_simple_bound_dominates_sharp():
    """B(lam, pi) <= pi*lam + rounding slack (Corollary 1)."""
    for _ in range(20):
        y = RNG.uniform(0, 1, 137)
        lam, pi = RNG.uniform(0.05, 0.95), RNG.uniform(0.05, 0.95)
        slack = lam / y.size  # ceil rounding admits at most one extra item
        assert max_bias(y, lam, pi) <= max_bias_simple(lam, pi) + slack + 1e-12


def test_monotone_in_lambda_and_pi():
    y = RNG.uniform(0, 1, 300)
    biases_lam = [max_bias(y, lam, 0.2) for lam in np.linspace(0, 1, 21)]
    assert all(b2 >= b1 - 1e-12 for b1, b2 in zip(biases_lam, biases_lam[1:]))
    biases_pi = [max_bias(y, 0.3, pi) for pi in np.linspace(0, 1, 21)]
    assert all(b2 >= b1 - 1e-12 for b1, b2 in zip(biases_pi, biases_pi[1:]))


def test_sharpness_attained():
    """The lower endpoint is attained by an explicit consistent (c, y*)."""
    y = RNG.uniform(0, 1, 100)
    lam, pi = 0.4, 0.15
    k = CSM(lam, pi).budget(y.size)
    d = deflation(y, lam)
    contaminated = np.argsort(d)[-k:]
    y_star = y.copy()
    y_star[contaminated] = item_floor(y, lam)[contaminated]
    # (i) consistency with A1 on the contaminated set
    lift = y[contaminated] - y_star[contaminated]
    cap = lam * (1 - y_star[contaminated])
    assert np.all(lift <= cap + 1e-9) and np.all(lift >= -1e-12)
    # (ii) attains the bound
    lo, _ = identified_interval(y, lam, pi)
    assert y_star.mean() == pytest.approx(lo, abs=1e-12)


# ---------- Gamma* ----------

def test_gamma_star_simple_closed_form():
    """Simple regime: Gamma* = Delta / pi (contamination frontier)."""
    y_a = np.full(100, 0.62)
    y_b = np.full(100, 0.60)
    g = gamma_star(y_a, y_b, pi=0.1, simple=True)
    assert g == pytest.approx(0.2)  # 0.02 / 0.1


def test_gamma_star_bisection_consistency():
    """B_A(Gamma*, pi) == Delta at the returned root."""
    y_a = RNG.uniform(0.3, 1.0, 400)
    y_b = np.clip(y_a - 0.05 + RNG.normal(0, 0.05, 400), 0, 1)
    pi = 0.2
    g = gamma_star(y_a, y_b, pi)
    if math.isfinite(g):
        delta = margin(y_a, y_b)
        assert max_bias(y_a, g, pi) == pytest.approx(delta, abs=1e-4)
        # robust just below the root, overturnable at the root
        assert is_robust(y_a, y_b, g - 1e-4, pi)
        assert not is_robust(y_a, y_b, g + 1e-4, pi)


def test_gamma_star_edge_cases():
    y_hi = np.full(50, 0.9)
    y_lo = np.full(50, 0.1)
    # Huge margin, tiny budget: robust to full memorization.
    assert math.isinf(gamma_star(y_hi, y_lo, pi=0.05))
    # Claim not even true at face value.
    assert gamma_star(y_lo, y_hi, pi=0.5) == 0.0


def test_gamma_star_monotone_in_pi():
    """More budget -> easier to overturn -> Gamma* nonincreasing in pi."""
    y_a = RNG.uniform(0.2, 0.9, 300)
    y_b = np.clip(y_a - 0.04, 0, 1)
    gs = [gamma_star(y_a, y_b, pi) for pi in (0.05, 0.1, 0.2, 0.4, 0.8)]
    finite = [g for g in gs if math.isfinite(g)]
    assert all(g2 <= g1 + 1e-9 for g1, g2 in zip(finite, finite[1:]))
