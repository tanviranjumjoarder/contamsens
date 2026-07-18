"""Tests for Phase-1 theory: the selection-bounded bridge model (Prop 3)."""

import numpy as np
import pytest

from contamsens import deflation, max_bias, max_bias_selection

RNG = np.random.default_rng(42)


def test_random_endpoint():
    """gamma_sel = 1 -> B = pi * mean(d) (random contamination)."""
    y = RNG.uniform(0, 1, 500)
    lam, pi = 0.4, 0.2
    b = max_bias_selection(y, lam, pi, gamma_sel=1.0)
    assert b == pytest.approx(pi * deflation(y, lam).mean(), rel=1e-9)


def test_adversarial_endpoint():
    """gamma_sel -> inf recovers the v0.1 top-k bound."""
    y = RNG.uniform(0, 1, 500)
    lam, pi = 0.4, 0.2
    b_inf = max_bias_selection(y, lam, pi, gamma_sel=np.inf)
    assert b_inf == pytest.approx(max_bias(y, lam, pi), rel=1e-12)
    b_big = max_bias_selection(y, lam, pi, gamma_sel=1e6)
    assert b_big == pytest.approx(max_bias(y, lam, pi), rel=1e-2)


def test_monotone_in_gamma_sel_and_bracketed():
    """B is nondecreasing in gamma_sel and lives between the two endpoints."""
    y = RNG.uniform(0, 1, 400)
    lam, pi = 0.3, 0.15
    lo_end = pi * deflation(y, lam).mean()
    hi_end = max_bias(y, lam, pi)
    prev = 0.0
    for gs in (1.0, 1.5, 2.0, 4.0, 10.0, 100.0):
        b = max_bias_selection(y, lam, pi, gs)
        assert b >= prev - 1e-10
        assert lo_end - 1e-10 <= b <= hi_end + 1e-10
        prev = b


def test_budget_respected_at_solution():
    """The internal (q_lo, q_hi) solution must spend exactly the budget."""
    # indirect check: at gamma_sel just above 1 the bias barely exceeds random
    y = RNG.uniform(0, 1, 300)
    b_rand = max_bias_selection(y, 0.3, 0.2, 1.0)
    b_near = max_bias_selection(y, 0.3, 0.2, 1.01)
    assert 0 <= b_near - b_rand < 0.005


def test_validates_gamma():
    with pytest.raises(ValueError):
        max_bias_selection(np.array([0.5]), 0.3, 0.2, gamma_sel=0.5)
