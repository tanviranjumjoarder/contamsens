"""Regression tests for the forensic-audit fixes (iteration 1).

The binary-guard tests pin the CRITICAL defect: single-draw binary scores
degenerate the sharp knapsack to 0 (d(0) = d(1) = 0), which would silently
certify every claim as contamination-robust.
"""

import math
import warnings

import numpy as np
import pandas as pd
import pytest

from contamsens import (
    audit,
    bh_fdr,
    fragility_pvalue,
    gamma_star,
    is_binary_like,
    is_robust,
    max_bias,
)

RNG = np.random.default_rng(42)


# ---------- C1: the binary-score guard ----------

def test_binary_detection():
    assert is_binary_like(np.array([0.0, 1.0, 1.0, 0.0]))
    assert not is_binary_like(RNG.uniform(0.1, 0.9, 100))
    # clipped-continuous data with a few exact 0/1 must NOT trip the detector
    y = np.clip(RNG.normal(0.5, 0.35, 1000), 0, 1)
    assert not is_binary_like(y)


def test_sharp_bound_degenerates_on_binary():
    """The defect itself: knapsack bias is exactly 0 on 0/1 scores."""
    y = (RNG.uniform(size=500) < 0.7).astype(float)
    assert max_bias(y, lam=0.5, pi=0.3) == 0.0


def test_gamma_star_auto_routes_binary_to_simple():
    y_a = (RNG.uniform(size=1000) < 0.72).astype(float)
    y_b = (RNG.uniform(size=1000) < 0.68).astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g_auto = gamma_star(y_a, y_b, pi=0.2)          # auto -> simple
        g_sharp = gamma_star(y_a, y_b, pi=0.2, simple=False)
    delta = y_a.mean() - y_b.mean()
    if delta > 0:
        assert math.isfinite(g_auto)                    # simple gives a real number
        assert g_auto == pytest.approx(delta / 0.2, rel=1e-9)
        # The degenerate trap: on binary data B(lam) = 0 for all lam < 1 and
        # jumps at lam = 1 (deflation returns y itself there), so the sharp
        # Gamma* is vacuously 1.0 (or inf for huge margins) -- never the
        # honest simple answer.
        assert math.isinf(g_sharp) or g_sharp == pytest.approx(1.0, abs=1e-5)
        assert g_sharp > 4 * g_auto
    # explicit simple=False on binary data must warn
    with pytest.warns(UserWarning, match="binary"):
        gamma_star(y_a, y_b, pi=0.2, simple=False)


def test_audit_auto_regime_on_binary_leaderboard():
    rng = np.random.default_rng(0)
    rows = [
        {"model": m, "item": i, "score": float(rng.uniform() < p)}
        for m, p in (("good", 0.75), ("bad", 0.65))
        for i in range(800)
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = audit(pd.DataFrame(rows), pi=0.1)
    assert out["regime"].iloc[0] == "simple"
    assert math.isfinite(out["gamma_star"].iloc[0])  # not vacuously robust
    assert out["m_draws"].iloc[0] == 1


def test_audit_multidraw_uses_sharp():
    """>= 10 binary draws per item: per-item means are continuous -> sharp."""
    rng = np.random.default_rng(1)
    p = {"good": 0.75, "bad": 0.65}
    rows = [
        {"model": m, "item": i, "score": float(rng.uniform() < p[m])}
        for m in p for i in range(120) for _ in range(12)
    ]
    out = audit(pd.DataFrame(rows), pi=0.1)
    assert out["regime"].iloc[0] == "sharp"
    assert out["m_draws"].iloc[0] == 12


def test_audit_few_binary_draws_forced_simple():
    """Binary draws with 1 < m < 10: E9 says the plug-in knapsack is
    unreliable -> simple bound."""
    rng = np.random.default_rng(2)
    p = {"good": 0.75, "bad": 0.65}
    rows = [
        {"model": m, "item": i, "score": float(rng.uniform() < p[m])}
        for m in p for i in range(150) for _ in range(4)
    ]
    out = audit(pd.DataFrame(rows), pi=0.1)
    assert out["regime"].iloc[0] == "simple"
    assert out["m_draws"].iloc[0] == 4


def test_audit_continuous_single_measurement_uses_sharp():
    """One CONTINUOUS measurement per item (e.g. MASE-skill) is regime R1:
    the sharp knapsack is valid; the E9 draw rule must not fire."""
    rng = np.random.default_rng(4)
    base = rng.beta(2, 2, 300)
    rows = [
        {"model": m, "item": i, "score": float(np.clip(base[i] + off, 0, 1))}
        for m, off in (("hi", 0.1), ("lo", 0.0)) for i in range(300)
    ]
    out = audit(pd.DataFrame(rows), pi=0.2)
    assert out["regime"].iloc[0] == "sharp"
    assert out["m_draws"].iloc[0] == 1


# ---------- rho consistency across the API ----------

def test_is_robust_rho_consistent_with_gamma_star():
    y_a = RNG.uniform(0.3, 0.95, 400)
    y_b = np.clip(y_a - 0.05, 0, 1)
    g = gamma_star(y_a, y_b, pi=0.2, rho=0.3)
    if math.isfinite(g):
        assert is_robust(y_a, y_b, g - 1e-4, 0.2, rho=0.3)
        assert not is_robust(y_a, y_b, g + 1e-4, 0.2, rho=0.3)


def test_negative_rho_rejected():
    y = RNG.uniform(0, 1, 50)
    with pytest.raises(ValueError):
        gamma_star(y, y, 0.1, rho=-0.5)
    with pytest.raises(ValueError):
        is_robust(y, y, 0.2, 0.1, rho=-1.0)


# ---------- BH-FDR ----------

def test_bh_fdr_hand_computed():
    """Classic example: p = (.01, .02, .03, .04, .05), m=5, q=.05
    thresholds = (.01, .02, .03, .04, .05) -> all pass at their ranks."""
    p = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    assert bh_fdr(p, q=0.05).all()
    # nothing passes when all p large
    assert not bh_fdr(np.array([0.5, 0.6, 0.9]), q=0.05).any()
    # step-up: a late large p does not block earlier small ones
    rej = bh_fdr(np.array([0.001, 0.9, 0.9, 0.9]), q=0.05)
    assert rej[0] and not rej[1:].any()


def test_bh_fdr_validation():
    with pytest.raises(ValueError):
        bh_fdr(np.array([1.5]))
    with pytest.raises(ValueError):
        bh_fdr(np.array([]))


# ---------- fragility p-value + certified audit ----------

def test_fragility_pvalue_directions():
    y_a = RNG.uniform(0.5, 1.0, 600)          # huge margin claim
    y_b = np.clip(y_a - 0.30, 0, 1)
    p_strong = fragility_pvalue(y_a, y_b, lam=0.1, pi=0.05, n_boot=200)
    y_c = np.clip(y_a - 0.005, 0, 1)          # knife-edge claim
    p_weak = fragility_pvalue(y_a, y_c, lam=0.3, pi=0.3, n_boot=200)
    assert p_strong < 0.05 < p_weak


def test_audit_fdr_columns():
    rng = np.random.default_rng(3)
    base = rng.beta(2, 2, 400)
    models = {"big": 0.25, "mid": 0.24, "low": 0.05}
    rows = [
        {"model": m, "item": i,
         "score": float(np.clip(base[i] + off, 0, 1))}
        for m, off in models.items() for i in range(400)
    ]
    out = audit(pd.DataFrame(rows), pi=0.1, lam_ref=0.2, n_boot=100)
    assert {"fragility_p", "certified_robust_fdr"} <= set(out.columns)
    # the wide claim certifies, the knife-edge one does not
    wide_row = out[out.margin > 0.1].iloc[0]
    tight_row = out[out.margin < 0.05].iloc[0]
    assert wide_row["certified_robust_fdr"]
    assert not tight_row["certified_robust_fdr"]
