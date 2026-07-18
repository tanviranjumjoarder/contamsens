"""Tests for the bootstrap layer and the leaderboard audit API."""

import math

import numpy as np
import pandas as pd
import pytest

from contamsens import audit, gamma_star_ci, joint_interval

RNG = np.random.default_rng(42)


def test_joint_interval_reduces_to_bootstrap_ci_at_lambda_zero():
    y = RNG.uniform(0.4, 0.8, 400)
    lo, hi = joint_interval(y, lam=0.0, pi=0.3, n_boot=500)
    assert lo < y.mean() < hi
    assert hi - lo < 0.06  # ordinary sampling width, no contamination inflation


def test_joint_interval_widens_with_contamination():
    y = RNG.uniform(0.4, 0.8, 400)
    # set-level construction: upper endpoint is invariant to lam
    lo0, hi0 = joint_interval(y, lam=0.0, pi=0.2, n_boot=400, target="set")
    lo1, hi1 = joint_interval(y, lam=0.3, pi=0.2, n_boot=400, target="set")
    assert (hi1 - lo1) > (hi0 - lo0)
    assert hi1 == pytest.approx(hi0, abs=1e-9)  # only the lower side moves
    # parameter-level (Imbens-Manski) construction: the critical value
    # adapts to the set width, so the upper endpoint tightens WEAKLY as
    # the set widens, while the interval still widens overall
    lo0p, hi0p = joint_interval(y, lam=0.0, pi=0.2, n_boot=400)
    lo1p, hi1p = joint_interval(y, lam=0.3, pi=0.2, n_boot=400)
    assert (hi1p - lo1p) > (hi0p - lo0p)
    assert hi1p <= hi0p + 1e-9


def test_gamma_star_ci_brackets_point():
    y_a = RNG.uniform(0.3, 0.95, 500)
    y_b = np.clip(y_a - 0.06 + RNG.normal(0, 0.03, 500), 0, 1)
    point, lo, hi = gamma_star_ci(y_a, y_b, pi=0.2, n_boot=200)
    assert math.isfinite(point)
    assert lo <= point <= hi or math.isinf(hi)


def test_audit_shape_and_ordering():
    rng = np.random.default_rng(0)
    models = {"strong": 0.85, "mid": 0.80, "weak": 0.60}
    rows = [
        {"model": m, "item": i, "score": float(np.clip(mu + rng.normal(0, 0.1), 0, 1))}
        for m, mu in models.items()
        for i in range(300)
    ]
    df = pd.DataFrame(rows)
    out = audit(df, pi=0.1, lam_ref=0.2)
    assert list(out["model_a"]) == ["strong", "mid"]
    assert list(out["model_b"]) == ["mid", "weak"]
    # tight claim fragile, wide claim robust at the reference CSM
    tight = out.iloc[0]
    wide = out.iloc[1]
    assert wide["margin"] > tight["margin"]
    assert wide["gamma_star"] > tight["gamma_star"] or math.isinf(wide["gamma_star"])


def test_audit_rejects_unpaired():
    df = pd.DataFrame(
        {"model": ["a", "a", "b"], "item": [0, 1, 0], "score": [0.5, 0.6, 0.4]}
    )
    with pytest.raises(ValueError):
        audit(df)
