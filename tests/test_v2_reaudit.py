"""Tests for the v2 re-audit upgrades: Imbens-Manski parameter intervals and
Benjamini-Yekutieli dependence-robust FDR."""

import numpy as np
import pytest

from contamsens import bh_fdr, joint_interval
from contamsens.csm import CSM
from contamsens.inference import _imbens_manski_c

RNG = np.random.default_rng(42)
Z_95_ONE = 1.6449  # z_{0.95}
Z_95_TWO = 1.9600  # z_{0.975}


# ---------- Imbens-Manski critical value ----------

def test_im_critical_value_limits():
    """w=0 -> two-sided z; w large -> one-sided z; monotone in between."""
    assert _imbens_manski_c(0.0, 0.05) == pytest.approx(Z_95_TWO, abs=2e-3)
    assert _imbens_manski_c(50.0, 0.05) == pytest.approx(Z_95_ONE, abs=2e-3)
    cs = [_imbens_manski_c(w, 0.05) for w in (0.0, 0.5, 1.0, 2.0, 5.0, 50.0)]
    assert all(c2 <= c1 + 1e-9 for c1, c2 in zip(cs, cs[1:]))
    assert all(Z_95_ONE - 3e-3 <= c <= Z_95_TWO + 3e-3 for c in cs)


def test_parameter_interval_narrower_than_set_interval():
    y = RNG.uniform(0.2, 0.9, 600)
    lo_p, hi_p = joint_interval(y, lam=0.3, pi=0.15, n_boot=400,
                                target="parameter")
    lo_s, hi_s = joint_interval(y, lam=0.3, pi=0.15, n_boot=400, target="set")
    # both must contain the point identified set [mu - B, mu]
    from contamsens import max_bias
    mu, b = y.mean(), max_bias(y, 0.3, 0.15)
    assert lo_p <= mu - b and hi_p >= mu
    assert lo_s <= mu - b and hi_s >= mu
    assert joint_interval(y, 0.3, 0.15, n_boot=400, target="parameter") == (lo_p, hi_p)


def test_joint_interval_rejects_bad_target():
    with pytest.raises(ValueError):
        joint_interval(RNG.uniform(0, 1, 50), 0.2, 0.1, target="banana")


def test_im_worst_case_parameter_coverage():
    """Coverage of theta at the adversarial extreme (theta = lower endpoint,
    attained by fully contaminating the argmax deflation set). Nominal 95%;
    IM is asymptotic, so we require >= 0.90 at n = 400."""
    lam, pi, n = 0.4, 0.15, 400
    cover = 0
    reps = 150
    rng = np.random.default_rng(7)
    for _ in range(reps):
        y_star = rng.beta(2, 2, n)
        # adversarial contamination: inflate the k items whose OBSERVED
        # deflation capacity will be largest, up to the A1 cap
        k = CSM(lam, pi).budget(n)
        y_obs = y_star.copy()
        # inflate the k items with largest headroom-capped gain at the cap
        gain = lam * (1 - y_star)
        idx = np.argsort(gain)[-k:]
        y_obs[idx] = y_star[idx] + gain[idx]
        theta = y_star.mean()
        lo, hi = joint_interval(y_obs, lam, pi, n_boot=200, target="parameter",
                                seed=int(rng.integers(1 << 31)))
        cover += int(lo <= theta <= hi)
    assert cover / reps >= 0.90


# ---------- Benjamini-Yekutieli ----------

def test_by_more_conservative_than_bh():
    p = np.array([0.001, 0.008, 0.012, 0.02, 0.04, 0.3, 0.7])
    rej_bh = bh_fdr(p, q=0.05, method="bh")
    rej_by = bh_fdr(p, q=0.05, method="by")
    assert rej_by.sum() <= rej_bh.sum()
    assert not rej_by[~rej_bh].any()  # BY rejections are a subset of BH's


def test_by_hand_computed():
    """m=3, harmonic sum = 1 + 1/2 + 1/3 = 11/6; BY thresholds =
    q/(11/6) * (1/3, 2/3, 1) = (0.00909, 0.01818, 0.02727) at q=0.05."""
    p = np.array([0.009, 0.018, 0.5])
    rej = bh_fdr(p, q=0.05, method="by")
    assert rej[0] and rej[1] and not rej[2]
    # 0.010 > 0.00909 and fails at every higher rank too -> nothing rejected
    assert not bh_fdr(np.array([0.010, 0.5, 0.5]), q=0.05, method="by").any()


def test_bh_fdr_rejects_bad_method():
    with pytest.raises(ValueError):
        bh_fdr(np.array([0.01]), method="banana")
