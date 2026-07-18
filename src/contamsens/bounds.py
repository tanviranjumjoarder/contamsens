"""Sharp and simple partial-identification bounds for the clean score theta.

THEORY.md SS3. All bounds are deterministic given per-item scores; sampling
uncertainty is layered on top in inference.py.
"""

from __future__ import annotations

import numpy as np

from .csm import CSM, _validate_scores, deflation


def max_bias(y: np.ndarray, lam: float, pi: float) -> float:
    """B(lam, pi): the sharp maximum total deflation (Proposition 1).

    Mean of the k = ceil(pi*n) largest per-item deflation capacities.
    Regime R1 (continuous / repeated-measure per-item scores) only.
    """
    y = _validate_scores(y)
    csm = CSM(lam, pi)
    k = csm.budget(y.size)
    if k == 0 or lam <= 0.0:
        return 0.0
    d = deflation(y, lam)
    topk = np.partition(d, y.size - k)[y.size - k:]
    return float(topk.sum() / y.size)


def max_bias_simple(lam: float, pi: float) -> float:
    """Distribution-free bound B <= pi * lam (Corollary 1).

    Also the sharp population bound in regime R2 (single-draw binary items).
    """
    CSM(lam, pi)  # validate
    return float(pi * lam)


def identified_interval(
    y: np.ndarray, lam: float, pi: float, *, simple: bool = False
) -> tuple[float, float]:
    """The sharp identified set [mu - B(lam, pi), mu] for theta (Proposition 1).

    simple=True uses the distribution-free width pi*lam instead of the knapsack
    (use for single-draw binary scores, where the per-item machinery is invalid
    -- see THEORY.md Remark on the ceiling effect).
    """
    y = _validate_scores(y)
    mu = float(y.mean())
    b = max_bias_simple(lam, pi) if simple else max_bias(y, lam, pi)
    return (max(0.0, mu - b), mu)


# ------------------- Phase-1 theory (THEORY.md SS10, Prop 3) -----------------

def max_bias_selection(
    y: np.ndarray, lam: float, pi: float, gamma_sel: float
) -> float:
    """Selection-bounded worst-case bias B(lam, pi, gamma_sel) — the bridge model.

    Contamination propensities q_i may vary across items only within a
    Rosenbaum-style odds band: odds(q_i)/odds(q_j) <= gamma_sel**2, subject to
    the budget mean(q) = pi. Worst case assigns the high propensity q_hi to the
    items with the largest deflation capacity (two-point LP solution), swept
    over the split point t.

    gamma_sel = 1   -> random contamination:      B = pi * mean(d)
    gamma_sel = inf -> adversarial (v0.1 top-k):  B = max_bias(y, lam, pi)
    Monotone nondecreasing in gamma_sel; bounds the EXPECTED bias.

    Budget convention: the propensity budget is continuous (mean q = pi),
    consistent with the expected-bias semantics; the deterministic knapsack
    endpoints use the conservative ceil(pi*n) item budget. The two differ by
    at most one item's deflation (< lam/n).
    """
    y = _validate_scores(y)
    if not gamma_sel >= 1.0:
        raise ValueError("gamma_sel must be >= 1")
    CSM(lam, pi)  # validate lam, pi
    d = deflation(y, lam)
    n = y.size
    if pi <= 0.0 or lam <= 0.0:
        return 0.0
    if np.isinf(gamma_sel):
        return max_bias(y, lam, pi)
    if gamma_sel == 1.0:
        return float(pi * d.mean())

    d_sorted = np.sort(d)[::-1]
    prefix = np.concatenate([[0.0], np.cumsum(d_sorted)])  # prefix[t] = sum top-t
    total = prefix[-1]
    g2 = gamma_sel**2
    t = np.arange(1, n + 1, dtype=float)

    # Vectorized bisection over q_lo in [0, 1] solving, for every split t:
    #   t * q_hi(q_lo) + (n - t) * q_lo = pi * n,
    #   q_hi = g2*q_lo / (1 + (g2 - 1)*q_lo)   (odds multiplied by g2).
    lo = np.zeros(n)
    hi = np.ones(n)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        q_hi = g2 * mid / (1.0 + (g2 - 1.0) * mid)
        f = t * q_hi + (n - t) * mid - pi * n
        too_low = f < 0.0
        lo = np.where(too_low, mid, lo)
        hi = np.where(too_low, hi, mid)
    q_lo = 0.5 * (lo + hi)
    q_hi = np.minimum(1.0, g2 * q_lo / (1.0 + (g2 - 1.0) * q_lo))
    bias_t = (q_hi * prefix[1:] + q_lo * (total - prefix[1:])) / n
    return float(max(bias_t.max(), pi * d.mean()))


# ----------------------- v0.2 upgrades (THEORY.md SS9) -----------------------

def max_inflation(y: np.ndarray, lam_minus: float, pi: float) -> float:
    """U(lam-, pi): maximum upward correction under two-sided CSM (U1).

    If contamination may also DEPRESS observed scores (stale memorization,
    post-training interference), a contaminated item's clean score can exceed
    the observed one by u_i = min(1 - y_i, y_i * lam- / (1 - lam-)).
    Returns the mean of the top-ceil(pi*n) inflation capacities.
    """
    y = _validate_scores(y)
    csm = CSM(lam_minus, pi)
    k = csm.budget(y.size)
    if k == 0 or lam_minus <= 0.0:
        return 0.0
    if lam_minus >= 1.0:
        u = 1.0 - y
    else:
        u = np.minimum(1.0 - y, y * lam_minus / (1.0 - lam_minus))
    topk = np.partition(u, y.size - k)[y.size - k:]
    return float(topk.sum() / y.size)


def identified_interval_twosided(
    y: np.ndarray,
    lam_plus: float,
    lam_minus: float,
    pi: float,
    *,
    spillover: float = 0.0,
) -> tuple[float, float]:
    """Two-sided identified set [mu - B(lam+, pi) - eps, mu + U(lam-, pi) + eps].

    lam_minus = 0 and spillover = 0 recover the one-sided v0.1 interval.

    `spillover` (eps >= 0) is the U4 channel (THEORY.md SS11): training on the
    leaked items may drift the model's scores on UNCONTAMINATED items by a
    bounded amount, shifting the whole frame; eps bounds the net drift and is
    calibratable from CONTAM-CTRL twin runs (mean |drift| on clean items).
    Negligible when leaked items are a vanishing fraction of the training
    corpus (the LLM regime); material at small scale.
    """
    y = _validate_scores(y)
    if spillover < 0:
        raise ValueError("spillover must be >= 0")
    mu = float(y.mean())
    return (
        max(0.0, mu - max_bias(y, lam_plus, pi) - spillover),
        min(1.0, mu + max_inflation(y, lam_minus, pi) + spillover),
    )


def max_bias_stratified(
    y: np.ndarray, lam: float, pi: float, eligible: np.ndarray
) -> float:
    """Stratified-budget bound (THEORY.md SS9 U3): only `eligible` items can be
    contaminated (e.g. items released before the model's training cutoff, or
    datasets with a high contamination prior). Budget pi applies to the full
    item count; ineligible items have c_i = 0 by construction.

    Never exceeds the unstratified bound; equals it when all items are eligible.
    """
    y = _validate_scores(y)
    eligible = np.asarray(eligible, dtype=bool)
    if eligible.shape != y.shape:
        raise ValueError("eligible mask must align with scores")
    csm = CSM(lam, pi)
    k = min(csm.budget(y.size), int(eligible.sum()))
    if k == 0 or lam <= 0.0:
        return 0.0
    d = np.where(eligible, deflation(y, lam), 0.0)
    topk = np.partition(d, y.size - k)[y.size - k:]
    return float(topk.sum() / y.size)


def max_bias_grouped(
    y: np.ndarray, lam: float, pi: float, groups: np.ndarray
) -> float:
    """Group-budget bound (U2): contamination is all-or-nothing per source group.

    Upper-bounds the adversary's knapsack by its fractional relaxation (greedy
    by per-item deflation density), which is valid (>= the integral optimum)
    hence conservative, and never exceeds the item-level bound at the same pi.
    """
    y = _validate_scores(y)
    groups = np.asarray(groups)
    if groups.shape != y.shape:
        raise ValueError("groups must align with scores")
    csm = CSM(lam, pi)
    budget = csm.budget(y.size)
    if budget == 0 or lam <= 0.0:
        return 0.0
    d = deflation(y, lam)
    total = 0.0
    remaining = budget
    # group totals and sizes, sorted by density (deflation per item)
    uniq = {}
    for g, di in zip(groups, d):
        s = uniq.setdefault(g, [0.0, 0])
        s[0] += float(di)
        s[1] += 1
    for gd, gn in sorted(uniq.values(), key=lambda s: s[0] / s[1], reverse=True):
        if remaining <= 0:
            break
        take = min(1.0, remaining / gn)
        total += take * gd
        remaining -= gn
    return float(total / y.size)
