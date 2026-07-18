"""Statistical layer: paired item bootstrap over the whole functional, and
Benjamini-Hochberg FDR control across a corpus of claims.

THEORY.md SS6. Joint (sampling + contamination) intervals, a percentile CI
for Gamma*, and the FDR machinery the pre-registration commits to. Items are
resampled jointly across models (pairing preserved).
"""

from __future__ import annotations

import math

import numpy as np

from .bounds import max_bias, max_bias_simple, max_inflation
from .csm import _validate_scores
from .gamma_star import _resolve_simple, gamma_star


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _imbens_manski_c(width_in_se: float, alpha: float) -> float:
    """The Imbens-Manski (2004) critical value: c solving
    Phi(c + w) - Phi(-c) = 1 - alpha, with w = set width in SE units.

    w = 0 recovers the two-sided z_{1-alpha/2}; w -> inf recovers the
    one-sided z_{1-alpha} (each endpoint only risks error on its own side).
    """
    lo, hi = 0.0, 10.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _norm_cdf(mid + width_in_se) - _norm_cdf(-mid) < 1.0 - alpha:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def joint_interval(
    y: np.ndarray,
    lam: float,
    pi: float,
    *,
    alpha: float = 0.05,
    n_boot: int = 2000,
    simple: bool | None = None,
    target: str = "parameter",
    seed: int = 42,
) -> tuple[float, float]:
    """Joint sampling + contamination interval for theta.

    target="parameter" (default): the Imbens-Manski (2004) interval for a
    partially identified PARAMETER -- endpoints widened by an adaptive
    critical value c in [z_{1-a}, z_{1-a/2}] that shrinks toward the
    one-sided value as the identified set widens (theta can only violate one
    endpoint at a time). This is the econometrics standard and the interval
    to report for theta itself.

    target="set": conservative coverage of the whole identified SET,
    [ q_{a/2}( mu_b - B_b ),  q_{1-a/2}( mu_b ) ] over bootstrap replicates
    -- wider, appropriate when the object of interest is the set.

    At lam = 0 both reduce to an ordinary CI for mu.
    simple=None auto-detects the regime (binary scores -> simple bound).
    """
    if target not in ("parameter", "set"):
        raise ValueError("target must be 'parameter' or 'set'")
    y = _validate_scores(y)
    use_simple = _resolve_simple(simple, y)
    rng = np.random.default_rng(seed)
    n = y.size
    lows = np.empty(n_boot)
    highs = np.empty(n_boot)
    for b in range(n_boot):
        yb = y[rng.integers(0, n, n)]
        mu_b = yb.mean()
        bias_b = max_bias_simple(lam, pi) if use_simple else max_bias(yb, lam, pi)
        lows[b] = mu_b - bias_b
        highs[b] = mu_b
    if target == "set":
        return (
            max(0.0, float(np.quantile(lows, alpha / 2))),
            float(np.quantile(highs, 1 - alpha / 2)),
        )
    mu = float(y.mean())
    bias = max_bias_simple(lam, pi) if use_simple else max_bias(y, lam, pi)
    lo_hat, hi_hat = mu - bias, mu
    se_lo = float(lows.std(ddof=1))
    se_hi = float(highs.std(ddof=1))
    se_max = max(se_lo, se_hi, 1e-12)
    c = _imbens_manski_c(bias / se_max, alpha)
    return (max(0.0, lo_hat - c * se_lo), min(1.0, hi_hat + c * se_hi))


def gamma_star_ci(
    y_a: np.ndarray,
    y_b: np.ndarray,
    pi: float,
    *,
    alpha: float = 0.05,
    n_boot: int = 1000,
    simple: bool | None = None,
    rho: float = 0.0,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Point estimate and paired-bootstrap percentile CI for Gamma*(pi).

    Returns (gamma_star, ci_low, ci_high). Replicates where the claim reverses
    (Delta <= 0) contribute Gamma* = 0; replicates robust to lam = 1 contribute
    ROBUST (inf), so ci_high may be inf -- that is a faithful summary.
    """
    y_a = _validate_scores(y_a)
    y_b = _validate_scores(y_b)
    if y_a.size != y_b.size:
        raise ValueError("paired bootstrap requires equal item counts")
    use_simple = _resolve_simple(simple, y_a, y_b)
    rng = np.random.default_rng(seed)
    n = y_a.size
    point = gamma_star(y_a, y_b, pi, simple=use_simple, rho=rho)
    reps = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        reps[b] = gamma_star(y_a[idx], y_b[idx], pi, simple=use_simple, rho=rho)
    finite = np.isfinite(reps)
    lo = float(np.quantile(reps[finite], alpha / 2)) if finite.any() else math.inf
    # Upper quantile over the extended reals: inf if too many ROBUST replicates.
    k_hi = int(np.ceil((1 - alpha / 2) * n_boot)) - 1
    hi = float(np.sort(reps)[k_hi])
    return point, lo, hi


def fragility_pvalue(
    y_a: np.ndarray,
    y_b: np.ndarray,
    lam: float,
    pi: float,
    *,
    n_boot: int = 1000,
    simple: bool | None = None,
    rho: float = 0.0,
    seed: int = 42,
) -> float:
    """Bootstrap p-value for H0: 'the claim A > B is NOT contamination-robust
    at CSM(lam, pi)' -- i.e. the share of paired replicates with
    Delta* <= worst-case bias*. Small p certifies robustness.

    Construction: confidence-interval inversion for the one-sided hypothesis
    Delta - B <= 0 -- p is the smallest alpha at which the percentile lower
    confidence bound on (Delta - B) exceeds zero. First-order accurate;
    near-boundary claims with skewed statistics would sharpen under BCa,
    at ~3x the compute. Add-one correction keeps p > 0 (Davison & Hinkley).
    """
    y_a = _validate_scores(y_a)
    y_b = _validate_scores(y_b)
    if y_a.size != y_b.size:
        raise ValueError("paired bootstrap requires equal item counts")
    use_simple = _resolve_simple(simple, y_a, y_b)
    rng = np.random.default_rng(seed)
    n = y_a.size
    hits = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        ya, yb = y_a[idx], y_b[idx]
        delta = float(ya.mean() - yb.mean())
        if use_simple:
            bias = max_bias_simple(lam, pi) * (1.0 + rho)
        else:
            bias = max_bias(ya, lam, pi)
            if rho > 0.0:
                bias += max_inflation(yb, min(1.0, rho * lam), pi)
        hits += int(delta <= bias)
    return (hits + 1) / (n_boot + 1)


def bh_fdr(
    pvalues: np.ndarray, q: float = 0.05, *, method: str = "bh"
) -> np.ndarray:
    """FDR step-up control: boolean mask of rejected hypotheses at level q.

    method="bh": Benjamini-Hochberg -- valid under independence or PRDS.
    Claims sharing models and items are positively dependent, which makes
    PRDS plausible but not proven for arbitrary claim corpora.
    method="by": Benjamini-Yekutieli -- valid under ARBITRARY dependence
    (thresholds divided by sum 1/i), the safe choice when the dependence
    structure is unverified. The pre-registered audit reports BH as primary
    with BY as the dependence-robust sensitivity.
    """
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1 or p.size == 0:
        raise ValueError("pvalues must be a non-empty 1-D array")
    if np.any(p < 0) or np.any(p > 1):
        raise ValueError("pvalues must lie in [0, 1]")
    if method not in ("bh", "by"):
        raise ValueError("method must be 'bh' or 'by'")
    m = p.size
    order = np.argsort(p)
    scale = float(np.sum(1.0 / np.arange(1, m + 1))) if method == "by" else 1.0
    thresholds = (q / scale) * (np.arange(1, m + 1) / m)
    passed = p[order] <= thresholds
    rejected = np.zeros(m, dtype=bool)
    if passed.any():
        k_max = int(np.max(np.nonzero(passed)[0]))
        rejected[order[: k_max + 1]] = True
    return rejected
