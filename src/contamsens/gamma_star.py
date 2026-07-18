"""The contamination robustness value Gamma* and the contamination frontier.

THEORY.md SS4. Gamma*(pi) is the minimum lift strength lam at which the claim
"A beats B" can be overturned by contamination of A alone (the worst case,
since contamination only inflates scores).
"""

from __future__ import annotations

import math
import warnings

import numpy as np

from .bounds import max_bias, max_bias_simple, max_inflation
from .csm import _validate_scores, is_binary_like

ROBUST = math.inf  # claim survives even lam = 1 within budget

_BINARY_MSG = (
    "scores look single-draw binary (regime R2): the sharp knapsack is invalid "
    "there (ceiling effect, THEORY.md SS5) and would certify every claim as "
    "robust. Using the simple population bound instead; pass simple=False "
    "explicitly only for genuinely continuous scores."
)


def _resolve_simple(simple: bool | None, *arrays: np.ndarray) -> bool:
    """Resolve the regime: None = auto-detect; explicit False on binary-looking
    data is honored but warned about loudly (it is almost always a mistake)."""
    binary = any(is_binary_like(a) for a in arrays)
    if simple is None:
        if binary:
            warnings.warn(_BINARY_MSG, UserWarning, stacklevel=3)
        return binary
    if simple is False and binary:
        warnings.warn(
            "simple=False on binary-looking scores: the sharp bound degenerates "
            "to 0 and the result is vacuously 'robust'. " + _BINARY_MSG,
            UserWarning,
            stacklevel=3,
        )
    return simple


def margin(y_a: np.ndarray, y_b: np.ndarray) -> float:
    """Observed margin Delta = mean(A) - mean(B)."""
    return float(_validate_scores(y_a).mean() - _validate_scores(y_b).mean())


def gamma_star(
    y_a: np.ndarray,
    y_b: np.ndarray,
    pi: float,
    *,
    simple: bool | None = None,
    rho: float = 0.0,
    tol: float = 1e-6,
) -> float:
    """Gamma*(pi) = inf{lam : worst-case bias >= Delta} via bisection.

    One-sided (rho=0, v0.1): worst case is A inflated by B_A(lam, pi).
    Two-sided (rho>0, THEORY.md SS9 U1): B may also be deflated with strength
    lam- = rho * lam, so the worst case is B_A(lam, pi) + U_B(rho*lam, pi).

    simple: None (default) auto-detects the regime -- single-draw binary
    scores are routed to the simple population bound (regime R2), continuous
    scores to the sharp knapsack (regime R1). Explicit True/False overrides.
    The simple closed form is Gamma* = Delta / (pi * (1 + rho)).

    Returns:
      0.0     if Delta <= 0 (the claim does not even hold at face value),
      ROBUST  (math.inf) if the claim survives lam = 1 within budget,
      lam*    otherwise.
    """
    if rho < 0.0:
        raise ValueError("rho must be >= 0")
    delta = margin(y_a, y_b)
    if delta <= 0.0:
        return 0.0
    use_simple = _resolve_simple(simple, y_a, y_b)

    if use_simple:
        if pi <= 0.0:
            return ROBUST
        g = delta / (pi * (1.0 + rho))
        return g if g <= 1.0 else ROBUST

    def bias(lam: float) -> float:
        b = max_bias(y_a, lam, pi)
        if rho > 0.0:
            b += max_inflation(y_b, min(1.0, rho * lam), pi)
        return b

    if bias(1.0) < delta:
        return ROBUST
    lo, hi = 0.0, 1.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if bias(mid) >= delta:
            hi = mid
        else:
            lo = mid
    return hi


def is_robust(
    y_a: np.ndarray,
    y_b: np.ndarray,
    lam: float,
    pi: float,
    *,
    simple: bool | None = None,
    rho: float = 0.0,
) -> bool:
    """Proposition 2 (two-sided form): the claim A > B holds over the whole
    identified set iff Delta > B_A(lam, pi) + U_B(rho*lam, pi)."""
    if rho < 0.0:
        raise ValueError("rho must be >= 0")
    delta = margin(y_a, y_b)
    if delta <= 0.0:
        return False
    use_simple = _resolve_simple(simple, y_a, y_b)
    if use_simple:
        b = max_bias_simple(lam, pi) * (1.0 + rho)
    else:
        b = max_bias(y_a, lam, pi)
        if rho > 0.0:
            b += max_inflation(y_b, min(1.0, rho * lam), pi)
    return delta > b


def frontier(
    y_a: np.ndarray,
    y_b: np.ndarray,
    pis: np.ndarray,
    *,
    simple: bool | None = None,
    rho: float = 0.0,
) -> np.ndarray:
    """Gamma*(pi) over a grid of budgets: the contamination frontier.

    Under the simple bound this is the hyperbola pi * lam * (1 + rho) = Delta.
    """
    return np.array(
        [gamma_star(y_a, y_b, float(p), simple=simple, rho=rho) for p in pis]
    )
