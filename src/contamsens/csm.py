"""The Contamination Sensitivity Model CSM(lam, pi).

Axioms (THEORY.md SS2):
  (A1) 0 <= y_i - y*_i <= c_i * lam * (1 - y*_i)   (monotone bounded lift)
  (A2) mean(c) <= pi                                (budget)

lam in [0,1]: fraction of an item's headroom that leakage can close.
pi  in [0,1]: maximum fraction of contaminated items.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _validate_scores(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or y.size == 0:
        raise ValueError("scores must be a non-empty 1-D array")
    if np.any(y < 0) or np.any(y > 1) or np.any(~np.isfinite(y)):
        raise ValueError("scores must lie in [0, 1]")
    return y


def is_binary_like(y: np.ndarray, tol: float = 0.02) -> bool:
    """True when scores look like single-draw binary outcomes (regime R2).

    In that regime the per-item sharp machinery is INVALID: d(0) = d(1) = 0
    (the ceiling effect, THEORY.md SS2 Remark), so the knapsack bound
    degenerates to 0 and would silently certify every claim as robust.
    Callers use this to route to the simple population bound instead.
    """
    y = np.asarray(y, dtype=float)
    frac_binary = float(np.mean((y == 0.0) | (y == 1.0)))
    return frac_binary >= 1.0 - tol


def item_floor(y: np.ndarray, lam: float) -> np.ndarray:
    """Per-item lower bound l_i(lam) on the clean score of a contaminated item.

    Inverts A1: y <= lam + (1-lam) y*  =>  y* >= (y - lam) / (1 - lam), clipped at 0.
    At lam = 1 the floor is 0 (full memorization possible).
    """
    y = _validate_scores(y)
    if lam >= 1.0:
        return np.zeros_like(y)
    return np.maximum(0.0, (y - lam) / (1.0 - lam))


def deflation(y: np.ndarray, lam: float) -> np.ndarray:
    """Per-item deflation capacity d_i(lam) = y_i - l_i(lam).

    d_i = y_i                       if y_i <= lam
        = lam * (1 - y_i)/(1 - lam) if y_i >  lam
    Continuous, maximized at y_i = lam with value lam, nondecreasing in lam.
    """
    y = _validate_scores(y)
    if lam <= 0.0:
        return np.zeros_like(y)
    if lam >= 1.0:
        return y.copy()
    return np.where(y <= lam, y, lam * (1.0 - y) / (1.0 - lam))


@dataclass(frozen=True)
class CSM:
    """A contamination sensitivity model with fixed (lam, pi)."""

    lam: float
    pi: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.lam <= 1.0):
            raise ValueError(f"lam must be in [0, 1], got {self.lam}")
        if not (0.0 <= self.pi <= 1.0):
            raise ValueError(f"pi must be in [0, 1], got {self.pi}")

    def budget(self, n: int) -> int:
        """Item budget k = ceil(pi * n), capped at n (conservative rounding)."""
        return min(n, int(np.ceil(self.pi * n - 1e-12)))
