"""Audit a leaderboard: Gamma* for every adjacent (and optionally all) pairwise
claim, with robustness flags at a reference CSM and optional FDR-controlled
robustness certificates across the whole claim corpus.

Input: a DataFrame with one row per (model, item[, draw]) and a score column.
Duplicate (model, item) rows are treated as repeated draws and averaged into
per-item means -- the draw count m is reported, and the E9 prescription
(sharp knapsack only at m >= 10) is applied automatically in regime "auto".
Output: one row per claim.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd

from .csm import is_binary_like
from .gamma_star import gamma_star, is_robust, margin
from .inference import bh_fdr, fragility_pvalue

_MIN_DRAWS_FOR_SHARP = 10  # E9 prescription (THEORY.md SS10)


def _to_wide(
    df: pd.DataFrame, model_col: str, item_col: str, score_col: str
) -> tuple[pd.DataFrame, int]:
    """Pivot to models x items. Returns (wide, min_draws_per_cell)."""
    counts = df.groupby([model_col, item_col])[score_col].size()
    m_draws = int(counts.min())
    wide = df.pivot_table(
        index=model_col, columns=item_col, values=score_col, aggfunc="mean"
    )
    if wide.isna().any().any():
        raise ValueError("every model must score every item (paired design)")
    return wide, m_draws


def audit(
    df: pd.DataFrame,
    *,
    model_col: str = "model",
    item_col: str = "item",
    score_col: str = "score",
    pi: float = 0.1,
    lam_ref: float = 0.2,
    rho: float = 0.0,
    all_pairs: bool = False,
    simple: bool | None = None,
    n_boot: int = 0,
    fdr_q: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """Per-claim contamination audit.

    Regime ("simple"): None (default) auto-resolves per the theory -- the
    simple population bound when scores are single-draw binary OR the draw
    count is below the E9 threshold (m < 10); the sharp knapsack otherwise.
    Explicit True/False overrides (False on binary data triggers a loud
    warning; it is almost always a mistake).

    With n_boot > 0, each claim also gets a paired-bootstrap fragility
    p-value (H0: not robust at CSM(lam_ref, pi, rho)) and a
    Benjamini-Hochberg certificate column `certified_robust_fdr` at level
    fdr_q across the audited claim corpus (the pre-registered procedure).

    Selection caveat: adjacent pairs are chosen FROM the observed ranking,
    so the audited claim set (and each claim's direction) is data-selected
    -- a mild winner's-curse effect on the corpus-level certificate rate.
    Each per-claim test remains marginally valid for its fixed pair;
    all_pairs=True audits the selection-free full set.

    Returns one row per claim:
      model_a, model_b, margin, gamma_star, robust_at_ref, pi, lam_ref, rho,
      m_draws, regime[, fragility_p, certified_robust_fdr],
      gamma_star_display.
    gamma_star = inf means the claim survives full memorization of the budget.
    """
    wide, m_draws = _to_wide(df, model_col, item_col, score_col)
    scores_binary = any(
        is_binary_like(wide.loc[m].to_numpy()) for m in wide.index
    )
    # Regime resolution (THEORY SS5 + E9):
    #  - aggregated per-item scores still binary  -> R2, simple bound
    #  - raw scores are binary DRAWS with m < 10  -> E9, simple bound
    #    (noisy p-hat: the plug-in knapsack is unreliable below ~10 draws)
    #  - continuous per-item measurements (m = 1 or not draws) -> sharp
    raw_binary = is_binary_like(df[score_col].to_numpy())
    if simple is None:
        use_simple = scores_binary or (
            raw_binary and 1 < m_draws < _MIN_DRAWS_FOR_SHARP
        )
    else:
        use_simple = simple
        if not simple and scores_binary:
            warnings.warn(
                "simple=False on binary-looking per-item scores: the sharp "
                "bound is invalid in regime R2 and certifies everything as "
                "robust (THEORY.md SS5). Strongly prefer simple=None/True.",
                UserWarning,
                stacklevel=2,
            )

    order = wide.mean(axis=1).sort_values(ascending=False).index.tolist()
    if all_pairs:
        pairs = [(a, b) for i, a in enumerate(order) for b in order[i + 1:]]
    else:
        pairs = list(zip(order[:-1], order[1:]))

    rows = []
    for a, b in pairs:
        ya, yb = wide.loc[a].to_numpy(), wide.loc[b].to_numpy()
        row = {
            "model_a": a,
            "model_b": b,
            "margin": margin(ya, yb),
            "gamma_star": gamma_star(ya, yb, pi, simple=use_simple, rho=rho),
            "robust_at_ref": is_robust(
                ya, yb, lam_ref, pi, simple=use_simple, rho=rho
            ),
            "pi": pi,
            "lam_ref": lam_ref,
            "rho": rho,
            "m_draws": m_draws,
            "regime": "simple" if use_simple else "sharp",
        }
        if n_boot > 0:
            row["fragility_p"] = fragility_pvalue(
                ya, yb, lam_ref, pi,
                n_boot=n_boot, simple=use_simple, rho=rho, seed=seed,
            )
        rows.append(row)

    out = pd.DataFrame(rows)
    if n_boot > 0 and len(out):
        out["certified_robust_fdr"] = bh_fdr(
            out["fragility_p"].to_numpy(), q=fdr_q
        )
    out["gamma_star_display"] = out["gamma_star"].map(
        lambda g: "robust" if math.isinf(g) else f"{g:.3f}"
    )
    return out
