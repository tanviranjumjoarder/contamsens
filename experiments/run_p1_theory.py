"""Phase-1 theory validation (THEORY.md SS10).

T1  Bridge interpolation: B(lam, pi, gamma_sel) sweeps from random to adversarial.
T2  Consistency of Gamma-hat*: RMSE vs n on log-log axes (slope ~ -1/2 expected).
T3  (E9) Finite-draws plug-in bias: selection-on-noise (up) vs concavity/Jensen
    (down) as a function of draws-per-item m; plus coverage consequence.

Deterministic: seed 42. CPU-only, ~1 minute. Outputs -> results/, figures/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from contamsens import (  # noqa: E402
    deflation,
    gamma_star,
    identified_interval,
    max_bias,
    max_bias_selection,
)

SEED = 42
RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"


# ------------------------------------------------------------------
# T1  Bridge interpolation figure
# ------------------------------------------------------------------
def t1_bridge(n_items=2000):
    rng = np.random.default_rng(SEED)
    y = rng.beta(2, 2, n_items)
    lam, pi = 0.3, 0.15
    gammas = np.concatenate([np.linspace(1, 10, 40), np.linspace(10.5, 100, 20)])
    biases = [max_bias_selection(y, lam, pi, g) for g in gammas]
    lo_end = pi * deflation(y, lam).mean()
    hi_end = max_bias(y, lam, pi)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.semilogx(gammas, biases, color="#023047", lw=2)
    ax.axhline(lo_end, color="#2a9d8f", ls="--", lw=1.2,
               label=r"random contamination  $\pi\,\bar d$  ($\Gamma_{sel}=1$)")
    ax.axhline(hi_end, color="#d62828", ls="--", lw=1.2,
               label=r"adversarial top-$k$  ($\Gamma_{sel}=\infty$)")
    ax.set_xlabel(r"selection-odds bound $\Gamma_{sel}$")
    ax.set_ylabel(r"worst-case bias $B(\Lambda,\pi,\Gamma_{sel})$")
    ax.set_title("Bridge model: v0.1 is the Rosenbaum-type endpoint")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "f9_bridge_interpolation.png", dpi=200)
    plt.close(fig)
    return lo_end, hi_end


# ------------------------------------------------------------------
# T2  Consistency of Gamma-hat*
# ------------------------------------------------------------------
def _draw_pair(rng, n):
    """Two models with a fixed population margin over shared items."""
    base = rng.beta(2.0, 1.5, n)
    y_a = np.clip(base + rng.normal(0, 0.05, n), 0, 1)
    y_b = np.clip(base - 0.03 + rng.normal(0, 0.05, n), 0, 1)
    return y_a, y_b


def t2_consistency(n_reps=200, pi=0.15):
    rng = np.random.default_rng(SEED)
    # 'true' Gamma* from one very large draw
    y_a_big, y_b_big = _draw_pair(rng, 1_000_000)
    g_true = gamma_star(y_a_big, y_b_big, pi)
    rows = []
    for n in (100, 400, 1600, 6400, 25600):
        errs = []
        for _ in range(n_reps):
            y_a, y_b = _draw_pair(rng, n)
            g = gamma_star(y_a, y_b, pi)
            if np.isfinite(g):
                errs.append(g - g_true)
        errs = np.array(errs)
        rows.append(
            {"n": n, "gamma_true": g_true, "bias": errs.mean(),
             "rmse": float(np.sqrt((errs**2).mean())), "n_finite": errs.size}
        )
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "p1_t2_consistency.csv", index=False)

    # empirical convergence rate: slope of log RMSE on log n
    slope = np.polyfit(np.log(df["n"]), np.log(df["rmse"]), 1)[0]

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.loglog(df["n"], df["rmse"], "o-", color="#023047", label="RMSE")
    ref = df["rmse"].iloc[0] * (df["n"] / df["n"].iloc[0]) ** -0.5
    ax.loglog(df["n"], ref, "--", color="#999", label=r"$n^{-1/2}$ reference")
    ax.set_xlabel("items n")
    ax.set_ylabel(r"RMSE of $\hat\Gamma^*$")
    ax.set_title(f"Consistency of Gamma-hat* (empirical slope {slope:.2f})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "f10_gamma_consistency.png", dpi=200)
    plt.close(fig)
    return df, slope


# ------------------------------------------------------------------
# T3  (E9) Finite-draws plug-in bias and coverage
# ------------------------------------------------------------------
def t3_finite_draws(n_items=2000, n_reps=300, pi=0.1):
    rng = np.random.default_rng(SEED)
    rows = []
    for lam in (0.1, 0.3, 0.6):
        for m in (2, 5, 10, 20, 50):
            rel_bias = []
            cover = 0
            cover_simple = 0
            for _ in range(n_reps):
                p = rng.beta(2, 2, n_items)
                b_true = max_bias(p, lam, pi)
                p_hat = rng.binomial(m, p, n_items) / m
                b_plug = max_bias(p_hat, lam, pi)
                rel_bias.append((b_plug - b_true) / b_true)
                # coverage consequence: does [mu_hat - B_plug, mu_hat] cover
                # theta under true contamination at (lam, pi)?
                c = np.zeros(n_items, dtype=bool)
                c[rng.choice(n_items, int(pi * n_items), replace=False)] = True
                lift = np.where(c, rng.uniform(0, lam, n_items) * (1 - p), 0.0)
                p_obs = np.clip(p + lift, 0, 1)
                y_hat = rng.binomial(m, p_obs, n_items) / m
                lo, hi = identified_interval(y_hat, lam, pi)
                # sampling slack on BOTH ends (3 binomial SEs on the mean),
                # mirroring the joint bootstrap interval of inference.py --
                # identification bounds are population objects
                se = float(np.sqrt(np.mean(p_obs * (1 - p_obs)) / (n_items * m)))
                cover += int(lo - 3 * se <= p.mean() <= hi + 3 * se)
                # remedy for small m: the simple bound pi*lam is immune to
                # both selection-on-noise and Jensen smoothing
                lo_s, hi_s = identified_interval(y_hat, lam, pi, simple=True)
                cover_simple += int(lo_s - 3 * se <= p.mean() <= hi_s + 3 * se)
            rows.append(
                {"lam": lam, "m_draws": m,
                 "rel_bias_mean": float(np.mean(rel_bias)),
                 "rel_bias_sd": float(np.std(rel_bias)),
                 "coverage": cover / n_reps,
                 "coverage_simple": cover_simple / n_reps}
            )
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "p1_t3_finite_draws.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for lam, color in ((0.1, "#8ecae6"), (0.3, "#219ebc"), (0.6, "#023047")):
        sub = df[df.lam == lam]
        ax.plot(sub["m_draws"], 100 * sub["rel_bias_mean"], "o-", color=color,
                label=f"lambda = {lam}")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("draws per item m")
    ax.set_ylabel("plug-in bias of B, % of true value")
    ax.set_title("E9: selection-on-noise (up) vs Jensen concavity (down)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "f11_finite_draw_bias.png", dpi=200)
    plt.close(fig)
    return df


if __name__ == "__main__":
    print("T1: bridge interpolation")
    lo, hi = t1_bridge()
    print(f"  endpoints: random {lo:.4f}  ->  adversarial {hi:.4f} "
          f"(ratio {hi / lo:.1f}x)  -> figures/f9")
    print("\nT2: Gamma-hat* consistency")
    df2, slope = t2_consistency()
    print(df2.to_string(index=False))
    print(f"  empirical log-log slope: {slope:.3f}  (theory: -0.5)")
    print("\nT3 (E9): finite-draws plug-in bias + coverage")
    df3 = t3_finite_draws()
    print(df3.to_string(index=False))
