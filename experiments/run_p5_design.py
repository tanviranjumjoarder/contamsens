"""Phase-5 design sensitivity: which benchmark designs buy identification?

Two levers, both prescriptive:
  D1  Fresh-item fraction f: items released after every audited model's
      training cutoff are ineligible for contamination (pi_s = 0 stratum).
      Gamma* as a function of f = the value of dating your items.
  D2  Draws per item m: below m ~ 10 the sharp knapsack is unreliable (E9),
      forcing the wider simple bound. Gamma*_knapsack / Gamma*_simple = the
      value of publishing repeated draws.

Seed 42. CPU, seconds. Outputs -> results/p5_design.csv, fig f14.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from contamsens import gamma_star, max_bias_stratified  # noqa: E402

SEED = 42
RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"
N_ITEMS, PI = 1000, 0.2


def gamma_star_stratified(y_a, eligible, delta, pi, tol=1e-6):
    if delta <= 0:
        return 0.0
    if max_bias_stratified(y_a, 1.0, pi, eligible) < delta:
        return math.inf
    lo, hi = 0.0, 1.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if max_bias_stratified(y_a, mid, pi, eligible) >= delta:
            hi = mid
        else:
            lo = mid
    return hi


def d1_freshness(rng):
    rows = []
    y_a = rng.beta(2, 2, N_ITEMS)
    for delta in (0.01, 0.03):
        for f in np.linspace(0.0, 0.9, 19):
            eligible = np.ones(N_ITEMS, dtype=bool)
            fresh = rng.choice(N_ITEMS, int(f * N_ITEMS), replace=False)
            eligible[fresh] = False
            g = gamma_star_stratified(y_a, eligible, delta, PI)
            rows.append({"lever": "fresh_fraction", "x": round(float(f), 3),
                         "margin": delta, "gamma_star": g})
    return rows


def d2_draws(rng):
    rows = []
    p_a = rng.beta(2, 2, N_ITEMS)
    p_b = np.clip(p_a - 0.03, 0, 1)
    for m in (2, 5, 10, 20, 50, 200):
        ya = rng.binomial(m, p_a, N_ITEMS) / m
        yb = rng.binomial(m, p_b, N_ITEMS) / m
        # E9 prescription: sharp knapsack only trustworthy at m >= 10
        g_sharp = gamma_star(ya, yb, PI) if m >= 10 else np.nan
        g_simple = gamma_star(ya, yb, PI, simple=True)
        rows.append({"lever": "draws_per_item", "x": m, "margin": 0.03,
                     "gamma_star": g_sharp, "gamma_star_simple": g_simple})
    return rows


def main():
    rng = np.random.default_rng(SEED)
    rows = d1_freshness(rng) + d2_draws(rng)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "p5_design.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    d1 = df[df.lever == "fresh_fraction"]
    for delta, color in ((0.01, "#219ebc"), (0.03, "#023047")):
        sub = d1[d1.margin == delta]
        ys = [min(g, 1.05) if math.isfinite(g) else 1.05
              for g in sub["gamma_star"]]
        axes[0].plot(sub["x"], ys, "o-", ms=3, color=color,
                     label=f"margin {delta * 100:.0f} pt")
    axes[0].axhspan(0.1, 0.45, color="#ffb703", alpha=0.15,
                    label="calibrated Lambda range")
    axes[0].set_xlabel("fraction of post-cutoff (fresh) items f")
    axes[0].set_ylabel(r"$\Gamma^*$")
    axes[0].set_title("D1: freshness buys robustness")
    axes[0].legend(fontsize=8)

    d2 = df[df.lever == "draws_per_item"]
    axes[1].plot(d2["x"], d2["gamma_star_simple"], "s--", color="#d62828",
                 label="simple bound (any m)")
    ok = d2.dropna(subset=["gamma_star"])
    axes[1].plot(ok["x"], ok["gamma_star"], "o-", color="#2a9d8f",
                 label="sharp knapsack (valid at m >= 10, E9)")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("draws per item m")
    axes[1].set_ylabel(r"$\Gamma^*$")
    axes[1].set_title("D2: repeated draws buy sharpness")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "f14_design_sensitivity.png", dpi=200)
    plt.close(fig)
    print(df.groupby("lever").size().to_string())
    print("\nD1 at margin 3pt: Gamma* rises from "
          f"{d1[(d1.margin == 0.03) & (d1.x == 0.0)]['gamma_star'].iloc[0]:.3f} (f=0) to "
          f"{min(d1[(d1.margin == 0.03) & (d1.x == 0.9)]['gamma_star'].iloc[0], 999):.3f} (f=0.9)")
    d2s = df[df.lever == 'draws_per_item']
    print("D2 at m=20: sharp", f"{d2s[d2s.x == 20]['gamma_star'].iloc[0]:.3f}",
          "vs simple", f"{d2s[d2s.x == 20]['gamma_star_simple'].iloc[0]:.3f}")


if __name__ == "__main__":
    main()
