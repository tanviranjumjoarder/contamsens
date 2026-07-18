"""Phase-4 PILOT re-audit on real data: the TSFM cross-domain transfer atlas.

*** EXPLORATORY PILOT — the confirmatory audit is governed by PREREGISTRATION.md
and uses a corpus frozen before analysis. Pairs analysed here are flagged. ***

Data: the author's own atlas (private repo `trustworthy-transferability-of-
time-series-foundation-models`), 6 frozen TSFMs x 17-18 datasets, with a
per-dataset `contamination_prior` annotation (low/high) made independently of
this project. Items = datasets; per-item score = in-domain zero-shot skill
s = 1/(1 + MASE) in (0,1], where pretraining contamination INFLATES skill --
matching the CSM direction.

Claims audited: "model A transfers/forecasts better than model B in-domain"
for adjacent pairs in the skill ranking. Regime R1 (continuous per-item
scores) -> sharp knapsack. Two analyses per claim:
  (a) unstratified budget  pi = 0.5      (any dataset may be contaminated)
  (b) stratified  (U3)     pi = 0.5 but only datasets with
      contamination_prior == 'high' are eligible
The (a) vs (b) gap is the real-data demonstration that provenance metadata
tightens identification for free.

Seed 42. CPU, seconds. Outputs -> results/p4_pilot_atlas.csv, fig f13.
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

from contamsens import ROBUST, max_bias_stratified  # noqa: E402

DATA = ROOT / "data" / "atlas_pilot"
RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"
PIS = (0.1, 0.5)  # pilot budgets: conservative and aggressive


def load_skill_matrix():
    delta = pd.read_csv(DATA / "atlas_delta.csv")
    corpus = pd.read_csv(DATA / "table1_corpus.csv")
    diag = delta[delta.source == delta.target].copy()
    diag["skill"] = 1.0 / (1.0 + diag["mase_indomain"])
    wide = diag.pivot_table(index="model", columns="target", values="skill")
    wide = wide.dropna(axis=1)  # shared datasets only (paired design)
    prior = corpus.set_index("dataset")["contamination_prior"]
    # align prior to the atlas dataset naming (monash_ prefixes etc.)
    def match_prior(col):
        for name, p in prior.items():
            if name in col or col in name:
                return p
        return "unknown"
    priors = pd.Series({c: match_prior(c) for c in wide.columns})
    return wide, priors


def gamma_star_stratified(y_a, eligible, delta, pi, tol=1e-6):
    """Gamma* under a stratified budget (bisection on max_bias_stratified)."""
    if delta <= 0:
        return 0.0
    if max_bias_stratified(y_a, 1.0, pi, eligible) < delta:
        return ROBUST
    lo, hi = 0.0, 1.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if max_bias_stratified(y_a, mid, pi, eligible) >= delta:
            hi = mid
        else:
            lo = mid
    return hi


def main():
    wide, priors = load_skill_matrix()
    n_items = wide.shape[1]
    n_high = int((priors == "high").sum())
    n_low = int((priors == "low").sum())
    order = wide.mean(axis=1).sort_values(ascending=False)
    print(f"items (shared datasets): {n_items} | prior high: {n_high} | "
          f"low: {n_low} | models: {list(order.index)}")

    eligible_all = np.ones(n_items, dtype=bool)
    # Conservative direction: a dataset whose prior could not be matched
    # ('unknown') stays ELIGIBLE for contamination -- only an explicit 'low'
    # annotation removes it from the adversary's reach. (Audit fix: treating
    # unknown as ineligible would shrink the adversary anti-conservatively.)
    eligible_high = (priors != "low").to_numpy()
    n_unknown = int((priors == "unknown").sum())
    if n_unknown:
        print(f"note: {n_unknown} dataset(s) with unmatched contamination "
              f"prior kept eligible (conservative)")

    rows = []
    models = order.index.tolist()
    for pi in PIS:
        for a, b in zip(models[:-1], models[1:]):
            y_a = wide.loc[a].to_numpy()
            y_b = wide.loc[b].to_numpy()
            delta = float(y_a.mean() - y_b.mean())
            g_unstrat = gamma_star_stratified(y_a, eligible_all, delta, pi)
            g_strat = gamma_star_stratified(y_a, eligible_high, delta, pi)
            rows.append({
                "claim": f"{a} > {b}",
                "margin_skill": round(delta, 4),
                "gamma_star_unstratified": g_unstrat,
                "gamma_star_stratified_high_only": g_strat,
                "pi": pi,
                "n_items": n_items,
                "status": "EXPLORATORY_PILOT",
            })

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "p4_pilot_atlas.csv", index=False)
    show = df.copy()
    for c in ("gamma_star_unstratified", "gamma_star_stratified_high_only"):
        show[c] = show[c].map(lambda g: "robust" if math.isinf(g) else f"{g:.3f}")
    print(show.to_string(index=False))

    # figure: per-claim Gamma* under both budgets, conservative pi
    sub = df[df.pi == PIS[0]].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(sub))
    gu = [min(g, 1.05) if math.isfinite(g) else 1.05
          for g in sub["gamma_star_unstratified"]]
    gs = [min(g, 1.05) if math.isfinite(g) else 1.05
          for g in sub["gamma_star_stratified_high_only"]]
    ax.bar(x - 0.18, gu, 0.36, color="#d62828", alpha=0.8,
           label="unstratified budget")
    ax.bar(x + 0.18, gs, 0.36, color="#2a9d8f", alpha=0.8,
           label="stratified: only high-prior datasets eligible (U3)")
    ax.axhspan(0.1, 0.45, color="#ffb703", alpha=0.15,
               label="calibrated Lambda central range (lambda_priors.csv)")
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.text(len(sub) - 0.5, 1.06, "robust", fontsize=8, ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace(" > ", "\n> ") for c in sub["claim"]],
                       fontsize=7)
    ax.set_ylabel(r"$\Gamma^*$ (higher = harder to overturn)")
    ax.set_title(f"PILOT: TSFM atlas in-domain ranking, pi = {PIS[0]} "
                 f"(exploratory; pre-registered audit pending)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "f13_pilot_atlas.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
