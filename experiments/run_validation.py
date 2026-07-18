"""Ground-truth validation of contamsens (synthetic CONTAM-CTRL-lite).

Hypothesis H2 (smoke version): when contamination is injected at known strength
(lam_true, pi_true), the identified set at CSM(lam >= lam_true, pi >= pi_true)
covers the true clean score theta. Plus: sharpness, misspecification behaviour,
Gamma* on a synthetic leaderboard, and the Phase-0 vacuousness gate.

Deterministic: seed 42. CPU-only, runs in well under a minute.
Outputs -> results/*.csv and results/figures/*.png (see PROVENANCE.md).
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
    audit,
    frontier,
    gamma_star,
    identified_interval,
)

SEED = 42
RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"
RESULTS.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)


def inject(rng, n, lam_true, pi_true, difficulty=(2.0, 2.0)):
    """Generate clean scores y* ~ Beta and contaminate a random pi_true fraction
    with lift U(0, lam_true) * headroom (i.e., A1 holds with lam = lam_true)."""
    y_star = rng.beta(*difficulty, n)
    c = np.zeros(n, dtype=bool)
    c[rng.choice(n, size=int(round(pi_true * n)), replace=False)] = True
    lift = np.where(c, rng.uniform(0, lam_true, n) * (1 - y_star), 0.0)
    return y_star, np.clip(y_star + lift, 0, 1), c


# ------------------------------------------------------------------
# E1  Coverage under known truth, and under misspecified (lam, pi)
# ------------------------------------------------------------------
def e1_coverage(n_reps=500, n_items=500):
    rng = np.random.default_rng(SEED)
    grid_true = [(0.1, 0.1), (0.2, 0.1), (0.3, 0.2), (0.5, 0.3)]
    # analysis knobs as multiples of the truth: 0.5x (too small), 1x, 1.5x
    misspec = [0.5, 1.0, 1.5]
    rows = []
    for lam_t, pi_t in grid_true:
        cover = {m: 0 for m in misspec}
        widths = []
        biases = []
        for _ in range(n_reps):
            y_star, y_obs, _ = inject(rng, n_items, lam_t, pi_t)
            theta = y_star.mean()
            biases.append(y_obs.mean() - theta)
            for m in misspec:
                lo, hi = identified_interval(
                    y_obs, lam=min(1.0, m * lam_t), pi=min(1.0, m * pi_t)
                )
                if m == 1.0:
                    widths.append(hi - lo)
                cover[m] += int(lo - 1e-12 <= theta <= hi + 1e-12)
        rows.append(
            {
                "lam_true": lam_t,
                "pi_true": pi_t,
                "coverage_at_0.5x": cover[0.5] / n_reps,
                "coverage_at_truth": cover[1.0] / n_reps,
                "coverage_at_1.5x": cover[1.5] / n_reps,
                "mean_width_at_truth": float(np.mean(widths)),
                "mean_true_bias": float(np.mean(biases)),
                "sharpness_ratio": float(np.mean(biases) / np.mean(widths)),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "e1_coverage.csv", index=False)
    return df


# ------------------------------------------------------------------
# E2  Identified set vs lambda (the money figure, F2)
# ------------------------------------------------------------------
def e2_identified_set_figure(n_items=1000):
    rng = np.random.default_rng(SEED)
    lam_t, pi_t = 0.3, 0.15
    y_star, y_obs, _ = inject(rng, n_items, lam_t, pi_t)
    theta, mu = y_star.mean(), y_obs.mean()
    lams = np.linspace(0, 1, 101)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for pi, color in [(0.05, "#8ecae6"), (0.15, "#219ebc"), (0.30, "#023047")]:
        lows = [identified_interval(y_obs, la, pi)[0] for la in lams]
        ax.fill_between(lams, lows, mu, alpha=0.25, color=color, label=f"pi = {pi}")
        ax.plot(lams, lows, color=color, lw=1.5)
    ax.axhline(mu, color="k", lw=1, ls="--", label="observed score (upper bound)")
    ax.axhline(theta, color="#d62828", lw=1.5, label="true clean score theta")
    ax.axvline(lam_t, color="#d62828", lw=1, ls=":", label="true lambda")
    ax.set_xlabel("assumed lift strength lambda")
    ax.set_ylabel("benchmark score")
    ax.set_title("Identified set for the clean score vs sensitivity parameters")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIGS / "f2_identified_set.png", dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------
# E3  Gamma* on a synthetic leaderboard + contamination frontier
# ------------------------------------------------------------------
def e3_leaderboard(n_items=800):
    rng = np.random.default_rng(SEED)
    # Six models; model "m2_contam" is secretly contaminated (lam=0.4, pi=0.2)
    y = {}
    base = rng.beta(2.2, 1.8, n_items)  # shared item difficulty structure
    skills = {"m1": 0.10, "m2_contam": 0.02, "m3": 0.00, "m4": -0.05, "m5": -0.12, "m6": -0.25}
    for name, s in skills.items():
        clean = np.clip(base + s + rng.normal(0, 0.08, n_items), 0, 1)
        if name == "m2_contam":
            c = np.zeros(n_items, dtype=bool)
            c[rng.choice(n_items, size=int(0.2 * n_items), replace=False)] = True
            lift = np.where(c, rng.uniform(0, 0.4, n_items) * (1 - clean), 0.0)
            y[name] = np.clip(clean + lift, 0, 1)
        else:
            y[name] = clean

    df = pd.DataFrame(
        [
            {"model": m, "item": i, "score": float(v[i])}
            for m, v in y.items()
            for i in range(n_items)
        ]
    )
    out = audit(df, pi=0.2, lam_ref=0.25)
    out.to_csv(RESULTS / "e3_leaderboard_audit.csv", index=False)

    # Frontier figure for the top adjacent claim
    order = out["model_a"].tolist() + [out["model_b"].iloc[-1]]
    a, b = order[0], order[1]
    pis = np.linspace(0.02, 0.5, 60)
    wa = df[df.model == a].sort_values("item")["score"].to_numpy()
    wb = df[df.model == b].sort_values("item")["score"].to_numpy()
    front = frontier(wa, wb, pis)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    finite = np.isfinite(front)
    ax.plot(pis[finite], front[finite], color="#023047", lw=2)
    ax.fill_between(pis[finite], front[finite], 1.0, alpha=0.15, color="#d62828")
    ax.fill_between(pis[finite], 0.0, front[finite], alpha=0.15, color="#2a9d8f")
    ax.text(0.35, 0.15, "claim SURVIVES", color="#2a9d8f", fontsize=10)
    ax.text(0.30, 0.8, "claim OVERTURNABLE", color="#d62828", fontsize=10)
    ax.set_xlabel("assumed contaminated fraction pi")
    ax.set_ylabel("lift strength lambda")
    ax.set_ylim(0, 1)
    ax.set_title(f"Contamination frontier: '{a}' beats '{b}'")
    fig.tight_layout()
    fig.savefig(FIGS / "f5_frontier.png", dpi=200)
    plt.close(fig)
    return out


# ------------------------------------------------------------------
# E4  Vacuousness gate: interval width vs realistic margins
# ------------------------------------------------------------------
def e4_gate():
    lam_grid = [0.05, 0.1, 0.2, 0.3]
    pi_grid = [0.05, 0.1, 0.2, 0.5]
    margins = {"frontier_tight": 0.01, "frontier_typical": 0.03, "cross_tier": 0.10}
    rows = []
    for lam in lam_grid:
        for pi in pi_grid:
            width = pi * lam  # simple-bound width (upper bound on sharp width)
            row = {"lam": lam, "pi": pi, "max_width": width}
            for name, m in margins.items():
                row[f"overturns_{name}"] = width >= m
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "gate_analysis.csv", index=False)
    n_cells = len(df)
    frac_kill_tight = df["overturns_frontier_tight"].mean()
    frac_kill_cross = df["overturns_cross_tier"].mean()
    verdict = "PASS" if 0.0 < frac_kill_tight < 1.0 and frac_kill_cross < 0.5 else "FAIL"
    summary = (
        f"gate cells: {n_cells} | tight 1pt margins overturnable in "
        f"{frac_kill_tight:.0%} of plausible (lam,pi) cells | 10pt cross-tier "
        f"margins overturnable in {frac_kill_cross:.0%} | verdict: {verdict}"
    )
    (RESULTS / "gate_verdict.txt").write_text(summary, encoding="utf8")
    return summary, df


if __name__ == "__main__":
    print("E1: coverage under known injected contamination")
    cov = e1_coverage()
    print(cov.to_string(index=False))
    print("\nE2: identified-set figure -> results/figures/f2_identified_set.png")
    e2_identified_set_figure()
    print("\nE3: synthetic leaderboard audit (m2_contam secretly contaminated)")
    lb = e3_leaderboard()
    print(lb[["model_a", "model_b", "margin", "gamma_star_display", "robust_at_ref"]]
          .to_string(index=False))
    print("\nE4: vacuousness gate")
    summary, _ = e4_gate()
    print(summary)
