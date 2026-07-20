"""How much does the confirmatory headline depend on the frozen Λ_ref?

The pre-registered rule declares a claim non-robust iff Γ*(π=0.1) < Λ_ref,
with Λ_ref frozen at 0.355 (0.42 on HellaSwag). A natural objection is that
the headline rests on one calibrated constant. This script answers it by
sweeping the constant.

Two facts make the sweep clean. In the single-draw binary regime the audit
uses, the simple bound gives Γ*(π) = Δ/(π(1+ρ)); therefore

    non-robust  <=>  Δ < π·Λ·(1+ρ)

so (i) the decision depends on π and Λ only through the product π·Λ, and the
whole (π, Λ) plane collapses onto a single curve indexed by the maximum
admissible bias B = π·Λ; and (ii) the audit is *algebraically* a calibrated
margin threshold. We report both, rather than leaving a reviewer to find it.

Outputs -> results/lambda_sensitivity.csv, results/lambda_sensitivity.txt,
results/figures/f18_lambda_sensitivity.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RESULTS = ROOT / "results"

try:  # Windows consoles default to cp1252 and cannot print Λ / Γ
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

PI_FROZEN = 0.1
LAM_POOLED = 0.355
LAM_HELLASWAG = 0.42
H4_THRESHOLD = 0.25


def main() -> None:
    d = pd.read_csv(RESULTS / "confirmatory_audit.csv")
    n = len(d)
    margins = d["margin"].to_numpy()

    # --- 1. the frozen operating point, recomputed from margins alone -------
    frozen_thresh = PI_FROZEN * d["lam_ref"].to_numpy()
    frozen_k = int((margins < frozen_thresh).sum())
    assert frozen_k == int(d.non_robust_primary.sum()), "rule mismatch"

    # --- 2. sweep the maximum admissible bias B = pi * Lambda ---------------
    # uniform Lambda across tasks (removes stratification, one clean family)
    lam_grid = np.linspace(0.01, 1.0, 199)
    rows = []
    for lam in lam_grid:
        B = PI_FROZEN * lam
        k = int((margins < B).sum())
        bt = binomtest(k, n, H4_THRESHOLD, alternative="greater")
        rows.append({"lambda_uniform": lam, "max_bias_B": B,
                     "n_non_robust": k, "frac_non_robust": k / n,
                     "h4_p": bt.pvalue})
    sw = pd.DataFrame(rows)
    sw.to_csv(RESULTS / "lambda_sensitivity.csv", index=False)

    # --- 3. the break-even: where does H4 stop being supported? -------------
    # smallest Lambda whose fraction still exceeds the 25% pre-registered bar
    supported = sw[sw.frac_non_robust > H4_THRESHOLD]
    lam_breakeven = supported.lambda_uniform.min() if len(supported) else np.nan
    B_breakeven = PI_FROZEN * lam_breakeven
    # and where it becomes statistically significant at 0.05
    sig = sw[sw.h4_p < 0.05]
    lam_sig = sig.lambda_uniform.min() if len(sig) else np.nan

    # fraction at some reference points
    def frac_at(lam: float) -> float:
        return float((margins < PI_FROZEN * lam).sum()) / n

    lines = [
        "Λ_ref SENSITIVITY OF THE CONFIRMATORY HEADLINE",
        "=" * 62,
        f"claims n = {n}; frozen rule gives {frozen_k}/{n} "
        f"= {frozen_k / n:.1%} non-robust",
        "",
        "In the simple binary regime Γ*(π)=Δ/(π(1+ρ)), so a claim is "
        "non-robust iff",
        "its margin Δ falls below the maximum admissible bias B = π·Λ.",
        "The frozen operating point is B = 0.1 × 0.355 = 0.0355 "
        "(0.042 on HellaSwag).",
        "",
        "Fraction non-robust under a UNIFORM Λ (π = 0.1):",
    ]
    for lam in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.355, 0.42, 0.50, 0.60,
                0.80, 1.00):
        lines.append(f"    Λ = {lam:<5.3f}  (B = {PI_FROZEN * lam:.4f})  "
                     f"->  {frac_at(lam):6.1%} non-robust")
    lines += [
        "",
        f"H4 (> {H4_THRESHOLD:.0%} non-robust) is SUPPORTED for every "
        f"Λ ≥ {lam_breakeven:.3f} (B ≥ {B_breakeven:.4f}),",
        f"and statistically significant at α=0.05 for every Λ ≥ {lam_sig:.3f}.",
        "",
        "Interpretation: the calibrated field range for Λ is [0.10, 0.45];",
        "the eight quality-eligible estimates are {0.02, 0.08, 0.23, 0.31,",
        "0.40, 0.40, 0.42, 0.47} (median 0.355); the measured",
        "concentrated-exposure ceiling is ~0.73-1.00. H4 would fail only if",
        f"the true contamination strength were below Λ ≈ {lam_breakeven:.3f}"
        " — beneath",
        "the floor of the calibrated field range, and lower than seven of the",
        "eight quality-eligible point estimates (the exception is the MMLU",
        "row at 0.02, a benchmark not in the audited task set). The headline",
        "is therefore not an artifact of the particular frozen constant.",
    ]
    txt = "\n".join(lines)
    (RESULTS / "lambda_sensitivity.txt").write_text(txt, encoding="utf8")
    print(txt)

    # --- 4. figure ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.plot(sw.lambda_uniform, 100 * sw.frac_non_robust, color="#264653",
            lw=2.2, zorder=3, label="claims not contamination-robust")

    ax.axhspan(0, 100 * H4_THRESHOLD, color="#d62828", alpha=0.07, zorder=0)
    ax.axhline(100 * H4_THRESHOLD, color="#d62828", ls="--", lw=1.4,
               zorder=2, label=f"pre-registered H4 bar ({H4_THRESHOLD:.0%})")

    # calibrated field range and the frozen point
    ax.axvspan(0.10, 0.45, color="#2a9d8f", alpha=0.13, zorder=1,
               label="calibrated field range for Λ [0.10, 0.45]")
    ax.axvline(LAM_POOLED, color="#ffb703", lw=2.4, zorder=2,
               label=f"frozen Λ_ref = {LAM_POOLED}")
    ax.plot([LAM_POOLED], [100 * frac_at(LAM_POOLED)], "o", ms=8,
            color="#ffb703", mec="#333", mew=1.0, zorder=5)
    ax.annotate(f"{frac_at(LAM_POOLED):.1%}",
                xy=(LAM_POOLED, 100 * frac_at(LAM_POOLED)),
                xytext=(LAM_POOLED + 0.06, 100 * frac_at(LAM_POOLED) - 11),
                fontsize=10, fontweight="bold", color="#8a5a00",
                arrowprops=dict(arrowstyle="->", color="#8a5a00", lw=1.2))

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim()[0] * PI_FROZEN, ax.get_xlim()[1] * PI_FROZEN)
    ax2.set_xlabel("equivalent margin threshold  B = π·Λ  (π = 0.1)",
                   fontsize=9)

    ax.set_xlabel(r"contamination strength $\Lambda$ (uniform across tasks)")
    ax.set_ylabel("% of adjacent claims not robust")
    ax.set_ylim(0, 101)
    ax.set_xlim(0.01, 1.0)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.95)
    ax.set_title("The headline does not hinge on the frozen constant",
                 fontsize=11)
    ax.grid(alpha=0.25, zorder=0)
    fig.tight_layout()
    fig.savefig(RESULTS / "figures" / "f18_lambda_sensitivity.png", dpi=200)
    plt.close(fig)
    print("\nfigure -> results/figures/f18_lambda_sensitivity.png")


if __name__ == "__main__":
    main()
