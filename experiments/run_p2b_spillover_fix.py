"""Phase-2 loop iteration: re-test H2 coverage on the configs that failed,
with the U4 spillover channel added to the interval.

Diagnosis from run_p2_contamctrl.py: logreg coverage failures (0.33-0.67) are
driven not by a wrong envelope (env_corr ~ +0.93) but by NET SPILLOVER --
adding leaked items to training drifts scores on uncontaminated items, moving
the whole frame by more than the tiny contaminated-mean effect. Remedy:
widen the interval by eps-hat = mean |clean-item drift| estimated on the
calibration seeds (a quantity CONTAM-CTRL measures directly).

Re-runs only the failing / marginal configs. Seed conventions identical to
run_p2_contamctrl.py. Outputs -> results/p2b_spillover_fix.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from contamsens import identified_interval_twosided  # noqa: E402
from run_p2_contamctrl import run_twin  # noqa: E402

RESULTS = ROOT / "results"
N_SEEDS = 6
CONFIGS = [  # every config with H2r coverage < 1.0 in the first pass
    ("logreg", 0.05, 4), ("logreg", 0.10, 4), ("logreg", 0.20, 4),
    ("logreg", 0.10, 1), ("logreg", 0.10, 16),
    ("mlp", 0.05, 4),
]


def calibrate(name, pi, dose, seeds):
    """lambda-hat (q95 envelope) and eps-hat (mean |clean drift|) from seeds."""
    ratios, spills = [], []
    for s in seeds:
        y_clean, y_obs, c = run_twin(name, s, pi, dose)
        lift = y_obs[c] - y_clean[c]
        hr = 1 - y_clean[c]
        ok = hr > 0.02
        ratios.append(lift[ok] / hr[ok])
        spills.append(float(np.mean(np.abs(y_obs[~c] - y_clean[~c]))))
    lam_hat = min(1.0, float(np.quantile(np.concatenate(ratios), 0.95)) * 1.1)
    eps_hat = float(np.mean(spills))
    return lam_hat, eps_hat


def main():
    rows = []
    for name, pi, dose in CONFIGS:
        lam_hat, eps_hat = calibrate(name, pi, dose, range(0, 3))
        cov_no_eps = cov_eps = 0
        n_test = 0
        for s in range(3, N_SEEDS):
            y_clean, y_obs, _c = run_twin(name, s, pi, dose)
            theta = y_clean.mean()
            lo0, hi0 = identified_interval_twosided(y_obs, lam_hat, 0.0, pi)
            lo1, hi1 = identified_interval_twosided(
                y_obs, lam_hat, 0.0, pi, spillover=eps_hat
            )
            cov_no_eps += int(lo0 <= theta <= hi0)
            cov_eps += int(lo1 <= theta <= hi1)
            n_test += 1
        rows.append({
            "model": name, "pi": pi, "dose": dose,
            "lambda_hat": round(lam_hat, 3),
            "eps_hat_spillover": round(eps_hat, 4),
            "coverage_without_spillover": cov_no_eps / n_test,
            "coverage_with_spillover": cov_eps / n_test,
        })
        print(f"  {name:7s} pi={pi:.2f} dose={dose:2d} | lam {lam_hat:.3f} "
              f"| eps {eps_hat:.4f} | cov {cov_no_eps}/{n_test} -> {cov_eps}/{n_test}")
    pd.DataFrame(rows).to_csv(RESULTS / "p2b_spillover_fix.csv", index=False)


if __name__ == "__main__":
    print("Phase-2b: H2 coverage with the U4 spillover channel")
    main()
