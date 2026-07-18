"""Phase-2 CONTAM-CTRL (small scale): REAL contamination via actual training.

Unlike run_validation.py (formula injection that satisfies A1 by construction),
here contamination is emergent: a fraction pi of test items, with their realized
labels, is planted in the training set (each repeated `dose` times), and a model
family actually memorizes them through its own capacity. This tests, externally:

  E5   A1 channel shape -- is the lift envelope proportional to headroom?
  E5b  Monotonicity -- how often does contamination HURT an item (rho evidence)?
  E5c  Spillover -- does contaminating some items shift the others (SUTVA)?
  E5d  Dose & capacity response -- mirrors arXiv 2601.04301 at small scale.
  H2r  Coverage on real memorization -- interval at (Lambda-hat, pi) must cover
       the clean twin's score. Lambda-hat estimated on held-out seeds.

Task: binary classification with irreducible noise (Bayes error > 0), so
memorization has real headroom to exploit. Per-item score = predicted
probability of the realized label (continuous, regime R1).

Seed 42. CPU-only, a few minutes. Outputs -> results/, figures/.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from contamsens import identified_interval  # noqa: E402

SEED = 42
RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"

N_TRAIN, N_TEST, DIM = 3000, 600, 10
N_SEEDS = 6  # seeds 0-2 calibrate Lambda-hat, seeds 3-5 test coverage


def make_task(rng):
    """Nonlinear binary task with irreducible label noise."""
    x = rng.normal(0, 1, (N_TRAIN + N_TEST, DIM))
    logit = (
        1.2 * x[:, 0] - 0.8 * x[:, 1] + 1.5 * x[:, 0] * x[:, 2]
        + 0.9 * np.sin(2 * x[:, 3]) - 0.6 * x[:, 4] ** 2 + 0.4
    )
    p = 1 / (1 + np.exp(-0.8 * logit))  # temper -> Bayes error ~ 20%
    y = (rng.uniform(size=p.size) < p).astype(int)
    return (x[:N_TRAIN], y[:N_TRAIN]), (x[N_TRAIN:], y[N_TRAIN:])


def model_factory(name, seed):
    if name == "logreg":
        return LogisticRegression(max_iter=1000, random_state=seed)
    if name == "mlp":
        return MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=400,
                             random_state=seed)
    if name == "rf":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=1,
                                      random_state=seed, n_jobs=-1)
    raise ValueError(name)


def run_twin(name, seed, pi, dose):
    """Train clean and contaminated twins; return per-item scores and c mask."""
    rng = np.random.default_rng(1000 * seed + SEED)
    (xtr, ytr), (xte, yte) = make_task(rng)
    n_c = int(round(pi * N_TEST))
    c = np.zeros(N_TEST, dtype=bool)
    c[rng.choice(N_TEST, n_c, replace=False)] = True

    clean = model_factory(name, seed).fit(xtr, ytr)
    x_aug = np.vstack([xtr] + [xte[c]] * dose)
    y_aug = np.concatenate([ytr] + [yte[c]] * dose)
    contam = model_factory(name, seed).fit(x_aug, y_aug)

    def per_item(model):
        proba = model.predict_proba(xte)
        return proba[np.arange(N_TEST), yte]  # prob of the realized label

    return per_item(clean), per_item(contam), c


def collect_runs(name, pi, dose, seeds):
    """Train the clean/contaminated twins once per seed (deterministic)."""
    return [run_twin(name, s, pi, dose) for s in seeds]


def summarize(name, pi, dose, runs):
    """Aggregate a list of (y_clean, y_obs, c) runs; return summary + raw lifts."""
    lifts, headrooms, ratios = [], [], []
    viol, spill = [], []
    for y_clean, y_obs, c in runs:
        lift = y_obs[c] - y_clean[c]
        hr = 1 - y_clean[c]
        lifts.append(lift)
        headrooms.append(hr)
        ok = hr > 0.02
        ratios.append(lift[ok] / hr[ok])
        viol.append(float(np.mean(lift < -0.01)))
        spill.append(float(np.mean(np.abs(y_obs[~c] - y_clean[~c]))))
    lift = np.concatenate(lifts)
    hr = np.concatenate(headrooms)
    ratio = np.concatenate(ratios)
    # channel shape: per-headroom-bin 95th-percentile lift envelope
    bins = np.quantile(hr, [0, 0.25, 0.5, 0.75, 1.0])
    mids, envs = [], []
    for lo, hi_ in zip(bins[:-1], bins[1:]):
        sel = (hr >= lo) & (hr <= hi_)
        if sel.sum() >= 10:
            mids.append(float(hr[sel].mean()))
            envs.append(float(np.quantile(lift[sel], 0.95)))
    env_corr = float(np.corrcoef(mids, envs)[0, 1]) if len(mids) >= 3 else np.nan
    return {
        "model": name, "pi": pi, "dose": dose,
        "mean_lift": float(lift.mean()),
        "lambda_q95": float(np.quantile(ratio, 0.95)),
        "lambda_max": float(np.clip(ratio.max(), 0, 1)),
        "violation_rate": float(np.mean(viol)),
        "spillover_mad": float(np.mean(spill)),
        "envelope_headroom_corr": env_corr,
    }, (hr, lift)




def main():
    configs = (
        [(m, pi, 4) for m in ("logreg", "mlp", "rf") for pi in (0.05, 0.1, 0.2)]
        + [(m, 0.1, d) for m in ("logreg", "mlp", "rf") for d in (1, 16)]
    )
    rows = []
    scatter_data = None
    for name, pi, dose in configs:
        # train each twin ONCE per seed; derive full and calibration summaries
        # from the same cached runs (they are deterministic per seed)
        runs = collect_runs(name, pi, dose, range(N_SEEDS))
        summary, (hr, lift) = summarize(name, pi, dose, runs)
        cal_summary, _ = summarize(name, pi, dose, runs[:3])
        lam_hat = min(1.0, cal_summary["lambda_q95"] * 1.1)  # 10% safety pad
        covered = 0
        for y_clean, y_obs, _c in runs[3:]:
            lo, hi_ = identified_interval(y_obs, lam=lam_hat, pi=pi)
            covered += int(lo - 1e-9 <= y_clean.mean() <= hi_ + 1e-9)
        summary["lambda_hat_cal"] = lam_hat
        summary["coverage_h2r"] = covered / max(1, len(runs) - 3)
        rows.append(summary)
        if (name, pi, dose) == ("rf", 0.1, 16):
            scatter_data = (hr, lift, summary["lambda_q95"])
        print(f"  {name:7s} pi={pi:.2f} dose={dose:2d} | mean_lift {summary['mean_lift']:+.3f} "
              f"| lam_q95 {summary['lambda_q95']:.3f} | viol {summary['violation_rate']:.3f} "
              f"| spill {summary['spillover_mad']:.4f} | env_corr {summary['envelope_headroom_corr']:+.2f} "
              f"| H2r cov {summary['coverage_h2r']:.2f}")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "p2_contamctrl.csv", index=False)

    if scatter_data is not None:
        hr, lift, lam = scatter_data
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        ax.scatter(hr, lift, s=8, alpha=0.35, color="#219ebc",
                   label="contaminated items (RF, pi=0.1, dose=16)")
        xs = np.linspace(0, 1, 50)
        ax.plot(xs, lam * xs, color="#d62828", lw=2,
                label=f"A1 envelope  lift = {lam:.2f} x headroom")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel("headroom  1 - y* (clean twin)")
        ax.set_ylabel("real memorization lift  y_obs - y*")
        ax.set_title("E5: emergent lift channel vs the A1 envelope")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGS / "f12_channel_shape.png", dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    print("Phase-2 CONTAM-CTRL (small scale, real training):")
    main()
    print("done -> results/p2_contamctrl.csv, figures/f12_channel_shape.png")
