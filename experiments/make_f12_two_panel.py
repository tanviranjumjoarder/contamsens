"""Regenerate f12 as a two-panel figure: total vs partial memorization.

Left:  RF, dose=16  -- the degenerate extreme (lift = 1.00 x headroom exactly).
Right: logreg, dose=16 -- the instructive case: A1 is an UPPER ENVELOPE over a
       scatter that includes negative lifts (the U1 two-sided motivation).

Reuses run_twin from run_p2_contamctrl.py (same seeds). Seconds on CPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from run_p2_contamctrl import run_twin  # noqa: E402

FIGS = ROOT / "results" / "figures"


def collect(name, dose, seeds=range(3)):
    hrs, lifts = [], []
    for s in seeds:
        y_clean, y_obs, c = run_twin(name, s, 0.1, dose)
        hrs.append(1 - y_clean[c])
        lifts.append(y_obs[c] - y_clean[c])
    return np.concatenate(hrs), np.concatenate(lifts)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for ax, (name, title) in zip(
        axes,
        [("rf", "Random forest (dose 16): total memorization"),
         ("logreg", "Logistic regression (dose 16): partial, with violations")],
    ):
        hr, lift = collect(name, 16)
        ok = hr > 0.02
        lam = float(np.clip(np.quantile(lift[ok] / hr[ok], 0.95), 0, 1))
        ax.scatter(hr, lift, s=7, alpha=0.3, color="#219ebc",
                   label="contaminated items (3 seeds)")
        xs = np.linspace(0, 1, 50)
        ax.plot(xs, lam * xs, color="#d62828", lw=2,
                label=f"A1 envelope: lift = {lam:.2f} x headroom")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel("headroom  1 - y* (clean twin)")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, loc="upper left")
    axes[0].set_ylabel("real memorization lift  y_obs - y*")
    fig.suptitle("E5: the A1 channel is an upper envelope, not a line", y=1.0)
    fig.tight_layout()
    fig.savefig(FIGS / "f12_channel_shape.png", dpi=200)
    print("f12 regenerated (two-panel)")


if __name__ == "__main__":
    main()
