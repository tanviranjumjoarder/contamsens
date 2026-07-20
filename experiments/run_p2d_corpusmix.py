"""P2d: does corpus dilution pull the measured Lambda toward the field range?

The 24-config grid measured Lambda under CONCENTRATED exposure (every training
token is a leaked item): a ceiling. Published field priors sit at [0.10, 0.45].
The corpus-mix run holds (model, pi, dose) fixed and dilutes each leaked copy
with `mix` neutral wikitext chunks per leaked chunk -- interpolating toward the
pretraining regime. If Lambda-hat falls from the ceiling toward the field range
as mix grows, the calibration table's central range stops being an assumption
and becomes an interpolated measurement.

Metrics per config are IDENTICAL to run_p2c_lora_analysis.py (lambda_q95,
violation_rate, spillover, envelope corr) so columns are comparable.

Also reports the SESSION-REPRODUCTION control: mix=0 duplicates two grid
configs; y_clean and the contamination index must match exactly (twin + seed
integrity), while y_obs differs by GPU training nondeterminism -- quantified
here as the noise floor against which mix effects are judged.

Outputs -> results/p2d_corpusmix.csv, results/figures/f19_corpusmix.png
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
RESULTS = ROOT / "results"
SRC = RESULTS / "contamctrl_output" / "contamctrl_corpusmix_peritem.csv"
GRID = RESULTS / "contamctrl_lora_peritem.csv"

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

FIELD_LO, FIELD_HI = 0.10, 0.45  # calibrated field range (SS8)


def config_metrics(g: pd.DataFrame) -> dict:
    """Same estimators as run_p2c_lora_analysis.py, applied to one config."""
    leaked = g[g.contaminated == 1]
    clean = g[g.contaminated == 0]
    lift = (leaked.y_obs - leaked.y_clean).to_numpy()
    hr = (1.0 - leaked.y_clean).to_numpy()
    ok = hr > 0.05
    ratio = np.clip(lift[ok] / hr[ok], 0, None)
    bins = np.quantile(hr, [0, 0.25, 0.5, 0.75, 1.0])
    mids, envs = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (hr >= lo) & (hr <= hi)
        if sel.sum() >= 5:
            mids.append(float(np.mean(hr[sel])))
            envs.append(float(np.quantile(lift[sel], 0.95)))
    env_corr = (float(np.corrcoef(mids, envs)[0, 1])
                if len(mids) >= 3 else np.nan)
    return {
        "n_leaked": int(len(leaked)),
        "mean_lift_leaked": float(lift.mean()),
        "lambda_q95": float(np.quantile(ratio, 0.95)),
        "violation_rate": float((lift < -0.01).mean()),
        "spillover_eps": float((clean.y_obs - clean.y_clean).abs().mean()),
        "spillover_net": float((clean.y_obs - clean.y_clean).mean()),
        "envelope_headroom_corr": env_corr,
    }


def main() -> None:
    d = pd.read_csv(SRC)
    grid = pd.read_csv(GRID)

    # ---- session-reproduction control (mix=0 vs the committed grid) --------
    print("SESSION-REPRODUCTION CONTROL (mix=0 vs 18-Jul grid, same seed)")
    noise = {}
    for dose in sorted(d[d["mix"] == 0].dose.unique()):
        a = d[(d.dose == dose) & (d["mix"] == 0)].sort_values("item_id")
        b = grid[(grid.model == "Qwen/Qwen2.5-1.5B") & (grid.pi == 0.1)
                 & (grid.dose == dose)].sort_values("item_id")
        assert np.allclose(a.y_clean, b.y_clean), "twin drift!"
        assert (a.contaminated.values == b.contaminated.values).all(), \
            "contamination index drift!"
        la = (a[a.contaminated == 1].y_obs - a[a.contaminated == 1].y_clean)
        lb = (b[b.contaminated == 1].y_obs - b[b.contaminated == 1].y_clean)
        noise[dose] = abs(float(la.mean()) - float(lb.mean()))
        lam_a = config_metrics(a)["lambda_q95"]
        lam_b = config_metrics(b)["lambda_q95"]
        noise[f"lam{dose}"] = abs(lam_a - lam_b)
        print(f"  dose {int(dose)}: y_clean EXACT, c-index EXACT; "
              f"mean-lift delta across sessions = {noise[dose]:.4f}; "
              f"lambda_q95 delta = {noise[f'lam{dose}']:.4f} "
              f"(GPU train nondeterminism)")
    floor = max(v for k, v in noise.items() if not str(k).startswith("lam"))
    lam_floor = max(v for k, v in noise.items() if str(k).startswith("lam"))
    print(f"  => session-noise floor: {floor:.4f} on mean lift, "
          f"{lam_floor:.4f} on lambda_q95\n")

    # ---- per-config metrics ------------------------------------------------
    rows = []
    for (model, pi, dose, mix), g in d.groupby(["model", "pi", "dose", "mix"]):
        rows.append({"model": model, "pi": pi, "dose": int(dose),
                     "mix": int(mix), **config_metrics(g)})
    out = pd.DataFrame(rows).sort_values(["dose", "mix"]).reset_index(drop=True)
    out.to_csv(RESULTS / "p2d_corpusmix.csv", index=False)

    cols = ["dose", "mix", "mean_lift_leaked", "lambda_q95", "violation_rate",
            "spillover_net", "envelope_headroom_corr"]
    print("DILUTION TABLE (Qwen2.5-1.5B, pi=0.1)")
    print(out[cols].to_string(index=False,
                              float_format=lambda x: f"{x:+.4f}"))

    # ---- the dilution question --------------------------------------------
    print("\nDILUTION READING")
    for dose in sorted(out.dose.unique()):
        sub = out[out.dose == dose].sort_values("mix")
        lams = sub.lambda_q95.tolist()
        in_field = [FIELD_LO <= l <= FIELD_HI for l in lams]
        print(f"  dose {dose}: lambda_q95 {' -> '.join(f'{l:.3f}' for l in lams)}"
              f"  (mix 0 -> 4 -> 20); in field range: "
              f"{' -> '.join(str(b) for b in in_field)}")

    # ---- figure ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.axhspan(FIELD_LO, FIELD_HI, color="#2a9d8f", alpha=0.15, zorder=0,
               label="calibrated field range [0.10, 0.45]")
    colors = {1: "#264653", 4: "#d62828"}
    for dose in sorted(out.dose.unique()):
        sub = out[out.dose == dose].sort_values("mix")
        ax.plot(sub["mix"], sub.lambda_q95, "o-", lw=2.2, ms=7,
                color=colors.get(dose, "#777"),
                label=f"dose {dose} (measured $\\hat\\Lambda_{{q95}}$)")
    ax.set_xlabel("mix = neutral wikitext chunks per leaked chunk")
    ax.set_ylabel(r"$\hat\Lambda_{q95}$  (lift / headroom, 95th pct)")
    ax.set_xticks([0, 4, 20])
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8.5, loc="upper right")
    ax.set_title("Dilution is not one thing: single exposures decay into the\n"
                 "field range; repeated exposures consolidate toward $\\Lambda=1$",
                 fontsize=10.5)
    ax.grid(alpha=0.25, zorder=0)
    fig.tight_layout()
    fig.savefig(RESULTS / "figures" / "f19_corpusmix.png", dpi=200)
    plt.close(fig)
    print("\nfigure -> results/figures/f19_corpusmix.png")


if __name__ == "__main__":
    main()
