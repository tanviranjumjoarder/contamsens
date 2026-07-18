"""Phase-2c: analyze the REAL LLM-scale CONTAM-CTRL run (Kaggle LoRA output).

Input: results/contamctrl_lora_peritem.csv, produced by
notebooks/contamctrl_lora_kaggle.ipynb on a Kaggle T4 (QUICK or full grid).
Per row: one MMLU item's clean-twin and contaminated-twin score (softmax prob
of the correct choice letter -- continuous, regime R1) for one (model, pi,
dose) configuration.

Per configuration this script measures every CSM parameter at LLM scale:
  lambda_q95 / lambda_max   A1 envelope strength (lift / headroom quantiles)
  violation_rate            share of leaked items HURT (U1 / rho evidence)
  spillover_eps             mean |drift| on clean items (U4)
  envelope_headroom_corr    channel-shape check (E5)
plus a CROSS-CONFIG consistency check: the identified interval built with
lambda-hat from the OTHER configuration (out-of-sample within the run) must
cover the clean twin's score.

Outputs -> results/p2c_lora_llm_scale.csv, figures/f15_llm_channel.png,
and 'measured' rows appended to results/lambda_priors.csv (pre-freeze:
PREREGISTRATION.md SS4 anticipates CONTAM-CTRL measurements entering the
prior table BEFORE the OSF freeze / unblinding).
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

from contamsens import identified_interval_twosided  # noqa: E402

RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"
IN = RESULTS / "contamctrl_lora_peritem.csv"


def summarize_config(g: pd.DataFrame) -> dict:
    c = g[g.contaminated]
    clean = g[~g.contaminated]
    lift = (c.y_obs - c.y_clean).to_numpy()
    hr = (1 - c.y_clean).to_numpy()
    ok = hr > 0.02
    ratio = lift[ok] / hr[ok]
    bins = np.quantile(hr, [0, 0.25, 0.5, 0.75, 1.0])
    mids, envs = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (hr >= lo) & (hr <= hi)
        if sel.sum() >= 10:
            mids.append(float(hr[sel].mean()))
            envs.append(float(np.quantile(lift[sel], 0.95)))
    env_corr = float(np.corrcoef(mids, envs)[0, 1]) if len(mids) >= 3 else np.nan
    return {
        "n_items": len(g),
        "n_leaked": int(c.shape[0]),
        "mean_lift_leaked": float(lift.mean()),
        "lambda_q95": float(np.quantile(ratio, 0.95)),
        "lambda_max": float(np.clip(ratio.max(), 0, 1)),
        "violation_rate": float((lift < -0.01).mean()),
        "spillover_eps": float((clean.y_obs - clean.y_clean).abs().mean()),
        "spillover_net": float((clean.y_obs - clean.y_clean).mean()),
        "envelope_headroom_corr": env_corr,
    }


def main() -> None:
    raw = pd.read_csv(IN)
    configs = list(raw.groupby(["model", "pi", "dose"]))
    rows = []
    for (model, pi, dose), g in configs:
        s = summarize_config(g)
        s.update({"model": model, "pi": pi, "dose": dose})
        rows.append(s)
    df = pd.DataFrame(rows)

    # cross-config consistency: lambda-hat and eps-hat from the OTHER config
    checks = []
    for i, ((model, pi, dose), g) in enumerate(configs):
        other = df.drop(index=i)
        same_model = other[other.model == model]
        if same_model.empty:
            checks.append(np.nan)
            continue
        lam_hat = min(1.0, float(same_model.lambda_q95.mean()) * 1.1)
        eps_hat = float(same_model.spillover_eps.mean())
        theta = g.y_clean.mean()
        lo, hi = identified_interval_twosided(
            g.y_obs.to_numpy(), lam_hat, 0.0, pi, spillover=eps_hat
        )
        checks.append(int(lo <= theta <= hi))
    df["covers_clean_crossconfig"] = checks
    df.to_csv(RESULTS / "p2c_lora_llm_scale.csv", index=False)

    cols = ["model", "pi", "dose", "mean_lift_leaked", "lambda_q95",
            "violation_rate", "spillover_eps", "envelope_headroom_corr",
            "covers_clean_crossconfig"]
    print(df[cols].round(4).to_string(index=False))

    # channel-shape figure at LLM scale (largest-pi config)
    (model, pi, dose), g = max(configs, key=lambda kv: kv[0][1])
    c = g[g.contaminated]
    lift = (c.y_obs - c.y_clean).to_numpy()
    hr = (1 - c.y_clean).to_numpy()
    lam = df[(df.model == model) & (df.pi == pi) & (df.dose == dose)
             ]["lambda_q95"].iloc[0]
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.scatter(hr, lift, s=9, alpha=0.35, color="#219ebc",
               label=f"leaked MMLU items ({model.split('/')[-1]}, "
                     f"pi={pi}, dose={dose})")
    xs = np.linspace(0, 1, 50)
    ax.plot(xs, min(lam, 1.0) * xs, color="#d62828", lw=2,
            label=f"A1 envelope: lift = {min(lam, 1.0):.2f} x headroom")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("headroom  1 - y* (clean twin)")
    ax.set_ylabel("real LLM memorization lift  y_obs - y*")
    ax.set_title("E5 at LLM scale: LoRA-injected contamination vs the A1 envelope")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGS / "f15_llm_channel.png", dpi=200)
    plt.close(fig)

    # dose-response figure (f16): the LLM-scale dose law, per model
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    colors = {"Qwen/Qwen2.5-1.5B": "#023047", "Qwen/Qwen2.5-0.5B": "#d62828"}
    for (model, pi), sub in df.groupby(["model", "pi"]):
        sub = sub.sort_values("dose")
        c = colors.get(model, "#888")
        axes[0].plot(sub.dose, 100 * sub.mean_lift_leaked, "o-", color=c,
                     alpha=0.55, lw=1.4)
        axes[1].plot(sub.dose, sub.lambda_q95.clip(upper=1.0), "o-", color=c,
                     alpha=0.55, lw=1.4)
    for ax, ylab, title in (
        (axes[0], "mean lift on leaked items (pp)",
         "raw lift: dose 1 is NET NEGATIVE"),
        (axes[1], r"$\hat\Lambda_{q95}$ (headroom units)",
         "lambda: capacity ordering restored"),
    ):
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 4, 16], labels=["1", "4", "16"])
        ax.set_xlabel("dose (epochs over leaked items)")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=10)
        ax.axhline(0, color="k", lw=0.8)
    import matplotlib.lines as mlines
    axes[0].legend(handles=[
        mlines.Line2D([], [], color=c, label=m.split("/")[-1])
        for m, c in colors.items()], fontsize=8)
    fig.suptitle("LLM-scale dose-response (lines = pi levels)", y=1.0)
    fig.tight_layout()
    fig.savefig(FIGS / "f16_dose_response.png", dpi=200)
    plt.close(fig)

    # 'measured' rows in the prior table: aggregated per (model, dose) over pi
    # (pre-freeze; CONTAM-CTRL rows are replaced wholesale on rerun)
    priors = pd.read_csv(RESULTS / "lambda_priors.csv")
    priors = priors[~priors["source"].str.startswith("CONTAM-CTRL", na=False)]
    agg = df.groupby(["model", "dose"]).agg(
        lift=("mean_lift_leaked", "mean"),
        lam_lo=("lambda_q95", "min"), lam_hi=("lambda_q95", "max"),
        viol=("violation_rate", "mean"), spill=("spillover_eps", "mean"),
        n_pi=("pi", "nunique"),
    ).reset_index()
    new = pd.DataFrame([{
        "source": "CONTAM-CTRL LoRA (this work)",
        "link": "notebooks/contamctrl_lora_kaggle.ipynb",
        "model": r.model,
        "benchmark": "MMLU (4 subjects)",
        "evidence": f"fp32 LoRA injection, dose={r.dose}, "
                    f"aggregated over {r.n_pi} pi levels",
        "lift_pp": round(100 * r.lift, 1),
        "clean_score_pp": "",
        "headroom_pp": "",
        "lambda_est": round(min(1.0, 0.5 * (r.lam_lo + min(r.lam_hi, 1.0))), 3),
        "quality": "measured",
        "notes": f"lambda_q95 range [{r.lam_lo:.2f}, {min(r.lam_hi, 1.0):.2f}]; "
                 f"mean violation_rate={r.viol:.3f}; mean spillover={r.spill:.4f}; "
                 f"concentrated-exposure ceiling",
    } for r in agg.itertuples()])
    pd.concat([priors, new], ignore_index=True).to_csv(
        RESULTS / "lambda_priors.csv", index=False
    )
    print(f"\n{len(new)} aggregated 'measured' rows written into lambda_priors.csv")


if __name__ == "__main__":
    main()
