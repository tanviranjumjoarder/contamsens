"""CONFIRMATORY contamination re-audit of Open LLM Leaderboard claims.

Executes PREREGISTRATION.md v1.0 (frozen 19 Jul 2026, git tag
prereg-freeze-v1.0, commit ca7773b) verbatim:

  corpus     OLL v1 archive, 16 fixed candidate models x 5 tasks (SS2)
  claims     adjacent pairs per task ranking; pilot-overlap pairs flagged
  estimator  simple bound (binary single-draw), Gamma* over pi grid,
             primary rule: non-robust iff Gamma*(pi=0.1) < stratum Lambda_ref
  strata     HellaSwag 0.42; all other tasks pooled 0.355 (SS4)
  inference  fragility p (2,000 paired boots, seed 42), BH q=.05 primary,
             BY sensitivity; rho=0 primary, rho=0.2 sensitivity
  H4         >= 25% of claims non-robust (exact binomial CI), reported with
             and without pilot-overlap pairs

Label: protocol v1.0; OSF timestamp pending user filing (OSF_FILING.md).
Outputs -> results/confirmatory_audit.csv, results/confirmatory_summary.txt,
results/figures/f17_confirmatory.png. Downloads cache -> data/oll_audit/.
"""

from __future__ import annotations

import io
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from contamsens import bh_fdr, fragility_pvalue, gamma_star, margin  # noqa: E402
from fetch_oll import _extract_scores, _item_key_column  # noqa: E402
import hashlib  # noqa: E402

HUB = "https://huggingface.co"
ARCHIVE = "open-llm-leaderboard-old"
RESULTS = ROOT / "results"
CACHE = ROOT / "data" / "oll_audit"
CACHE.mkdir(parents=True, exist_ok=True)

MODELS = [  # PREREGISTRATION SS2, fixed a priori
    "meta-llama__Llama-2-7b-hf", "meta-llama__Llama-2-13b-hf",
    "meta-llama__Llama-2-70b-hf", "meta-llama__Meta-Llama-3-8B",
    "meta-llama__Meta-Llama-3-70B", "mistralai__Mistral-7B-v0.1",
    "mistralai__Mixtral-8x7B-v0.1", "Qwen__Qwen1.5-7B", "Qwen__Qwen1.5-14B",
    "Qwen__Qwen1.5-72B", "01-ai__Yi-34B", "google__gemma-7b",
    "microsoft__phi-2", "upstage__SOLAR-10.7B-v1.0", "EleutherAI__gpt-j-6b",
    "tiiuae__falcon-7b",
]
TASKS = {  # task -> (min plausible item count, stratum Lambda_ref)
    "arc:challenge|25": (1000, 0.355),
    "hellaswag|10": (9000, 0.42),
    "truthfulqa:mc|0": (600, 0.355),
    "winogrande|5": (1100, 0.355),
    "gsm8k|5": (1100, 0.355),
}
PI_GRID = (0.05, 0.1, 0.2, 0.3)
PI_PRIMARY = 0.1
PILOT_MODELS = {"meta-llama__Llama-2-7b-hf", "meta-llama__Llama-2-13b-hf",
                "mistralai__Mistral-7B-v0.1", "meta-llama__Meta-Llama-3-8B"}


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "contamsens/0.2"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return r.read()


_TREES: dict[str, list] = {}


def model_tree(model: str) -> list:
    if model not in _TREES:
        _TREES[model] = json.loads(_get(
            f"{HUB}/api/datasets/{ARCHIVE}/details_{model}/tree/main"
            f"?recursive=true"))
    return _TREES[model]


def fetch_items(model: str, task: str, min_items: int) -> pd.DataFrame | None:
    """Per-item (item, score) for one model on one task, cached; None + log
    reason if unavailable (dropped per SS2, no substitution)."""
    cache = CACHE / f"{model}__{task.replace(':', '_').replace('|', '_')}.csv"
    if cache.exists():
        return pd.read_csv(cache)
    try:
        marker = f"details_harness|{task}_"
        hits = sorted(t["path"] for t in model_tree(model)
                      if t["type"] == "file" and marker in t["path"]
                      and t["path"].endswith(".parquet"))
        if not hits:
            print(f"  DROP {model} on {task}: no parquet")
            return None
        best = None
        for fname in reversed(hits):
            url = (f"{HUB}/datasets/{ARCHIVE}/details_{model}/resolve/main/"
                   f"{urllib.parse.quote(fname)}")
            cand = pd.read_parquet(io.BytesIO(_get(url)))
            if best is None or len(cand) > len(best):
                best = cand
            if len(cand) >= min_items:
                best = cand
                break
        df = best
        out = pd.DataFrame({
            "item": [hashlib.sha1(str(x).encode("utf8")).hexdigest()[:16]
                     for x in _item_key_column(df)],
            "score": _extract_scores(df, model).astype(float),
        }).groupby("item", as_index=False)["score"].mean()
        out.to_csv(cache, index=False)
        return out
    except Exception as e:  # noqa: BLE001 -- log-and-drop per SS2
        print(f"  DROP {model} on {task}: {type(e).__name__}: {str(e)[:90]}")
        return None


def audit_task(task: str, lam_ref: float, min_items: int) -> pd.DataFrame:
    print(f"\n=== {task} (Lambda_ref {lam_ref}) ===", flush=True)
    per_model = {}
    for m in MODELS:
        df = fetch_items(m, task, min_items)
        if df is not None and len(df) >= min_items * 0.8:
            per_model[m] = df.set_index("item")["score"]
            print(f"  {m}: {len(df)} items, mean {df.score.mean():.3f}",
                  flush=True)
    if len(per_model) < 3:
        print(f"  task dropped: only {len(per_model)} models")
        return pd.DataFrame()
    # Item-universe consistency (SS2 drop-with-log rule): a model whose
    # snapshot uses a different example template shares no item hashes with
    # the rest and would annihilate the paired intersection. Drop any model
    # overlapping the modal item universe by < 90%.
    from collections import Counter
    cnt = Counter()
    for s in per_model.values():
        cnt.update(s.index)
    modal = {i for i, c in cnt.items() if c >= len(per_model) // 2}
    for m in list(per_model):
        ov = len(set(per_model[m].index) & modal)
        if ov < 0.9 * len(modal):
            print(f"  DROP {m}: item-universe mismatch "
                  f"(overlap {ov}/{len(modal)} with modal)")
            del per_model[m]
    if len(per_model) < 3:
        print(f"  task dropped after universe check: {len(per_model)} models")
        return pd.DataFrame()
    shared = None
    for s in per_model.values():
        shared = s.index if shared is None else shared.intersection(s.index)
    wide = pd.DataFrame({m: s.loc[shared] for m, s in per_model.items()})
    print(f"  paired items: {len(shared)} across {len(per_model)} models")
    order = wide.mean().sort_values(ascending=False).index.tolist()

    rows = []
    for a, b in zip(order[:-1], order[1:]):
        ya, yb = wide[a].to_numpy(), wide[b].to_numpy()
        delta = margin(ya, yb)
        g = {f"gamma_star_pi{p}": gamma_star(ya, yb, p, simple=True)
             for p in PI_GRID}
        g_primary = g[f"gamma_star_pi{PI_PRIMARY}"]
        rows.append({
            "task": task, "model_a": a, "model_b": b,
            "n_items": len(shared), "margin": delta, **g,
            "lam_ref": lam_ref,
            "non_robust_primary": bool(g_primary < lam_ref),
            "non_robust_rho02": bool(
                gamma_star(ya, yb, PI_PRIMARY, simple=True, rho=0.2) < lam_ref),
            "fragility_p": fragility_pvalue(
                ya, yb, lam_ref, PI_PRIMARY, n_boot=2000, simple=True, seed=42),
            "pilot_overlap": bool(task.startswith("arc")
                                  and {a, b} <= PILOT_MODELS),
        })
    return pd.DataFrame(rows)


def main() -> None:
    frames = [audit_task(t, lam, n) for t, (n, lam) in TASKS.items()]
    out = pd.concat([f for f in frames if len(f)], ignore_index=True)
    out["certified_robust_bh"] = bh_fdr(out["fragility_p"].to_numpy(), 0.05)
    out["certified_robust_by"] = bh_fdr(out["fragility_p"].to_numpy(), 0.05,
                                        method="by")
    out.to_csv(RESULTS / "confirmatory_audit.csv", index=False)

    lines = ["CONFIRMATORY AUDIT — protocol v1.0 (frozen 19 Jul 2026, "
             "tag prereg-freeze-v1.0, commit ca7773b); OSF timestamp pending",
             "=" * 76]
    for label, sub in (("all claims", out),
                       ("excl. pilot-overlap", out[~out.pilot_overlap])):
        k, n = int(sub.non_robust_primary.sum()), len(sub)
        bt = binomtest(k, n, 0.25, alternative="greater")
        ci = bt.proportion_ci(confidence_level=0.95, method="exact")
        lines.append(
            f"H4 [{label}]: {k}/{n} non-robust = {k / n:.1%} "
            f"(95% CI [{ci.low:.1%}, {ci.high:.1%}]); "
            f"H4 (>=25%) one-sided p = {bt.pvalue:.4g}")
    k2 = int(out.non_robust_rho02.sum())
    lines.append(f"rho=0.2 sensitivity: {k2}/{len(out)} non-robust "
                 f"= {k2 / len(out):.1%}")
    lines.append(f"FDR-certified robust: BH {int(out.certified_robust_bh.sum())}"
                 f"/{len(out)}, BY {int(out.certified_robust_by.sum())}/{len(out)}")
    for t, sub in out.groupby("task"):
        lines.append(f"  {t}: {int(sub.non_robust_primary.sum())}/{len(sub)} "
                     f"non-robust; median Gamma*(0.1) = "
                     f"{sub[f'gamma_star_pi{PI_PRIMARY}'].median():.3f}")
    summary = "\n".join(lines)
    (RESULTS / "confirmatory_summary.txt").write_text(summary, encoding="utf8")
    print("\n" + summary)

    # f17: per-task Gamma*(0.1) vs the stratum reference
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    tasks = list(out.task.unique())
    rng = np.random.default_rng(0)
    for i, t in enumerate(tasks):
        sub = out[out.task == t]
        g = sub[f"gamma_star_pi{PI_PRIMARY}"].clip(upper=1.2)
        x = i + rng.uniform(-0.13, 0.13, len(sub))
        colors = np.where(sub.non_robust_primary, "#d62828", "#2a9d8f")
        ax.scatter(x, g, c=colors, s=34, alpha=0.85, zorder=3)
        ax.hlines(sub.lam_ref.iloc[0], i - 0.3, i + 0.3, color="#ffb703",
                  lw=2.5, zorder=2)
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels([t.split("|")[0] for t in tasks], fontsize=9)
    ax.set_ylabel(r"$\Gamma^*(\pi = 0.1)$")
    ax.set_title("Confirmatory audit: red = not contamination-robust at the "
                 "frozen $\\Lambda_{ref}$ (gold bars)")
    fig.tight_layout()
    fig.savefig(RESULTS / "figures" / "f17_confirmatory.png", dpi=200)
    plt.close(fig)
    print("\nfigure -> results/figures/f17_confirmatory.png")


if __name__ == "__main__":
    main()
