"""Is the confirmatory headline an artifact of auditing ADJACENT pairs?

Adjacent pairs in a ranking are selected to have the smallest margins, so a
sceptic will argue the 84.5% non-robust figure is mechanically induced by the
claim-selection rule rather than by contamination sensitivity. This script
answers that directly: it recomputes the audit over ALL pairs of models per
task (the full comparison set, no selection) and over rank-gap strata, using
the same frozen rule (non-robust iff margin < pi * Lambda_ref).

Reads the cached per-item scores in data/oll_audit/ written by
run_confirmatory_audit.py -- no network access, no refitting.

Outputs -> results/pair_selection_check.csv, results/pair_selection_check.txt
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
CACHE = ROOT / "data" / "oll_audit"
RESULTS = ROOT / "results"

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

PI = 0.1
LAM = {"arc:challenge|25": 0.355, "hellaswag|10": 0.42,
       "winogrande|5": 0.355, "gsm8k|5": 0.355}
# models dropped by the frozen protocol, per task (see the audit log)
DROP = {"arc:challenge|25": {"tiiuae__falcon-7b"},
        "winogrande|5": {"EleutherAI__gpt-j-6b"}}


def load(task: str) -> dict[str, pd.Series]:
    tag = task.replace(":", "_").replace("|", "_")
    out = {}
    for f in CACHE.glob(f"*__{tag}.csv"):
        model = f.name.split("__" + tag)[0]
        if model in DROP.get(task, set()):
            continue
        df = pd.read_csv(f)
        out[model] = df.set_index("item")["score"]
    return out


def main() -> None:
    rows = []
    for task, lam in LAM.items():
        per = load(task)
        if len(per) < 3:
            print(f"skip {task}: {len(per)} models cached")
            continue
        shared = None
        for s in per.values():
            shared = s.index if shared is None else shared.intersection(s.index)
        wide = pd.DataFrame({m: s.loc[shared] for m, s in per.items()})
        order = wide.mean().sort_values(ascending=False).index.tolist()
        rank = {m: i for i, m in enumerate(order)}
        thresh = PI * lam
        for a, b in itertools.combinations(order, 2):
            delta = float(wide[a].mean() - wide[b].mean())
            gap = rank[b] - rank[a]
            rows.append({"task": task, "model_a": a, "model_b": b,
                         "rank_gap": gap, "margin": delta,
                         "adjacent": gap == 1,
                         "non_robust": bool(delta < thresh),
                         "thresh": thresh})
    d = pd.DataFrame(rows)
    d.to_csv(RESULTS / "pair_selection_check.csv", index=False)

    adj = d[d.adjacent]
    allp = d
    lines = [
        "IS THE HEADLINE AN ARTIFACT OF AUDITING ADJACENT PAIRS?",
        "=" * 64,
        "Same frozen rule (non-robust iff margin < pi*Lambda_ref, pi=0.1).",
        "",
        f"ADJACENT pairs only : {int(adj.non_robust.sum())}/{len(adj)} "
        f"= {adj.non_robust.mean():.1%} non-robust",
        f"ALL pairs           : {int(allp.non_robust.sum())}/{len(allp)} "
        f"= {allp.non_robust.mean():.1%} non-robust",
        "",
        "By rank gap (gap 1 = adjacent):",
    ]
    for g, sub in d.groupby("rank_gap"):
        if g > 8:
            continue
        lines.append(f"    gap {g:<2d}: {int(sub.non_robust.sum()):3d}/"
                     f"{len(sub):3d} = {sub.non_robust.mean():6.1%} non-robust"
                     f"   (median margin {sub.margin.median():.3f})")
    big = d[d.rank_gap >= 5]
    lines += [
        "",
        f"Distant pairs (rank gap >= 5): {int(big.non_robust.sum())}/{len(big)}"
        f" = {big.non_robust.mean():.1%} non-robust",
        "",
        "Reading: adjacency IS a selection on small margins, and the",
        "all-pairs rate is necessarily lower. The scientific claim is not",
        "that most model COMPARISONS are fragile -- distant pairs are mostly",
        "robust, which is the 'ordering of eras' finding -- but that the",
        "comparisons a leaderboard actually invites, between neighbouring",
        "entries, are the fragile ones. Both numbers are reported so the",
        "selection is explicit rather than buried.",
    ]
    txt = "\n".join(lines)
    (RESULTS / "pair_selection_check.txt").write_text(txt, encoding="utf8")
    print(txt)


if __name__ == "__main__":
    main()
