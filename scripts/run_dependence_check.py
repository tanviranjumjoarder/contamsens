"""Does H4 survive the dependence between adjacent claims?

The exact binomial test in the confirmatory audit treats the 58 claims as
independent Bernoulli draws. They are not: within a task, adjacent claims
chain through shared models ((m1,m2),(m2,m3),...), and the same 16 models
recur across tasks. This script quantifies how much the H4 conclusion
depends on that assumption, three ways:

1. ALTERNATING SPLIT. Within each task, odd-indexed and even-indexed
   adjacent claims share no model. Testing H4 separately on each subset
   uses only claims that are model-disjoint within tasks.
2. WORST SINGLE TASK. The most conservative clustering treats each task as
   one unit; report the per-task rates so the reader sees no task carries
   the result alone.
3. EFFECTIVE SAMPLE SIZE. The smallest n at the observed 84.5% rate for
   which the one-sided exact test against 25% still rejects at alpha=0.05.
   If dependence deflated the information in 58 claims to n_eff, H4 stands
   for any n_eff at or above this number.

Reads results/confirmatory_audit.csv. Outputs ->
results/dependence_check.txt (appended to PROVENANCE).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass


def h4(k: int, n: int) -> float:
    return binomtest(k, n, 0.25, alternative="greater").pvalue


def main() -> None:
    d = pd.read_csv(RESULTS / "confirmatory_audit.csv")
    lines = ["H4 UNDER DEPENDENCE BETWEEN ADJACENT CLAIMS", "=" * 60]

    # claims appear in ranking order per task, so parity = chain parity
    d["idx_in_task"] = d.groupby("task").cumcount()
    for label, sub in (("odd  (model-disjoint within task)",
                        d[d.idx_in_task % 2 == 0]),
                       ("even (model-disjoint within task)",
                        d[d.idx_in_task % 2 == 1])):
        k, n = int(sub.non_robust_primary.sum()), len(sub)
        lines.append(f"  {label}: {k}/{n} = {k / n:.1%} non-robust, "
                     f"one-sided p vs 25% = {h4(k, n):.3g}")

    lines.append("  per task:")
    for t, sub in d.groupby("task"):
        k, n = int(sub.non_robust_primary.sum()), len(sub)
        lines.append(f"    {t}: {k}/{n} = {k / n:.1%}  (p = {h4(k, n):.3g})")

    k_all, n_all = int(d.non_robust_primary.sum()), len(d)
    rate = k_all / n_all
    n_eff = next(n for n in range(3, n_all + 1)
                 if h4(round(rate * n), n) < 0.05)
    lines += [
        f"  observed rate {rate:.1%} ({k_all}/{n_all})",
        f"  minimal effective sample size for alpha=0.05: n_eff = {n_eff}",
        f"  (dependence would have to deflate 58 claims below {n_eff} "
        "independent claims' worth of information to lose significance)",
    ]
    txt = "\n".join(lines)
    (RESULTS / "dependence_check.txt").write_text(txt, encoding="utf8")
    print(txt)


if __name__ == "__main__":
    main()
