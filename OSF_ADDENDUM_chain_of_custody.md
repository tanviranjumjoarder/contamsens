# Chain-of-custody addendum

Filed alongside `PREREGISTRATION.md` v1.0-FREEZE. This note states precisely
what was frozen at the tag and what was written afterwards, so that the freeze
claim can be audited exactly rather than taken on trust.

Prepared 20 July 2026. Verification commands are given so any reader with the
repository can reproduce every claim below.

## 1. Frozen at git tag `prereg-freeze-v1.0`

Commit `ca7773b57b385e5f3c605fba0f66fd4ef773787d`, committed 2026-07-19
00:17:46 +0600. The following were verified byte-identical (git blob SHA-1)
between the tag and the filing date:

| File | Role |
|---|---|
| `PREREGISTRATION.md` | protocol, hypothesis H4, corpus, estimator, decision rule |
| `THEORY.md` | estimator definitions referenced by §3 |
| `results/lambda_priors.csv` | the frozen Λ calibration table of §4 |
| `src/contamsens/{csm,bounds,gamma_star,inference,leaderboard}.py` | estimator implementations |

`results/lambda_priors.csv` has blob hash
`590d29523466e11525d72ef6992039dcd5e0e75b`. It encodes the frozen calibration
constants Λ_ref = 0.355 (pooled median of the eight quality-eligible estimates)
and Λ_ref = 0.42 (HellaSwag stratum).

The estimator library frozen at the tag implements Γ\*, the simple and sharp
bounds, fragility p-values, BH/BY FDR control, and Imbens–Manski intervals.
Verify with:

    git rev-parse prereg-freeze-v1.0:results/lambda_priors.csv
    git ls-tree -r --name-only prereg-freeze-v1.0 -- src

## 2. Written AFTER the freeze — full disclosure

`scripts/run_confirmatory_audit.py`, the driver that executes the frozen
protocol, was **not** in the tagged commit; it was committed afterwards in
`0e532e8`. It contains no estimator logic — it calls the frozen library with
the frozen constants. It is disclosed here explicitly because
`PREREGISTRATION.md` §2 references it by name, and a reader inspecting the tag
will not find it there.

Both post-freeze commits are disclosed:

1. **`0e532e8`** — audit execution, plus **amendment 1**: an item-universe
   consistency rule (drop any model whose per-item snapshot overlaps the modal
   item universe by less than 90%). This was triggered by `tiiuae/falcon-7b` on
   ARC-Challenge, whose snapshot shares zero item hashes with the 1,170-item
   modal universe common to the other 15 models; without the rule the paired
   intersection is empty and the task cannot be analysed at all. The drop is
   printed in the run log.

2. **`564869b`** — reporting correction only. The summary printed a *one-sided*
   binomial confidence interval, an artifact of calling `proportion_ci` on a
   `binomtest(alternative="greater")`. It was corrected to the two-sided
   Clopper–Pearson interval that the manuscript reports. Per-claim results
   (`results/confirmatory_audit.csv`) and the figure reproduce bit-identically;
   no estimate, test, or decision changed.

## 3. Protocol deviations at execution

All were pre-specified in §2 as "drop with a logged reason, no substitutions":

- **`truthfulqa:mc|0` dropped entirely** — the v1 archive stores `mc1`/`mc2`
  rather than the frozen `acc_norm`/`acc` specification. Dropped per the frozen
  spec rather than silently substituting a different metric.
- **`EleutherAI/gpt-j-6b` dropped on `winogrande|5`** — malformed score column.
- **`tiiuae/falcon-7b` dropped on `arc:challenge|25`** — amendment 1 above.

## 4. Statement

No constant in §3 (estimator specification) or §4 (Λ calibration) was changed
at any point after the freeze. The primary decision rule, the π grid, ρ, the
bootstrap seed, the FDR level, the model list, and the task list are exactly as
filed.
