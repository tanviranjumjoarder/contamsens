# contamsens — Contamination Sensitivity Analysis for Benchmark Claims

**How contaminated would the benchmark have to be for this leaderboard claim to be wrong?**

Partial identification and sensitivity analysis for ML evaluation under unmeasured
data contamination. Instead of trying to *detect* contamination (impossible for
closed-weight models), we *bound* its effect: every claim gets a **contamination
robustness value Γ\*** — the minimum contamination strength that would overturn it.

*Precision on "reference-free":* **computing** Γ\* needs only the published
scores — no training corpus, reference models, or rephrased benchmarks.
**Interpreting** Γ\* against a plausible contamination strength uses the public
calibration table (`results/lambda_priors.csv`: literature clean/dirty gaps plus
our own injection measurements), shared across all audits rather than built
per-audit. Reporting the full Γ\*(Λ, π) curve requires no calibration at all.

**Pre-registered confirmatory result.** Applied to the Open LLM Leaderboard v1
archive under a protocol frozen in advance, **49 of 58 adjacent leaderboard
claims (84.5%) are not contamination-robust** at the calibrated contamination
strength (two-sided 95% CI [72.6%, 92.7%]; one-sided *p* = 2.7×10⁻²¹ against the
pre-registered 25% threshold). Only 8 claims earn FDR-controlled robustness
certificates, every one of them with a margin above 5 points. The leaderboard's
*ordering of eras* is contamination-robust; its *ordering of neighbours* is not.

Pre-registration: <https://osf.io/2436z> (DOI `10.17605/OSF.IO/2436Z`, 20 July
2026), with the analysis library and calibration constants frozen beforehand at
git tag `prereg-freeze-v1.0` (commit `ca7773b`). `PREREGISTRATION.md`,
`THEORY.md`, and `results/lambda_priors.csv` are byte-identical to that tag;
every post-freeze change is enumerated in
[OSF_ADDENDUM_chain_of_custody.md](OSF_ADDENDUM_chain_of_custody.md).

Status: theory, library (57 tests), ground-truth validation at two scales,
Λ calibration, and the pre-registered confirmatory audit are complete. See
[THEORY.md](THEORY.md) for the formal note, [paper/](paper/) for the manuscript
and proofs appendix, and [PROVENANCE.md](PROVENANCE.md) for the mapping from
every reported number to the script that produced it.

---

## The story (explain it like I'm five)

Imagine your class has a big spelling test on Friday, and the teacher uses it to
decide who is the **best speller in class**.

Now imagine some kids might have *seen some of the test questions before* — maybe
a page of the test fell out of the teacher's bag last week and got passed around.
Nobody knows who saw it. Nobody knows how many pages fell out. The kids who saw it
aren't telling, and you **can't look inside their heads** to check.

Rafi scores 92. Mina scores 89. The teacher announces: "Rafi is the best speller!"

But wait. What if Rafi saw some questions early and Mina didn't? Then Rafi's 92
isn't really a 92 — it's partly real spelling skill and partly *remembering*.

For years, everyone tried to solve this one way: **catch the peeking**. Search
bags. Ask around. Compare homework. But you can never search *every* bag, and the
biggest, fanciest kids keep their bags locked. So the argument never ends:
"He peeked!" — "No I didn't!" — "Prove it!" — "You prove it!" Nobody can.

This research says: **stop trying to catch the peeking. Ask a different question.**

> *"How MUCH peeking would it take to change the answer?"*

Here's the trick. Rafi beat Mina by 3 points. Peeking can only *help* your score,
and only by so much — seeing a question early doesn't make you a wizard, it just
lifts you a bit on that one question. So we can calculate:

- If peeking lifts you a **tiny** bit, 3 points is a big lead. Rafi is really the
  best, peeking or no peeking. The claim is **safe**.
- If peeking lifts you a **lot**, 3 points could be fake. We honestly **can't
  tell** who is the best speller. The claim is **fragile**.

The line between safe and fragile is one number. We call it **Γ\*** (say:
"gamma star"). Every claim — "this model beats that model" — gets its own Γ\*:

> **Γ\* = the smallest amount of peeking that would flip the answer.**

Big Γ\* → the claim survives even a lot of peeking → trust it.
Small Γ\* → a whisper of peeking flips it → don't build anything important on it.

And the beautiful part: to compute Γ\* you **never need to search anyone's bag**.
You only need the scores — which everyone already publishes. So it works on the
locked bags too.

One more trick: how much does peeking *actually* lift a kid's score? We can
measure that! Use questions written **after** the test was locked in the safe —
nobody could have peeked at those. Compare kids' scores on old questions vs.
brand-new questions of the same difficulty. The gap tells you how strong peeking
really is in the wild. That real number is what we compare every Γ\* against.

Grown-up translation: kids = models, test = benchmark, peeking = training-data
contamination, locked bags = closed-weight models, the safe = the training
cutoff date, and "who is best" = every leaderboard on the internet.

Doctors solved this exact problem 40 years ago for medicines ("how much hidden
bias would it take to fake this drug result?" — Rosenbaum bounds, E-values).
Machine learning just never borrowed the idea. We're borrowing it.

---

## What's in the box

```
THEORY.md                     Formal note: estimand, CSM(Λ,π), sharp bounds, Γ*
src/contamsens/
  csm.py                      The contamination sensitivity model
  bounds.py                   Sharp + simple partial-identification bounds
  gamma_star.py               Γ* per claim, contamination frontier
  inference.py                Paired item bootstrap (sampling ⊕ contamination)
  leaderboard.py              Audit a whole leaderboard DataFrame
tests/                        Unit tests (hand-computed cases, invariants)
experiments/run_validation.py Ground-truth validation: injected contamination
results/                      Generated tables + figures
```

## Quick start

```bash
pip install -e .[experiments,test]
python reproduce.py                  # tests + all fast experiments (~1 min)
python reproduce.py --full           # + the Phase-2 twin-training grid (~30 min)
```

Minimal API:

```python
import numpy as np
from contamsens import identified_interval, gamma_star, audit

scores_a = np.array([...])  # per-item scores in [0,1], model A
scores_b = np.array([...])  # per-item scores in [0,1], model B

lo, hi = identified_interval(scores_a, lam=0.2, pi=0.1)  # clean-score bounds for A
g = gamma_star(scores_a, scores_b, pi=0.1)               # min Λ that overturns "A beats B"

# whole-leaderboard audit with FDR-certified robustness (pre-registered procedure)
table = audit(df, pi=0.1, lam_ref=0.2, n_boot=1000)      # df: model,item,score rows
```

**Regimes are auto-detected** (`simple=None` default): single-draw binary
scores route to the simple population bound — the sharp per-item machinery is
provably vacuous there (ceiling effect; it would certify *every* claim) — and
binary draws with `1 < m < 10` follow the E9 prescription. Continuous
per-item scores get the sharp knapsack. Two-sided contamination via `rho`,
provenance strata via `max_bias_stratified`, source-level leakage via
`max_bias_grouped`, selection-bounded budgets via `max_bias_selection`.

## Reproducibility

Global seed 42. `requirements.txt` is the exact-version results lock
(`pyproject.toml` carries loose bounds). Every number in `results/` maps to a
script via `PROVENANCE.md`. CPU-only except the optional Kaggle LoRA notebook
(`notebooks/`). `python reproduce.py` re-derives everything fast in ~1 minute.
