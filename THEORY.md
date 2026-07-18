# Contamination Sensitivity Analysis — Formal Note (Phase 0)

*Working draft, v0.1 — 16 July 2026. This is the two-page note the whole project
gates on. Everything here is implemented in `src/contamsens` and stress-tested in
`experiments/run_validation.py`.*

---

## 1. Setup and estimand

A benchmark has items $i = 1, \dots, n$. Fix a model $m$ (suppressed where clear).

- $y_i^\* \in [0,1]$ — the **clean (counterfactual) score**: the per-item score the
  model would obtain had item $i$ (and its near-duplicates) never appeared in its
  training data.
- $y_i \in [0,1]$ — the **observed score**.
- $c_i \in \{0,1\}$ — latent contamination indicator ($c_i = 1$: item $i$ leaked
  into training).

**Estimand.** The clean benchmark score
$$\theta \;=\; \frac{1}{n}\sum_{i=1}^n y_i^\*.$$

The leaderboard reports $\hat\mu = \frac1n \sum_i y_i$ and treats it as $\theta$.
That identification requires $c \equiv 0$ (or $c \perp y^\*$ with zero lift), which
is exactly what contamination violates. Since $c$ is unobservable — for
closed-weight models, unobservable *in principle* — $\theta$ is **partially
identified**, and the honest object to report is its identified set.

## 2. The Contamination Sensitivity Model CSM(Λ, π)

Two axioms, indexed by sensitivity parameters $\Lambda \in [0,1]$ (lift strength)
and $\pi \in [0,1]$ (contaminated fraction):

**(A1) Monotone bounded lift.** For every item,
$$0 \;\le\; y_i - y_i^\* \;\le\; c_i \,\Lambda\, (1 - y_i^\*).$$
Contamination never hurts, and closes at most a fraction $\Lambda$ of the item's
remaining headroom. $\Lambda = 0$: contamination is harmless. $\Lambda = 1$: a
leaked item may be fully memorized ($y_i = 1$ regardless of skill).

**(A2) Budget.** $\frac1n \sum_i c_i \le \pi.$

Interpretation of $\Lambda$: *what fraction of the way to a perfect score does
leakage carry you on a leaked item.* It is the quantity the matched
pre/post-cutoff calibration (Phase 3) estimates from data.

**Remark (ceiling effect).** Under A1 with $\Lambda < 1$, an observed $y_i = 1$
forces $y_i^\* = 1$: the multiplicative-headroom channel cannot reach the ceiling
from below. Consequence: **realized single-draw binary scores must not be fed to
the per-item machinery** (a correct answer would wrongly count as
uncontaminatable). Binary benchmarks are handled at the probability level — §5.

## 3. Sharp partial identification of θ

For a contaminated item, invert A1: $y_i \le \Lambda + (1-\Lambda) y_i^\*$ gives
the per-item floor
$$\ell_i(\Lambda) \;=\; \max\!\Big(0,\; \frac{y_i - \Lambda}{1 - \Lambda}\Big),
\qquad y_i^\* \in [\ell_i(\Lambda),\, y_i],$$
and the **per-item deflation capacity**
$$d_i(\Lambda) \;=\; y_i - \ell_i(\Lambda) \;=\;
\begin{cases}
y_i, & y_i \le \Lambda,\\[2pt]
\dfrac{\Lambda\,(1 - y_i)}{1 - \Lambda}, & y_i > \Lambda.
\end{cases}$$

$d_i$ is continuous, maximized at $y_i = \Lambda$ with value $\Lambda$, and
nondecreasing in $\Lambda$ pointwise.

**Proposition 1 (sharp identified set).** Under CSM(Λ, π) with per-item scores
observed, the identified set for $\theta$ is the interval
$$\theta \;\in\; \big[\,\hat\mu - B(\Lambda,\pi),\;\; \hat\mu\,\big],
\qquad
B(\Lambda,\pi) \;=\; \frac1n \sum_{i \in \mathrm{top}\text{-}k}\; d_i(\Lambda),
\quad k = \lceil \pi n \rceil \wedge n,$$
where top-$k$ ranks items by $d_i(\Lambda)$.

*Proof sketch.* Any $(c, y^\*)$ consistent with the data has
$\theta = \hat\mu - \frac1n\sum_{i: c_i = 1} (y_i - y_i^\*) \ge \hat\mu - B$ since
each contaminated item deflates by at most $d_i$ and at most $k$ items are
contaminated; picking the $k$ largest $d_i$ maximizes total deflation. The lower
endpoint is attained by contaminating exactly the argmax set with
$y_i^\* = \ell_i$; the upper endpoint by $c \equiv 0$; intermediate values by
continuously raising $y^\*$ on the contaminated set. ∎

Budget rounding uses $\lceil \pi n\rceil$ (conservative).

**Corollary 1 (simple distribution-free bound).** Since $\max_y d(y) = \Lambda$,
$$B(\Lambda, \pi) \;\le\; \pi\,\Lambda
\qquad\Longrightarrow\qquad
\text{identified-set width} \;\le\; \pi\Lambda .$$
The worst per-unit-budget bias is exactly $\Lambda$, attained at items with
observed score $\Lambda$. (Sharp when the score distribution places mass at
$\Lambda$; otherwise the knapsack bound of Prop. 1 is strictly tighter.)

**Corollary 2 (reduction).** At $\Lambda = 0$ or $\pi = 0$ the set collapses to
$\{\hat\mu\}$ and everything reduces to standard (sampling-only) evaluation —
e.g. Rank-Intervals-style analysis is the $\Lambda = 0$ special case.

## 4. Ranking claims and Γ\*

Claim: **A ≻ B** with observed margin $\hat\Delta = \hat\mu_A - \hat\mu_B > 0$.

Contamination only inflates, so the worst case for the claim is *A contaminated,
B clean* (contamination of B only lowers B's clean score, strengthening the
claim). Hence:

**Proposition 2 (robustness condition).** Under CSM(Λ, π) applied to A, the claim
A ≻ B holds over the entire identified set iff
$$\hat\Delta \;>\; B_A(\Lambda, \pi).$$

**Definition (contamination robustness value).**
$$\Gamma^\*(\pi) \;=\; \inf\{\Lambda \in [0,1] : B_A(\Lambda,\pi) \ge \hat\Delta\},$$
with $\Gamma^\*(\pi) = \infty$ ("robust") if $B_A(1,\pi) < \hat\Delta$ — note
$B_A(1,\pi)$ is the mean of the top-$k$ observed scores of A, so a sufficiently
large margin survives even full memorization of the top-$k$ items.

$B_A$ is continuous and nondecreasing in $\Lambda$, so $\Gamma^\*$ is
well-defined and computable by bisection. Under Corollary 1 the closed form
$$\Gamma^\*_{\text{simple}}(\pi) \;=\; \hat\Delta / \pi$$
gives the **contamination frontier**: the hyperbola $\pi\Lambda = \hat\Delta$
separating (π, Λ) into "claim survives" and "claim overturnable" regions.
Prop. 1's knapsack version shifts this frontier outward (more robust) whenever
few items sit near $y \approx \Lambda$.

*Reading:* Γ\* is to contamination what the E-value is to unmeasured confounding.
"Γ\* = 0.4 at π = 0.1" reads: *to overturn this claim, at least 10% of items must
be leaked AND leakage must close ≥40% of the headroom on each.* Compare against
the empirically calibrated Λ distribution to judge plausibility.

## 5. Measurement regimes

**(R1) Continuous / repeated-measure items** — per-item score is a mean over
$k \gg 1$ draws (pass@k with samples, multi-seed accuracy, MASE-based skill
scores in $[0,1]$): Prop. 1 applies directly to the per-item means (with
sampling noise handled in §6).

**(R2) Single-draw binary items** — only $y_i \in \{0,1\}$, one draw. Model at
the probability level: $p_i = p_i^\* + c_i\delta_i$, $\delta_i \le
\Lambda(1-p_i^\*)$, $\theta = \frac1n\sum p_i^\*$. Per-item floors are not
identified (a single draw carries no information about $p_i$ beyond its sign),
and the sharp population bound degenerates to the simple bound:
$$\theta \;\in\; [\hat\mu - \pi\Lambda,\; \hat\mu ] \quad(\text{up to sampling error}).$$

**Degeneracy note (audit fix, 17 Jul 2026).** On single-draw binary data the
per-item machinery is not merely inappropriate — it is *vacuous with a
discontinuity*: $d(0) = d(1) = 0$ for every $\Lambda < 1$, while at
$\Lambda = 1$ the deflation jumps to $y_i$ itself. A sharp-bound $\Gamma^\*$
computed on such data therefore returns exactly $1.0$ (or $\infty$),
falsely reporting "only total memorization could flip this claim" when the
honest simple bound may say $0.2$. The library now auto-detects binary-like
scores (`is_binary_like`), routes to the simple bound (`simple=None`
default), and warns loudly if the sharp path is forced.

**Honesty note (product-only identification in R2).** In this regime $\Lambda$
and $\pi$ enter the bound only through their product: the data cannot separate
"many items, weak lift" from "few items, strong lift," and the 2-D
$(\pi,\Lambda)$ frontier is a reparametrized 1-D family. The frontier earns its
second dimension only where the knapsack is active (regime R1) or where the
budget is stratified (§9 U3, post-cutoff strata with $\pi_s = 0$). State this
plainly in the paper; the constructive answer is the design-sensitivity
prescription — repeated draws and dated item strata are what buy back the
second identification dimension.

**Structural takeaway:** richer per-item measurement *strictly sharpens
identification* — repeated draws buy back the knapsack. This is a design lever
(cf. design sensitivity, Phase 5), and an argument benchmarks should publish
per-item, multi-draw results.

*Open refinement (P1):* in R1 with finite $k$, ranking by noisy $d_i(\hat p_i)$
makes the plug-in knapsack slightly anti-conservative (selection on noise).
Options: shrink $\hat p_i$ first, or bootstrap the whole functional (§6), or use
the simple bound which is immune. Quantified in the validation experiment.

## 6. Statistical layer

Prop. 1 is deterministic given per-item scores; sampling of items adds ordinary
uncertainty. Joint (sampling ⊕ contamination) interval: paired nonparametric
bootstrap over items (resampling item indices jointly across models, preserving
pairing and, where present, item-source clusters), applied to the *entire*
functional — $\hat\mu$, $B(\Lambda,\pi)$, $\hat\Delta$, $\Gamma^\*$:
$$\Big[\; q_{\alpha/2}\big(\hat\mu^{(b)} - B^{(b)}\big),\;\; q_{1-\alpha/2}\big(\hat\mu^{(b)}\big) \;\Big],$$
and a percentile CI for $\Gamma^\*$. At $\Lambda = 0$ this is the usual bootstrap
CI (Corollary 2 again). FDR control across a corpus of claims via
Benjamini–Hochberg on the per-claim robustness tests.

## 7. The vacuousness gate (Phase-0 kill criterion)

The framework is useful only if the identified sets are **informative at
plausible sensitivity values** — neither so wide that nothing is ever robust,
nor so narrow that everything is.

Analytically: width $\le \pi\Lambda$. With literature-plausible ranges
($\Lambda \in [0.05, 0.3]$ lift strength, $\pi \in [0.05, 0.5]$ leaked fraction)
the width spans **0.25 to 15 points**. Meanwhile top-leaderboard margins are
commonly **0.5–3 points** and cross-tier margins **5–15+ points**. So the
machinery sits exactly in the discriminating regime: *small margins between
frontier models will typically not be contamination-robust; large margins will
be; Γ\* separates the two claim populations.* Neither degenerate outcome occurs.

**Gate verdict: PASS** (analytic; confirmed empirically in
`results/gate_analysis.csv` — see PROVENANCE.md). The asymmetric case (one model
post-dated by the benchmark, hence clean by construction) tightens bounds
further, exactly as anticipated in the roadmap.

## 8. What Phase 1–3 add (not in this note)

- Paraphrase/near-duplicate channel: $\Lambda$ decomposed into verbatim and
  semantic components with different calibrations.
- Covariate-refined budgets: $\pi$ varying with item age/popularity strata
  (tightens the knapsack via stratified budgets).
- Λ calibration from matched pre/post-cutoff item pairs, with its own
  uncertainty propagated into Γ\*.
- Design sensitivity: choose item-freshness mix to maximize $\Gamma^\*$ at fixed
  budget.
- CONTAM-CTRL: LoRA-injected ground-truth contamination for external validity of
  A1 (is the monotone-headroom channel the right shape for real memorization?).

## 9. v0.2 upgrades adopted from the red-team review (16 July 2026)

Three model upgrades, each forced by a specific finding in the contamination
literature; implemented in `src/contamsens` and covered by tests.

**(U1) Two-sided lift: CSM±(Λ⁺, Λ⁻, π).** Empirical work shows contamination
can *depress* observed scores (post-training dynamics, format mismatch,
memorized-but-stale answers; see Impact-of-Post-training, arXiv 2601.06103;
Generalization-or-Memorization, arXiv 2402.15938). Axiom A1 generalizes to
$$-\Lambda^-\, y_i^\* \;\le\; y_i - y_i^\* \;\le\; \Lambda^+ (1 - y_i^\*),$$
i.e. leakage may also erase up to a fraction $\Lambda^-$ of true skill. The
per-item **inflation capacity** of a deflated item is
$u_i(\Lambda^-) = \min\!\big(1 - y_i,\; y_i \Lambda^- / (1-\Lambda^-)\big)$
(from $y_i^\* \le y_i/(1-\Lambda^-)$, capped at 1), and the identified set
becomes $[\hat\mu - B(\Lambda^+\!,\pi),\; \hat\mu + U(\Lambda^-\!,\pi)]$ with
$U$ the mean of the top-$\lceil\pi n\rceil$ inflation capacities. For a claim
A ≻ B the worst case is now *A inflated and B deflated*:
robust iff $\hat\Delta > B_A(\Lambda^+\!,\pi) + U_B(\Lambda^-\!,\pi)$.
$\Gamma^\*$ is reported along the ray $\Lambda^- = \rho\,\Lambda^+$ with
$\rho = 0$ (v0.1, monotone) and $\rho = 0.2$ (deflation stress) as defaults;
$\rho$ is a stated modelling choice, calibratable from the same dose-response
literature that calibrates $\Lambda^+$.

**(U2) Group-level budgets.** Real leakage happens at dataset/source
granularity — a whole benchmark file is scraped, not random items. With items
partitioned into groups $g$ and contamination all-or-nothing per group, the
adversary solves a knapsack (weights = group sizes, values = group deflation
totals, capacity $\lceil\pi n\rceil$). We bound it by the **fractional
relaxation** (greedy by deflation density), which is a valid upper bound on
the adversary's power, hence a conservative identified set; exact DP is
feasible when the number of groups is moderate. Group structure can only
*constrain* the adversary, so the grouped bound is never wider than the
item-level bound at the same $\pi$ — source metadata tightens identification
for free.

**(U3) Stratified budgets and model-stratified Λ.** Items released after a
model's training cutoff have $c_i = 0$ *by construction*: the budget becomes
$\pi_s$ per stratum with $\pi_{\text{post-cutoff}} = 0$, tightening the
knapsack exactly as the roadmap's "asymmetric case" anticipated, now at item
granularity. Dose-response evidence (arXiv 2601.04301: contamination lift
grows with model capacity and shrinks with unique-corpus size and
overtraining) implies $\Lambda$ is **model-stratified**, not global; the
calibration protocol therefore emits $\Lambda_m$ ranges per capacity/corpus
class, and published effect estimates (ConStat, NeurIPS 2024; Magar &
Schwartz 2022; dose-response curves) serve as meta-analytic priors.

## 10. Phase-1 estimation theory (validated 16 July 2026, `run_p1_theory.py`)

**Proposition 3 (bridge to Rosenbaum-type selection models).** Let contamination
propensities $q_i = P(c_i = 1)$ satisfy the budget $\bar q = \pi$ and the
selection-odds band $\mathrm{odds}(q_i)/\mathrm{odds}(q_j) \le \Gamma_{sel}^2$.
The worst-case expected bias $B(\Lambda, \pi, \Gamma_{sel})$ is attained by a
two-point propensity assigning $q_{hi}$ to the largest deflation capacities
(LP extreme point, swept over the split), and satisfies
$$B(\Lambda,\pi,1) = \pi\,\bar d \;\le\; B(\Lambda,\pi,\Gamma_{sel})
\;\le\; B(\Lambda,\pi,\infty) = \text{(Prop. 1 top-}k\text{)},$$
nondecreasing and continuous in $\Gamma_{sel}$. **Interpretation:** the v0.1
adversarial bound is the $\Gamma_{sel}=\infty$ endpoint of a Rosenbaum-style
family; $\Gamma_{sel}$ measures how strongly leakage can *select* on an item's
usefulness to the leaderboard claim. Random scraping is $\Gamma_{sel}=1$
(bias $= \pi\bar d$, typically far smaller). Empirically the two endpoints
differ by ~1.5× at reference settings (fig. f9), so a defensible
$\Gamma_{sel}$ assumption buys real tightening. Implemented as
`max_bias_selection` (vectorized bisection, exact to 2⁻⁶⁰).

**Theorem 4 (consistency and rate of $\hat\Gamma^*$).** $\hat B(\lambda,\pi)$
is an L-statistic (trimmed upper mean of bounded transforms); under iid item
sampling and continuity of the $d$-distribution at its $(1-\pi)$-quantile,
$\hat B \to B$ a.s., uniformly in $\lambda$ (monotone + continuous limit).
With $B$ strictly increasing at $\Gamma^*$ and $\Gamma^*$ interior,
$\hat\Gamma^* = \hat B^{-1}(\hat\Delta)$ is consistent and
$\sqrt n$-asymptotically normal (Hadamard differentiability of trimmed
L-functionals + delta method); the paired item bootstrap of §6 is therefore
valid for $\Gamma^*$. *Empirical check:* RMSE slope on log-log axes
$-0.518$ vs theoretical $-0.5$; bias $\le 0.0025$ at every $n$ from 100 to
25,600 (`p1_t2_consistency.csv`, fig. f10).

**E9 (finite draws per item).** With $m$ draws per item, the plug-in knapsack
$\hat B$ carries two opposing biases: Jensen smoothing ($d$ is a minimum of
two linear functions, hence concave, so noise shrinks apparent capacities)
and selection-on-noise (top-$k$ of noisy values inflates). Measured net bias
(`p1_t3_finite_draws.csv`, fig. f11): **Jensen dominates at small $m$**
($-40\%/-26\%/-14\%$ at $m=2$ for $\lambda = 0.1/0.3/0.6$), crossing to a
mild $+4\%$ around $m=10$ and $<1\%$ by $m=50$. Joint (sampling ⊕
identification) intervals maintain $\ge 99.3\%$ coverage throughout; the
simple bound $\pi\Lambda$ — structurally immune to both effects — achieves
100%. **Prescription:** use the sharp knapsack only when $m \ge 10$ draws
per item are published; below that, fall back to the simple bound. This is
also a design-sensitivity lever: *publishing $\ge 10$ draws per item is what
entitles a benchmark to sharp identification.*

## 11. Phase-2 CONTAM-CTRL findings: the channel model is now empirical
*(small scale, real training — `run_p2_contamctrl.py`, `run_p2b_spillover_fix.py`;
full LLM scale pending `notebooks/contamctrl_lora_kaggle.ipynb`)*

Contamination here is **emergent, not injected**: a fraction $\pi$ of test
items (with realized labels, repeated `dose` times) is planted in the training
corpus of clean/contaminated model twins (logistic regression / MLP / random
forest — a capacity spectrum), and the lift arises from the models' own
memorization. Findings:

1. **A1's shape is right.** The 95th-percentile lift envelope tracks headroom
   with correlation $+0.91$ to $+1.00$ in all 15 configurations (fig. f12:
   the envelope is an upper bound over a scatter, not a line).
2. **Λ is capacity- and dose-stratified**, as U3 assumed: logreg
   $\hat\Lambda \in [0.02, 0.23]$ (monotone in dose 1→16), MLP $\approx 0.96$,
   RF $= 1.00$ (an interpolating learner memorizes fully — the $\Lambda = 1$
   extreme is real, not hypothetical). Small-scale caveat: leaked items are a
   large fraction of these training sets; LLM-scale $\Lambda$ (literature:
   0.1–0.45) is diluted by corpus size, exactly per the dose-response law.
3. **Monotonicity violations are real**: up to 27% of leaked items score
   *lower* in the contaminated twin (low-capacity models, high dose) —
   direct empirical justification for the two-sided channel (U1, ρ > 0).
4. **(U4) The spillover channel.** Training on leaked items drifts scores on
   *uncontaminated* items (mean $|$drift$|$: logreg 0.003–0.04, RF ~0.03,
   MLP 0.13–0.20) — an interference effect outside A1, which caused real
   coverage failures (1/3–2/3) for low-capacity twins. Remedy, now in the
   library: widen the interval by a calibrated $\varepsilon$
   (`identified_interval_twosided(..., spillover=eps)`), where $\varepsilon$
   is the mean absolute clean-item drift measured on calibration runs.
   **With $\hat\varepsilon$, coverage is 3/3 in every previously failing
   configuration** (`p2b_spillover_fix.csv`). At LLM scale $\varepsilon$
   should shrink toward zero with corpus size (leaked fraction → 0); the
   Kaggle notebook measures it directly.

The channel model CSM±(Λ⁺, Λ⁻, π, Γ_sel, ε) is therefore no longer an
axiom system — each component now has a measured, falsifiable counterpart:
Λ (envelope slope), ρ (violation rate), Γ_sel (selection structure), ε
(spillover), π (budget/provenance strata).

**LLM-scale full grid (17–18 Jul 2026, fp32, Kaggle T4;
`run_p2c_lora_analysis.py`, figs f15–f16).** Qwen2.5-1.5B and
Qwen2.5-0.5B (same-family pair — capacity isolated), 905 MMLU items,
π ∈ {0.05, 0.1, 0.25, 0.5} × dose ∈ {1, 4, 16}; **all 24 configs complete**
(the resumed session reproduced the prior 20 bit-exactly), y_clean
cross-config spread exactly 0 (twin integrity held through every adapter
unload):

| dose | mean lift (leaked) | Λ̂_q95 | violation rate | ε̂ (spillover) |
|---|---|---|---|---|
| 1 | **−2.9 to +1.3 pt (negative in 7/8 configs)** | 0.14–0.55 | **31–56%** | 0.04–0.08 |
| 4 | +3.1 to +23.4 pt | 0.53–0.99 | 4–31% | 0.06–0.14 |
| 16 | +34 to +56 pt | **1.000 (all 8)** | **≤ 0.9% (7 of 8 exactly 0)** | 0.20–0.30 |

Five conclusions. (i) **The dose law at LLM scale**: a *single* exposure to
test items is net-harmful — the disruption channel dominates before
memorization sets in — while dose 16 is complete memorization
(Λ̂ = 1.000, zero violations, every config). (ii) **ρ is dose-dependent**:
violation rates fall monotonically 31–56% → 4–31% → 0. Light contamination
is *predominantly* two-sided; heavy contamination is purely one-sided. The
U1 channel is the story of the realistic low-dose regime, not a corner
case. (iii) **The headroom parametrization is vindicated by the capacity
comparison**: raw lift *anti*-orders with capacity (0.5B gains +44/+51 pt
at dose 16 vs 1.5B's +34/+38) purely because the smaller model has more
headroom; in Λ units the ordering flips back to match the dose-response
law. Raw clean/dirty gaps mislead; Λ = lift/headroom compares correctly
across models. (iv) **A1's envelope holds where there is signal** (corr up
to +0.99 at dose ≥ 4; at dose 1 the correlation degenerates, 0.08–0.99,
because near-zero lift leaves no envelope to fit — reported, not hidden).
(v) **Spillover grows with dose** (0.04 → 0.30) and stays positive
(format-familiarity); cross-config consistency checks pass 21/24, and all
three failures are exactly the extreme dose-16, high-π configs checked with
λ̂ pooled across doses — direct evidence that **Λ calibration must be dose-
and model-stratified** (U3), which the pre-registration already requires.
Measured Λ̂ remains the concentrated-exposure ceiling; the corpus-mixed
injection interpolating toward the field prior [0.1, 0.45] is the
outstanding experiment.

**Positioning note (ConStat).** ConStat (arXiv 2405.16281) also works from
performance alone and *estimates* contamination effects — but requires
reference benchmarks (rephrased/synthetic) and reference models assumed
clean, and answers a model-level detection question. CSM answers a
*claim-level identification* question with no reference assets, and consumes
ConStat's estimates as one $\Lambda$-calibration source. The two are
complements, and ConStat is the natural strong baseline in the re-audit.
