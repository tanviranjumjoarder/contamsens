# How Contaminated Would It Have To Be? Partial Identification and Sensitivity Analysis for Benchmark Claims Under Unmeasured Data Contamination

**Tanvir Anjum Joarder**¹
¹ Department of Computer Science and Engineering, Rajshahi University of Engineering & Technology, Rajshahi-6204, Bangladesh — `ussash93@gmail.com`

*Draft v0.2 — 20 July 2026. Every number in this draft is regenerable via
`python reproduce.py` and maps to a script through PROVENANCE.md. The
pre-registered confirmatory audit (§9.1) and the 24-configuration GPU-scale
CONTAM-CTRL grid (§7) are complete; the single remaining [PENDING] marker
awaits the corpus-mixed injection run.*

---

## Abstract

Leaderboard claims of the form "model A outperforms model B" are treated as
measurements of capability, but they are observational estimates whose
dominant error source — training-data contamination — is unmodeled. The
field's response has been *detection*, which requires access to training
corpora (impossible for closed-weight models), reference benchmarks, or
reference models assumed clean, and which cannot answer the only question
that matters: does the contamination change the conclusion? We import the
epistemology causal inference developed for unmeasured confounding: stop
trying to detect the bias and *bound* it. We formalize contamination as an
identification failure, introduce a Contamination Sensitivity Model
CSM±(Λ⁺, Λ⁻, π, Γ_sel, ε) whose every parameter has a measured empirical
counterpart, derive sharp partial-identification bounds for the
uncontaminated score, and define the **contamination robustness value Γ\***
— the minimum contamination strength that overturns a claim — computable
from published scores alone. The v0.1 adversarial bound arises as the
endpoint of a Rosenbaum-type selection family; Γ̂\* is consistent and
√n-normal, and a paired bootstrap with Benjamini–Hochberg control yields
FDR-certified robustness across a claim corpus. Ground-truth experiments
with *emergent* contamination (models genuinely trained on leaked items)
validate the assumed lift channel (envelope–headroom correlation ≥ 0.91 in
15/15 configurations), expose and repair two failure modes (two-sided lift;
spillover), and confirm coverage. On real data, every adjacent ranking claim
in a six-model time-series foundation-model benchmark is fragile
(Γ\* ≤ 0.114 at π = 0.1), and on Open LLM Leaderboard ARC-Challenge details
the method certifies exactly the claim it should (Llama-2-13B ≻ 7B,
Γ\* = 0.504, FDR-certified) while flagging frontier margins as fragile
(Γ\* ≤ 0.188). Freshness and repeated draws emerge as quantified design
levers. In a **pre-registered confirmatory audit** of 58 adjacent-pair
claims across 16 models and four Open LLM Leaderboard benchmarks,
**84.5% of claims are not contamination-robust** at the frozen calibration
(exact 95% CI [72.6%, 92.7%]; one-sided p = 2.7×10⁻²¹ against the
pre-registered 25% threshold), and only 8 claims — all with margins above
5 points — earn FDR-controlled robustness certificates. The finding is not
an artifact of the frozen constant: sweeping contamination strength across
its whole plausible range leaves the conclusion intact, and the hypothesis
clears its pre-registered bar for every Λ ≥ 0.055 — beneath the floor of the
calibrated field range. In the single-draw binary regime the rule provably
reduces to a *calibrated margin threshold*; we show why deriving that
threshold, rather than asserting one, is the contribution, and where richer
data makes the machinery bind strictly tighter. Sensitivity analysis
converts an unanswerable question — *is the benchmark contaminated?* — into
an answerable one, with a one-number reporting standard any leaderboard can
adopt.

**Keywords:** benchmark contamination; partial identification; sensitivity
analysis; evaluation methodology; leaderboards; Rosenbaum bounds.

---

## 1. Introduction

Every empirical claim in machine learning ultimately rests on a benchmark
score, and benchmark scores rest on an assumption nobody can verify: that
the test items did not leak into training. Contamination is documented and
material — clean/dirty gaps of 15.3 points on HellaSwag for LLaMA-70B
[Touvron et al., 2023], estimated gains up to 14 points on flagged subsets
[Llama 3, 2024], 13-point drops on a clean GSM8K mirror — yet the response
of both producers and auditors has been **detection**: n-gram overlap,
membership inference, kernel divergence, lineage audits [2404.00699;
2510.13654; 2502.00678]. Detection fails structurally three times over.
For closed-weight models the pretraining corpus is unobservable, so
detection is impossible *in principle*, not merely in practice. Detection
yields a contestable binary that vendors dispute and no protocol
adjudicates. And detection is silent on the decision-relevant question:
a benchmark can be contaminated with the ranking robust, or barely
contaminated with the ranking fragile — detection cannot tell these apart.

Meanwhile the statistical wing of evaluation reform quantifies the *wrong
error*. Rank intervals [2606.08679] propagate sampling uncertainty, which
shrinks with test-set size; contamination bias does not. A confidence
interval that excludes the dominant error source is not a confidence
interval.

Causal inference confronted exactly this epistemology half a century ago.
When a confounder cannot be measured, one does not detect it; one reports
**how strong it would have to be to overturn the conclusion** — Rosenbaum's
Γ, the marginal sensitivity model, the E-value [Rosenbaum, 1987; VanderWeele
& Ding, 2017; Chernozhukov et al., 2022]. This paper ports that machinery
to ML evaluation. The reduction is, once stated, almost obvious:
**contamination is unmeasured confounding between training-data membership
and measured skill, and benchmark scores are therefore partially
identified.** Nothing else in this paper is obvious; the port requires a new
sensitivity model (contamination acts through a monotone, bounded,
headroom-proportional channel — an assumption we *test* with real
memorization experiments, then twice repair), a new calibration strategy
(post-cutoff items are clean by construction), and a new design theory
(freshness and repeated draws purchase identification).

**Contributions.**
1. The reduction: a potential-outcomes formulation of benchmark scores under
   latent contamination, with the estimand (the clean score θ) and its
   identification failure stated precisely (§3).
2. The Contamination Sensitivity Model CSM±(Λ⁺, Λ⁻, π, Γ_sel, ε) and sharp
   partial identification of θ, including grouped (source-level),
   stratified (provenance-aware), selection-bounded (Rosenbaum-bridge), and
   spillover-robust variants; every axiom has a measured counterpart (§4).
3. The contamination robustness value **Γ\*** and the contamination
   frontier, the evaluation analogue of the E-value; a claim is reported
   with the minimum contamination strength that overturns it (§5).
4. Estimation theory: consistency and √n-normality of Γ̂\*, validity of the
   paired item bootstrap, a two-sided finite-draw bias analysis with a
   practical prescription (sharp bounds require ≥ 10 draws per item), and
   FDR-certified robustness across claim corpora (§6).
5. Ground-truth validation with **emergent** contamination — models
   actually trained on leaked items — which confirmed the channel shape,
   *falsified* strict monotonicity (up to 27% of leaked items are hurt),
   and surfaced a spillover channel; both repairs are part of the model,
   not the rhetoric (§7).
6. Empirically calibrated Λ priors from the published contamination
   literature, with a meta-analytic central range [0.1, 0.45] (§8).
7. Exploratory case studies on two real benchmarks — a six-model TSFM
   transfer atlas and Open LLM Leaderboard ARC-Challenge details — plus a
   pre-registered protocol for the confirmatory audit (§9).
8. Design sensitivity: quantified prescriptions for benchmark builders
   (§10), and an open, tested, one-command-reproducible library.

## 2. Related work

**Contamination detection and quantification.** Surveys catalogue detection
methods and document their fragile assumptions [2404.00699; 2410.18966;
2406.04244]. Magar & Schwartz [2022] introduced controlled-injection
quantification (memorization vs exploitation); dose-response studies show
lift grows with capacity and shrinks with unique-corpus size [2601.04301];
post-training can *reverse* lift [2601.06103]. ConStat [Dekoninck et al.,
NeurIPS 2024] is the strongest performance-only method: a statistical test
and effect estimate requiring reference benchmarks and reference models
assumed clean. CapBencher [2505.18102] builds overfitting alarms into new
benchmarks; inference-time decontamination [2601.19334] mitigates at
evaluation time for open models. **None of these answers the claim-level
question**; all require assets (references, model access, benchmark
redesign) that legacy claims and closed models do not offer. We consume
their effect estimates as calibration inputs and use ConStat as the natural
baseline and convergent-validity check.

**Leaderboard statistics.** Rank intervals [2606.08679] give
sampling-uncertainty-aware rankings — our Λ = 0 special case. Perturbation
and social-choice analyses [2605.15761; 2605.23628; 2402.01781] document
instability empirically without an inferential model. Psychometric and
efficiency work [2402.14992; 2501.17200] models items, not contamination.

**Causal sensitivity analysis.** Rosenbaum bounds, marginal sensitivity
models, E-values, and omitted-variable-bias frameworks [Rosenbaum 1987; Tan
2006; VanderWeele & Ding 2017; 2112.13398; 2403.14152; 2602.24261] supply
our machinery; none has been applied to evaluation. The construct-validity
programme [2511.04703; 2510.23191; 2604.03244] argues evaluation needs
measurement theory but supplies no estimator; we supply one.

## 3. Setup: the clean score is partially identified

A benchmark has items i = 1…n; fix a model m. Let y\*ᵢ ∈ [0,1] be the
**clean score** — the counterfactual per-item score had item i never
entered training — and yᵢ the observed score; cᵢ ∈ {0,1} the latent
contamination indicator. The estimand is θ = (1/n) Σ y\*ᵢ. Leaderboards
report μ̂ = (1/n) Σ yᵢ and treat it as θ, which requires c ≡ 0. Since items
popular enough to benchmark are popular enough to scrape — and that
popularity correlates with the skill being measured — c ⊥̸ y\*: textbook
confounding, with c unobservable. θ is partially identified; the honest
report is its identified set. (Full formalism: THEORY.md §1–2.)

## 4. The contamination sensitivity model

**CSM(Λ, π)** (one-sided core): (A1) 0 ≤ yᵢ − y\*ᵢ ≤ cᵢ·Λ·(1 − y\*ᵢ) —
leakage closes at most a fraction Λ of an item's headroom; (A2) mean(c) ≤ π.
Λ = 1 is full memorization; both extremes are *observed* in our experiments
(§7).

**Proposition 1 (sharp identified set).** θ ∈ [μ̂ − B(Λ,π), μ̂] with
B = the mean of the ⌈πn⌉ largest per-item deflation capacities
dᵢ(Λ) = yᵢ − max(0, (yᵢ−Λ)/(1−Λ)); the bound is attained. **Corollary 1:**
width ≤ πΛ, the worst per-unit-budget bias being exactly Λ at items with
yᵢ = Λ. **Corollary 2:** Λ=0 recovers sampling-only analysis (rank
intervals nest here).

**Regimes.** Continuous or repeated-measure per-item scores (R1) admit the
sharp knapsack. Single-draw binary scores (R2) do **not**: dᵢ(0)=dᵢ(1)=0
for all Λ<1 with a jump at Λ=1, so the sharp machinery is vacuous-with-a-
discontinuity — it reports "only total memorization flips this claim" —
and the honest bound is the simple population bound with width πΛ, where
(Λ, π) are identified only through their product. The library auto-detects
the regime; richer measurement *strictly sharpens identification*, a design
lever we quantify in §10.

**Empirically forced extensions** (each falsifiable, each measured in §7):
**U1 two-sided** — leakage can hurt (stale memorization, post-training
interference [2601.06103]): −Λ⁻y\*ᵢ ≤ yᵢ−y\*ᵢ; the worst case for A ≻ B
becomes *A inflated and B deflated*. **U2 grouped** — datasets leak whole;
all-or-nothing group budgets, bounded by the fractional knapsack, never
wider than item-level. **U3 stratified** — post-cutoff items have cᵢ = 0 by
construction; provenance metadata tightens bounds for free. **U4 spillover**
— training on leaked items drifts scores on clean items by a bounded ε,
widening the interval symmetrically; ε is measured by twin runs and shrinks
with corpus size. **Proposition 3 (bridge)** — bounding the *selection odds*
of contamination (Rosenbaum-style Γ_sel) interpolates continuously between
random contamination (bias πd̄) and the adversarial top-k (Γ_sel = ∞): the
adversarial bound is an endpoint of a classical family, and the endpoints
differ by ~1.5× at reference settings, so a defensible Γ_sel assumption
buys real tightening.

## 5. The contamination robustness value

For a claim A ≻ B with margin Δ̂, contamination only helps the worst case
run one way (A up, B down under U1), and the claim is **robust at
CSM(Λ,π)** iff Δ̂ exceeds the worst-case bias (Proposition 2).

**Definition.** Γ\*(π) = inf{Λ : worst-case bias ≥ Δ̂}, computed by
bisection (the bias is continuous and monotone in Λ); Γ\* = ∞ when the
claim survives even Λ = 1 within budget. In the simple regime the closed
form is the **contamination frontier** πΛ(1+ρ) = Δ̂.

Γ\* is to contamination what the E-value is to unmeasured confounding: one
number per claim, reportable in any results table. *Computing* Γ\* needs
only the published scores — no corpus, reference model, or reference
benchmark; *interpreting* it against plausible contamination strength uses
the shared public calibration table of §8 (or none at all, when the full
Γ\*(Λ, π) curve is reported). Per-model clean-score intervals are reported
at the parameter level via Imbens–Manski (2004), with set-level intervals
as the conservative alternative; corpus-level certificates use BH-FDR with
Benjamini–Yekutieli as the dependence-robust sensitivity.

## 6. Estimation

**Theorem 4.** B̂ is a trimmed-upper-mean L-statistic; under iid item
sampling and continuity at the (1−π)-quantile of the d-distribution, Γ̂\* is
consistent and √n-asymptotically normal, and the paired item bootstrap is
valid. Empirically the RMSE slope is −0.518 against the theoretical −0.5,
with bias ≤ 0.0025 from n = 100 to 25,600.

**Finite draws (E9).** With m draws per item, the plug-in knapsack carries
two *opposing* biases — Jensen smoothing (d is concave: a minimum of two
linear functions) versus selection-on-noise. Measured: −40% at m = 2,
crossing to +4% at m = 10, < 1% by m = 50. Joint sampling⊕identification
intervals maintain ≥ 99.3% coverage throughout; the simple bound, immune to
both effects, achieves 100%. **Prescription: sharp bounds require m ≥ 10
published draws per item.**

**Corpus-level inference.** Each claim receives a paired-bootstrap
fragility p-value (H₀: not robust at the reference CSM; add-one corrected);
Benjamini–Hochberg across the corpus yields **FDR-certified robustness**
(valid under the positive dependence induced by shared items).

## 7. Ground-truth validation with emergent contamination

Formula-injected validation is circular — the injection satisfies A1 by
construction. We therefore train **clean/contaminated twins** where a
fraction π of test items (with realized labels, repeated `dose` times) is
planted in the training corpus and the lift *emerges from real
memorization*, across a capacity spectrum (logistic regression, MLP,
random forest; 15 configurations × 6 seeds).

Findings. (i) **The channel shape is right**: the 95th-percentile lift
envelope tracks headroom with correlation +0.91 to +1.00 in 15/15
configurations. (ii) **Λ is capacity- and dose-stratified**: logistic
regression Λ̂ ∈ [0.02, 0.23] monotone in dose; MLP ≈ 0.96; random forest
= 1.00 — the Λ = 1 extreme is real (an interpolating learner), matching the
dose-response law at LLM scale [2601.04301]. (iii) **Monotonicity fails
non-trivially**: up to 27% of leaked items score *lower* in the
contaminated twin — the two-sided channel U1 is necessary. (iv) **Spillover
is real at small scale**: clean-item drift up to 0.20 (MLP) broke naive
coverage (1/3); with the calibrated ε-widening, coverage is **3/3 in every
previously failing configuration**. Injected-contamination coverage under
known (Λ, π) is 1.000 across all regimes, degrading informatively (to
0.10–0.34) only when the analyst *underestimates* contamination by half.
A blind synthetic leaderboard audit flags exactly the secretly contaminated
model's winning claim (Γ\* = 0.173 vs ≥ 0.29 for clean claims).

**At LLM scale** (same-family Qwen2.5-0.5B/1.5B pair, 905 MMLU items,
fp32 LoRA injection, the complete 24-configuration grid over
π ∈ {0.05…0.5} × dose ∈ {1, 4, 16}): the grid reveals a **dose law**. A
single exposure to test items is *net-harmful* — mean lift is negative in
7 of 8 dose-1 configurations (down to −2.9 pt) with individual-item
violation rates of 31–56% — while dose 16 is complete memorization
(Λ̂ = 1.000 in all eight configurations, with lifts up to +56 pt). The two-sided channel is therefore the
*dominant* behavior of light contamination, not a correction term. The
capacity comparison vindicates the headroom parametrization itself: raw
lift anti-orders with capacity (the 0.5B model gains more points, purely
from headroom), while Λ restores the correct ordering — raw clean/dirty
gaps mislead across models. Spillover grows with dose (0.04 → 0.30,
always positive: format familiarity), and all three (of 24) cross-config
consistency failures occur exactly where λ̂ was pooled across doses —
direct evidence for the dose-stratified calibration the pre-registration
mandates. Measured Λ̂ remains the concentrated-exposure ceiling above the
field range [0.1, 0.45]. [PENDING: the corpus-mixed injection
interpolating concentrated and diluted exposure.]

## 8. Calibrating Λ from the literature

Published clean/dirty comparisons convert directly to Λ = lift/headroom:
LLaMA-70B HellaSwag 15.3pt gap on 36.5pt headroom → Λ ≈ 0.42; GPT-3 SQuAD
≈ 0.40; Llama-3 flagged-subset estimates ≈ 0.40–0.47 (upper bounds by
design); GSM8K clean-mirror ≈ 0.21; C-Eval ≈ 0.31; MMLU ≈ 0.02 — strongly
task-heterogeneous, with a meta-analytic central range **[0.1, 0.45]**
(eighteen sourced rows in `results/lambda_priors.csv`, of which eight are
quality-eligible for the pooled median under the pre-registered policy of
§9.1; `measured` rows enter only as upper-bound anchors). Λ is model-stratified
per the dose-response law; ConStat effect estimates and our own CONTAM-CTRL
measurements enter the same table. The confirmatory audit interprets every
Γ\* against this pre-frozen range (PREREGISTRATION.md).

## 9. Case studies on real benchmarks (exploratory)

**TSFM transfer atlas** (6 frozen models × 17 datasets, continuous
MASE-skill, sharp regime, provenance strata from the corpus's own
contamination-prior annotations): **every adjacent ranking claim is
fragile** — Γ\* ≤ 0.114 at π = 0.1, at or below the calibrated range —
and stratification cannot rescue margins of ≤ 1.3 skill points because
14/17 datasets carry a high contamination prior.

**Open LLM Leaderboard, ARC-Challenge** (1,170 paired items × 4 models,
single-draw binary → auto simple regime, n_boot = 500): Llama-3-8B ≻
Mistral-7B (margin 0.26pt) has Γ\* = 0.026 — a whisker of contamination
flips it; Mistral-7B ≻ Llama-2-13B (1.9pt) has Γ\* = 0.188, inside the
calibrated range; **Llama-2-13B ≻ Llama-2-7B (5.0pt) has Γ\* = 0.504,
above the entire calibrated range, and is FDR-certified robust
(p = 0.006)** — precisely the claim that independent scaling evidence says
should survive. The instrument separates the claim population exactly as
designed.

### 9.1 The pre-registered confirmatory audit

Protocol frozen (PREREGISTRATION.md v1.0, 19 July 2026, git tag
`prereg-freeze-v1.0`) before any confirmatory Γ\* was computed: 16 fixed
candidate models × 5 Open LLM Leaderboard v1-archive tasks, adjacent-pair
claims, primary rule Γ\*(π = 0.1) < Λ_ref (pooled 0.355; HellaSwag stratum
0.42), BH-FDR primary with BY sensitivity, ρ ∈ {0, 0.2}. Protocol notes,
all logged mechanically at run time: TruthfulQA was dropped entirely (the
archive stores mc1/mc2, not the frozen acc/acc_norm specification);
falcon-7b was dropped on ARC (its snapshot's item template shares no items
with the 15-model modal universe) and GPT-J on Winogrande (malformed score
column).

**Result (Fig. 1 / f17): 49 of 58 claims — 84.5% — are not
contamination-robust** (exact two-sided 95% CI [72.6%, 92.7%]; one-sided
p = 2.7×10⁻²¹ against the pre-registered H4 threshold of 25%; excluding
pilot-overlap pairs: 47/56 = 83.9%, CI [71.7%, 92.4%]). The ρ = 0.2
sensitivity leaves the count unchanged. Per task, the median Γ\*(0.1) is
0.084–0.120 on HellaSwag, Winogrande, and ARC — the typical adjacent claim
tolerates less than one-third of the calibrated contamination strength —
and 0.212 on GSM8K, whose margins are larger. Only **8 claims earn
BH-FDR robustness certificates (6 under BY)**, and every certified claim
has a margin above 5 points: cross-generation capability gaps
(e.g. Llama-3-70B ≻ Qwen1.5-14B on GSM8K) and "everything beats GPT-J."
The leaderboard's *ordering of eras* is contamination-robust; its
*ordering of neighbors* is not.

The protocol was pre-registered on the Open Science Framework
(<https://osf.io/2436z>, DOI `10.17605/OSF.IO/2436Z`, registered 20 July
2026, public and un-embargoed) before any confirmatory number was
publicised. The registration carries the protocol verbatim, the frozen Λ
calibration table (`lambda_priors.csv`, git blob `590d295…`), and a
chain-of-custody addendum recording exactly which files were frozen at tag
`prereg-freeze-v1.0` (commit `ca7773b`) and which were written afterwards —
including the driver script, the item-universe amendment, and a post-freeze
correction to a confidence-interval *report* that left every estimate
unchanged. A mirror is archived independently at the Internet Archive
(`osf-registrations-2436z-v1`), giving a second timestamp not under author
control.

### 9.2 What the audit reduces to, and why that is the point

A reader who works through §5 will notice something the headline conceals,
so we state it ourselves. In the single-draw binary regime the simple bound
gives Γ\*(π) = Δ̂ / (π(1+ρ)). The pre-registered rule "non-robust iff
Γ\*(0.1) < Λ_ref" is therefore *algebraically identical* to

> Δ̂ < π · Λ_ref = 0.0355 (0.042 on HellaSwag),

a threshold on the raw margin. We verified this holds for all 58 claims. The
confirmatory audit could have been executed with a ruler.

This is not a defect; it is the result. Three points follow.

**First, the arithmetic was never the hard part.** Anyone can declare that
gaps below three-and-a-half points are untrustworthy. The question is *which*
threshold is defensible, and that is a measurement problem, not an arithmetic
one. Our threshold is not chosen for convenience: it is π times the median of
eight quality-filtered estimates of contamination-induced lift, converted to
a common headroom parametrisation (§8), frozen before unblinding, and
attached to a third-party timestamp. The contribution is the derivation of
0.0355 from contamination evidence, not the comparison against it. Strip out
the theory and you have a number nobody can justify; strip out the arithmetic
and you have lost nothing.

**Second, the reduction is a property of the regime, not of the method.**
Γ\* collapses to Δ̂/π only when items are binary and measured once, because
the per-item deflation capacity degenerates (Lemma A.1): d(0) = d(1) = 0 for
every Λ < 1. That is a statement about how impoverished single-draw
accuracy data is, and it is worth hearing. Where richer data exists the
machinery genuinely separates: with continuous per-item scores the sharp
knapsack binds strictly tighter than πΛ (§9, TSFM atlas), and at m = 20
repeated draws it certifies Γ\* = 0.172 against the simple bound's 0.154
(§10). The leaderboards audited here publish the weakest data they could;
the reduction is the price of that, and it is the argument for §10's
prescription that benchmarks publish repeated draws.

**Third, the reduction is what makes the result auditable.** Because the rule
is transparent, a sceptical reader can re-derive every classification from
the published margins without trusting our code — and can immediately see
what would change it.

**Sensitivity to the frozen constant (Fig. 2 / f18).** Since the decision
depends on π and Λ only through their product, the entire (π, Λ) plane
collapses to a single curve, which we sweep. The fraction of non-robust
claims is 44.8% at Λ = 0.10, 70.7% at 0.20, 82.8% at 0.30, 84.5% at the
frozen 0.355, and is flat at 84.5% throughout Λ ∈ [0.355, 0.50]. H4 clears
its 25% bar for **every Λ ≥ 0.055** and is significant at α = 0.05 for every
Λ ≥ 0.080. For the headline to fail, true contamination strength would have
to sit below Λ ≈ 0.055 — beneath the floor of the calibrated field range
[0.10, 0.45], below seven of the eight quality-eligible estimates, and an
order of magnitude below the concentrated-exposure ceiling we measured
directly (§7). The qualitative finding survives any calibration a reviewer
could reasonably substitute; only the exact percentage moves.

### 9.3 Is the headline an artifact of auditing adjacent pairs?

Adjacent pairs are *selected* to have the smallest margins in a ranking, so
the objection writes itself: 84.5% may be induced by the claim-selection
rule rather than by contamination sensitivity. We take the objection
seriously enough to measure it. Applying the identical frozen rule to **all
450 model pairs** across the four tasks — the full comparison set, no
selection — gives 30.4% non-robust. Stratifying by rank gap:

| rank gap | non-robust | median margin |
|---|---|---|
| 1 (adjacent) | 49/58 = **84.5%** | 0.011 |
| 2 | 39/54 = 72.2% | 0.023 |
| 3 | 28/50 = 56.0% | 0.033 |
| 4 | 16/46 = 34.8% | 0.046 |
| 5 | 5/42 = 11.9% | 0.054 |
| ≥ 6 | 0/102 = **0.0%** | ≥ 0.068 |

The objection is correct that adjacency selects small margins — and the
gradient it produces is the cleanest quantitative statement of our thesis
that the paper contains. Robustness is not uniformly absent: it is a
monotone function of how far apart two models sit. Comparisons across five
or more rank positions are essentially all robust (2.1% non-robust at gap
≥ 5); comparisons between neighbours are essentially all fragile. That is
precisely "the ordering of eras is contamination-robust, the ordering of
neighbours is not," now with a dose-response curve behind it.

We therefore report both numbers, and are explicit about which claim each
supports. The 30.4% all-pairs figure is the answer to "how often does
contamination sensitivity matter across arbitrary model comparisons." The
84.5% adjacent figure is the answer to the question a leaderboard actually
poses, because a leaderboard is an ordering: it invites the reader to
compare entry k with entry k+1, and those are the comparisons that do not
survive. Neither number is the headline alone; the gradient is.

**What would change our mind.** The finding is falsifiable on three fronts,
pre-specified in the registration. (i) If contamination-induced lift in the
wild were measured at Λ < 0.055 for these benchmarks, H4 would fail — this
is a single well-designed injection study away, and we would report it. (ii)
If the A1 bounded-headroom envelope were rejected at calibrated Λ, the audit
would be re-run under the full-memorization channel and both reported. (iii)
If per-item records for these leaderboards proved systematically unusable,
the corpus falls back to continuous-score data and the substitution is
disclosed. None of these is a hypothetical hedge; each is a study someone
could run against us.

## 10. Design sensitivity: what buys robustness

Two quantified prescriptions for benchmark builders. **Freshness:** at a
3-point margin, raising the post-cutoff item fraction from 0 to 90% lifts
Γ\* from 0.168 to 0.453 — from fragile-everywhere to robust across the
calibrated range. Dating your items is not hygiene; it is identification.
**Repeated draws:** at m = 20 draws the sharp bound certifies Γ\* = 0.172
versus 0.154 for the simple bound (~12% more certified robustness), and
below m = 10 the sharp bound is unreliable (§6). Publishing ≥ 10 draws per
item is what entitles a benchmark to sharp identification.

## 11. Limitations and threats to validity

(1) A1's channel shape is now validated with real memorization at both
small scale and LLM scale (§7), but the LLM measurement is
concentrated-exposure; the corpus-diluted regime that matches pretraining
is bracketed (field priors below, measured ceiling above), not yet
interpolated — the corpus-mixed injection is the outstanding experiment. (2) In the single-draw binary
regime (most current leaderboards), (Λ, π) enter only through their
product; the 2-D frontier earns its second dimension only under repeated
draws or provenance strata — we say so plainly, and §10 is the constructive
answer. (3) Λ calibration from pre/post-cutoff comparisons can be
confounded by temporal drift; the pre-registration pairs items on
difficulty and reports calibration uncertainty. (4) Both case studies are
exploratory; the headline audit is governed by a pre-registration frozen
before any confirmatory Γ\* is computed. (5) ε-spillover should vanish at
LLM corpus scale but is measured, not assumed. (6) This artifact was
audited by its own author (four forensic iterations, including a critical
binary-regime guard); an independent adversarial audit is the correct next
scrutiny.

*Formal statements and proofs of Propositions 1–3, Theorem 4, and the
supporting lemmas (R2 degeneracy, finite-draw bias directions,
Imbens–Manski properties, spillover extension) are in Appendix A
(`appendix_proofs.md`), each pinned to a named unit test.*

## 12. Reproducibility statement

Seed 42 throughout; exact-version lock (`requirements.txt`); 57 unit tests
including constructive sharpness attainment; `python reproduce.py`
regenerates every fast number in ~45 s (`--full` adds the twin-training
grid, ~30 min, CPU); every reported number maps to its generating script in
PROVENANCE.md; the package installs via `pip install -e .` (v0.2.0, MIT).

The confirmatory analysis (§9.1) was pre-registered at
<https://osf.io/2436z> (DOI `10.17605/OSF.IO/2436Z`, 20 July 2026), with
the analysis library and calibration constants frozen beforehand at git tag
`prereg-freeze-v1.0` (commit `ca7773b`). `PREREGISTRATION.md`, `THEORY.md`,
and `results/lambda_priors.csv` in the repository are byte-identical to that
tag and to the copies attached to the registration; deviations and
post-freeze changes are enumerated in the chain-of-custody addendum filed
with it, not silently absorbed.

## References

*Generated by `scripts/make_bibliography.py`; every arXiv entry was
fetched live from the arXiv API. BibTeX: `paper/references.bib`.*

- Tom B. Brown et al. (2020). *Language Models are Few-Shot Learners.* arXiv:2005.14165. `[brown2020gpt3]`
- Victor Chernozhukov et al. (2021). *Long Story Short: Omitted Variable Bias in Causal Machine Learning.* arXiv:2112.13398. `[chernozhukov2021ovb]`
- Inbal Magar and Roy Schwartz (2022). *Data Contamination: From Memorization to Exploitation.* arXiv:2203.08242. `[magar2022contamination]`
- Hugo Touvron et al. (2023). *LLaMA: Open and Efficient Foundation Language Models.* arXiv:2302.13971. `[touvron2023llama]`
- Yucheng Li et al. (2023). *An Open Source Data Contamination Report for Large Language Models.* arXiv:2310.17589. `[li2023contamreport]`
- Aaron Grattafiori et al. (2024). *The Llama 3 Herd of Models.* arXiv:2407.21783. `[grattafiori2024llama3]`
- Cheng Xu et al. (2024). *Benchmark Data Contamination of Large Language Models: A Survey.* arXiv:2406.04244. `[xu2024survey]`
- Felipe Maia Polo et al. (2024). *tinyBenchmarks: evaluating LLMs with fewer examples.* arXiv:2402.14992. `[polo2024tinybenchmarks]`
- Jasper Dekoninck et al. (2024). *ConStat: Performance-Based Contamination Detection in Large Language Models.* arXiv:2405.16281. `[dekoninck2024constat]`
- Mathieu Ravaut et al. (2024). *A Comprehensive Survey of Contamination Detection Methods in Large Language Models.* arXiv:2404.00699. `[ravaut2024survey]`
- Norah Alzahrani et al. (2024). *When Benchmarks are Targets: Revealing the Sensitivity of Large Language Model Leaderboards.* arXiv:2402.01781. `[alzahrani2024benchmarks]`
- Siyu Heng et al. (2024). *Towards Robust Matched Observational Studies with General Treatment Types: Consistency, Efficiency, and Adaptivity.* arXiv:2403.14152. `[heng2024robust]`
- Yujuan Fu et al. (2024). *Does Data Contamination Detection Work (Well) for LLMs? A Survey and Evaluation on Detection Assumptions.* arXiv:2410.18966. `[fu2024doescontam]`
- Andrew M. Bean et al. (2025). *Measuring what Matters: Construct Validity in Large Language Model Benchmarks.* arXiv:2511.04703. `[bean2025construct]`
- Denis Federiakin (2025). *Improving LLM Leaderboards with Psychometrical Methodology.* arXiv:2501.17200. `[federiakin2025psychometric]`
- Hyeong Kyu Choi et al. (2025). *How Contaminated Is Your Benchmark? Quantifying Dataset Leakage in Large Language Models with Kernel Divergence.* arXiv:2502.00678. `[choi2025kds]`
- Marcel Meyer et al. (2025). *Rethinking Evaluation in the Era of Time Series Foundation Models: (Un)known Information Leakage Challenges.* arXiv:2510.13654. `[meyer2025tsfm]`
- Takashi Ishida et al. (2025). *CapBencher: Give Your LLM Benchmark a Built-in Alarm for Test-Set Overfitting.* arXiv:2505.18102. `[ishida2025capbencher]`
- Timo Freiesleben and Sebastian Zezulka (2025). *The Benchmarking Epistemology: Construct Validity for Evaluating Machine Learning Models.* arXiv:2510.23191. `[freiesleben2025epistemology]`
- Bitya Neuhof and Yuval Benjamini (2026). *Rank Intervals for Leaderboards: A Hierarchical Framework for Model Evaluation.* arXiv:2606.08679. `[neuhof2026rankintervals]`
- Han Jiang et al. (2026). *AI Evaluation Should Require Standardized Item-Level Data Releases.* arXiv:2604.03244. `[jiang2026itemlevel]`
- Hongkai Li et al. (2026). *TSFMAudit: Data Contamination Auditing in Forecasting Time Series Foundation Models.* arXiv:2605.26161. `[li2026tsfmaudit]`
- Hosna Oyarhoseini et al. (2026). *A Unified Perturbation Framework for Analyzing Leaderboard Stability and Manipulation.* arXiv:2605.15761. `[oyarhoseini2026perturbation]`
- Jianzhe Chai et al. (2026). *When Benchmarks Leak: Inference-Time Decontamination for LLMs.* arXiv:2601.19334. `[chai2026inferencetime]`
- Md. Niamul Islam Sium (2026). *Quantifying Robustness to Unmeasured Confounding in Time-Varying Treatment Confounder Settings: An Extension of E-value Approach.* arXiv:2602.24261. `[sium2026timevarying]`
- Muhammed Yusuf Kocyigit and Caglar Yildirim (2026). *The Impact of Post-training on Data Contamination.* arXiv:2601.06103. `[kocyigit2026posttraining]`
- Polina Gordienko et al. (2026). *How Hard is it to Rig a Benchmark? A Social Choice Analysis of Leaderboard Robustness.* arXiv:2605.23628. `[gordienko2026socialchoice]`
- Rylan Schaeffer et al. (2026). *Quantifying the Effect of Test Set Contamination on Generative Evaluations.* arXiv:2601.04301. `[schaeffer2026quantifying]`

Classical statistics references (transcribed; verify against the
publisher record before submission):

- Rosenbaum, P. R. (1987). *Sensitivity analysis for certain permutation inferences in matched observational studies.* Biometrika 74(1), 13–26. `[rosenbaum1987sensitivity]`
- Tan, Z. (2006). *A distributional approach for causal inference using propensity scores.* JASA 101(476), 1619–1637. `[tan2006distributional]`
- VanderWeele, T. J. and Ding, P. (2017). *Sensitivity analysis in observational research: introducing the E-value.* Annals of Internal Medicine 167(4), 268–274. `[vanderweele2017evalue]`
- Imbens, G. W. and Manski, C. F. (2004). *Confidence intervals for partially identified parameters.* Econometrica 72(6), 1845–1857. `[imbens2004confidence]`
- Benjamini, Y. and Hochberg, Y. (1995). *Controlling the false discovery rate.* JRSS-B 57(1), 289–300. `[benjamini1995controlling]`
- Benjamini, Y. and Yekutieli, D. (2001). *The control of the false discovery rate in multiple testing under dependency.* Annals of Statistics 29(4), 1165–1188. `[benjamini2001control]`
