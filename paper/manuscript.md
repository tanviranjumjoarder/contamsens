# How Contaminated Would It Have To Be? Partial Identification and Sensitivity Analysis for Benchmark Claims Under Unmeasured Data Contamination

**Tanvir Anjum Joarder**¹
¹ Department of Computer Science and Engineering, Rajshahi University of Engineering & Technology, Rajshahi-6204, Bangladesh — `ussash93@gmail.com`

*Draft v0.1 — 17 July 2026. Every number in this draft is regenerable via
`python reproduce.py` and maps to a script through PROVENANCE.md. Sections
marked [PENDING] await the pre-registered confirmatory audit and the
GPU-scale CONTAM-CTRL run.*

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
levers. Sensitivity analysis converts an unanswerable question — *is the
benchmark contaminated?* — into an answerable one, with a one-number
reporting standard any leaderboard can adopt.

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
post-training can *reverse* lift [2601.06103]. ConStat [dekoninck et al.,
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
(twelve sourced rows in `results/lambda_priors.csv`). Λ is model-stratified
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
designed. [PENDING: the pre-registered confirmatory audit over the frozen
corpus; these pairs are flagged exploratory.]

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

## 12. Reproducibility statement

Seed 42 throughout; exact-version lock (`requirements.txt`); 50 unit tests
including constructive sharpness attainment; `python reproduce.py`
regenerates every fast number in ~45 s (`--full` adds the twin-training
grid, ~30 min, CPU); every reported number maps to its generating script in
PROVENANCE.md; the package installs via `pip install -e .` (v0.2.0, MIT).

## References

*(Verified links; to be converted to formal citations at submission.)*
Rosenbaum (1987), Observational Studies sensitivity bounds · Tan (2006)
JASA marginal sensitivity model · VanderWeele & Ding (2017) Annals of
Internal Medicine, E-values · Chernozhukov et al., omitted variable bias in
causal ML — arXiv:2112.13398 · Magar & Schwartz (ACL 2022) —
arXiv:2203.08242 · Touvron et al. (2023) — arXiv:2302.13971 · Llama 3
(2024) — arXiv:2407.21783 · Brown et al. (2020) — arXiv:2005.14165 ·
ConStat (NeurIPS 2024) — arXiv:2405.16281 · CapBencher — arXiv:2505.18102 ·
Contamination-detection surveys — arXiv:2404.00699, 2410.18966, 2406.04244 ·
TSFM leakage — arXiv:2510.13654, 2605.26161 · Dose-response —
arXiv:2601.04301 · Post-training contamination — arXiv:2601.06103 ·
Inference-time decontamination — arXiv:2601.19334 · Open-source
contamination report — arXiv:2310.17589 · Rank intervals — arXiv:2606.08679
· Leaderboard perturbation/social choice — arXiv:2605.15761, 2605.23628,
2402.01781 · tinyBenchmarks — arXiv:2402.14992 · IRT leaderboards —
arXiv:2501.17200 · Construct validity — arXiv:2511.04703, 2510.23191,
2604.03244 · Generalized Rosenbaum — arXiv:2403.14152 · Time-varying
E-values — arXiv:2602.24261 · KDS — arXiv:2502.00678.
