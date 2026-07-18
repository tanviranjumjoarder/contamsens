# Red-Team Report — contamsens v0.1 → v0.2
### Adversarial improvement cycle, 16 July 2026

**Protocol:** Stage 1–13 elite-review loop (novelty-assessment skill: 6 adversarial
search rounds; simulate-reviewers skill: 3 independent personas, NeurIPS-family
calibration, tagged `inferred-from-family` since this is an idea-stage artifact with
no live CFP to verify against). **This is a simulation to improve the work, not a
prediction of any real outcome.**

---

## 1. Executive summary

The idea survives the attack, but v0.1 had **three genuine defects**, all found by
literature rounds targeting its weakest joints, all now fixed in v0.2 (THEORY.md §9,
code + 8 new tests, 27/27 passing):

1. **The monotonicity axiom was empirically false.** Published evidence shows
   contamination can *depress* scores (post-training interference, stale
   memorization: [2601.06103](https://arxiv.org/pdf/2601.06103),
   [2402.15938](https://arxiv.org/pdf/2402.15938)). One-sided bounds therefore
   missed a real overturn channel: *B deflated* can flip "A ≻ B" just as *A
   inflated* can. → **Fixed: two-sided CSM±(Λ⁺, Λ⁻, π)**, with worst case
   B_A(Λ⁺,π) + U_B(Λ⁻,π) and a two-sided frontier πΛ(1+ρ) = Δ.
2. **The "detection is impossible on closed models" wedge was overstated.**
   [ConStat (NeurIPS 2024)](https://arxiv.org/pdf/2405.16281) detects *and
   quantifies* contamination from performance alone — no corpus access. The wedge
   survives but must be restated precisely: ConStat needs **reference benchmarks
   and reference models assumed clean**; CSM needs neither. → **Fixed: repositioned
   as "reference-free, claim-level identification"; ConStat becomes both the strong
   baseline and a Λ-calibration input.**
3. **Item-level budgets misdescribe how leakage happens.** Datasets leak whole,
   not item-by-item. → **Fixed: group-level (all-or-nothing per source) budgets**,
   bounded by the fractional-knapsack relaxation — provably never wider than the
   item-level bound, so source metadata tightens identification for free.

Two **gifts** were also found: (a) a Jan-2026 dose-response study
([2601.04301](https://arxiv.org/html/2601.04301v2)) provides exactly the empirical
Λ-calibration curves Phase 3 assumed would need to be created — and shows Λ must be
**model-stratified** (grows with capacity, shrinks with unique-corpus size);
(b) item-level Open LLM Leaderboard data exists at scale
(~[87.9M response observations](https://arxiv.org/pdf/2501.17200)), de-risking the
real-data phase entirely.

**Novelty verdict: NOVEL, confidence medium-high** (details §4). The
partial-identification / sensitivity-analysis framing for benchmark *claims* has no
occupant after 6 targeted rounds; the nearest neighbors (ConStat, CapBencher, Rank
Intervals) each miss at least two of {claim-level, worst-case identification,
reference-free, design theory}.

---

## 2. Stage 1 — Idea understanding and the problem's Q1-worthiness

Problem: leaderboard claims are treated as measurements while their dominant error
source (contamination) is unmodeled. Hidden assumptions found in v0.1 and now
surfaced: (i) monotone lift (falsified → U1); (ii) item-independent leakage
(unrealistic → U2); (iii) a single global Λ (falsified by dose-response → U3);
(iv) per-item continuous scores (binary single-draw is the dominant real case —
acknowledged in v0.1 regime R2, now central to the design-sensitivity pitch).
The problem itself is Q1-important: it sits on every empirical claim in ML, has
regulatory tailwind, and the field's own position papers
([2511.04703](https://arxiv.org/abs/2511.04703),
[2510.23191](https://arxiv.org/abs/2510.23191)) demand an estimator no one has
supplied.

## 3. Stage 2–3 — Literature map and what each cluster fails to do

| Cluster | Representative | What they solved | What remains unsolved (our lane) |
|---|---|---|---|
| Detection (corpus/membership) | surveys [2404.00699](https://arxiv.org/html/2404.00699v4), [2406.04244](https://arxiv.org/pdf/2406.04244); assumptions critique [2410.18966](https://arxiv.org/pdf/2410.18966) | flagging overlap when corpus/model access exists | no statement about whether any *conclusion* changes; assumptions shown fragile by their own survey |
| Performance-based detection | [ConStat](https://arxiv.org/pdf/2405.16281), [CapBencher](https://arxiv.org/pdf/2505.18102), KDS [2502.00678](https://arxiv.org/abs/2502.00678), CoDeC [2510.27055](https://arxiv.org/html/2510.27055v1) | model-level tests/estimates of inflation | require reference models/benchmarks or fine-tuning access; model-level not claim-level; point estimates not identified sets |
| Effect quantification | Magar & Schwartz 2022; dose-response [2601.04301](https://arxiv.org/html/2601.04301v2); post-training [2601.06103](https://arxiv.org/pdf/2601.06103) | measured lift under controlled injection | never turned into an inferential correction for published claims — **this is our Λ-calibration supply chain** |
| Mitigation | inference-time decontamination [2601.19334](https://arxiv.org/abs/2601.19334) | reduce lift at eval time (open models) | needs model internals; unverifiable itself; silent on legacy claims |
| Leaderboard statistics | [Rank Intervals 2606.08679](https://arxiv.org/abs/2606.08679), [2605.15761](https://arxiv.org/pdf/2605.15761), [tinyBenchmarks](https://arxiv.org/pdf/2402.14992), IRT [2501.17200](https://arxiv.org/pdf/2501.17200) | sampling uncertainty; item efficiency | systematic bias unmodeled at any n — our Λ=0 special case |
| Causal sensitivity | Rosenbaum; [E-values](https://www.acpjournals.org/doi/10.7326/M17-1485); [2112.13398](https://arxiv.org/pdf/2112.13398) | bounding unmeasured bias in observational studies | never applied to ML evaluation — our machinery source |

## 4. Stage 3/8 — Novelty audit (novelty-assessment format)

```json
{
  "decision": "novel",
  "confidence": "medium-high",
  "justification": "6 adversarial rounds (3 in the discovery session, 3 targeted at the weakest joints in this cycle) found no partial-identification or sensitivity-analysis treatment of benchmark claims. The three nearest neighbors each fail >=2 of the 4 defining properties (claim-level / worst-case identified sets / reference-free / design sensitivity).",
  "most_similar_papers": [
    {"title": "ConStat: Performance-Based Contamination Detection in LLMs", "year": 2024,
     "overlap": "~30% - statistical, performance-only, quantifies inflation; but model-level detection requiring reference benchmarks AND reference models assumed clean; no identified sets, no claim-level robustness, no design theory"},
    {"title": "Rank Intervals for Leaderboards", "year": 2026,
     "overlap": "~20% - claim-level intervals, but sampling-only; our Lambda=0 special case"},
    {"title": "CapBencher: built-in alarm for test-set overfitting", "year": 2025,
     "overlap": "~15% - statistical test about contamination effects on scores, but a benchmark-design-time alarm, not retrospective identification of existing claims"},
    {"title": "Quantifying the Effect of Test Set Contamination on Generative Evaluations", "year": 2026,
     "overlap": "~10% - measures the dose-response we consume as calibration; no inferential framework"}
  ],
  "differentiation": "contamsens is the only framework that (a) states the clean-score estimand and its identification failure, (b) returns worst-case identified sets and a per-claim robustness value from scores alone, (c) assumes no clean reference assets, and (d) yields a constructive design theory (repeated draws + fresh-item strata buy identification).",
  "novelty_illusion_risk": "MODERATE. The core bound is mathematically elementary; if the paper leads with the bound rather than the estimand/identification framing + calibration + re-audit, reviewers will read it as a relabeled Manski exercise. Depth must come from the MSM bridge theorem, the grouped/stratified extensions, noisy-knapsack selection correction, and design sensitivity asymptotics."
}
```

ConStat overlap (~30%) is the binding constraint and sits at the 30% refinement
threshold of the discovery protocol — **the refinement is v0.2's repositioning**:
claim-level identification consuming ConStat as calibration, with ConStat as
baseline. Post-refinement overlap is comfortably below threshold because the
estimand, output object, and assumption set all differ.

## 5. Stage 10 — Reviewer simulation (3 independent personas, NeurIPS-family, harshness 4–5)

### Reviewer 1 — causal-inference statistician (harshness 5)
**Strengths.** (1) The estimand-first formalization of contamination as an
identification failure is correct and overdue; (2) Proposition 1's attainment
construction is verified in code (`test_sharpness_attained`); (3) the reduction to
sampling-only analysis at Λ=0 cleanly nests prior work.
**Weaknesses.** (W1.1) Prop 1 is a sorting argument; for JMLR/NeurIPS theory
credibility the paper needs the CSM↔marginal-sensitivity-model bridge theorem,
asymptotics for Γ̂\*, and a formal treatment of the noisy-knapsack selection effect
— v0.1 defers all three ("open refinement"). (W1.2) One-sided monotonicity is
contradicted by the paper's own citation base. *(Fixed: U1.)* (W1.3) In regime R2,
(Λ, π) enter only through their product — the 2-D frontier is cosmetic for binary
benchmarks; say so honestly and lean on regimes where they separate. **Soundness
3/5 (v0.1) → 4/5 (v0.2 plan), confidence 5.**

### Reviewer 2 — LLM-evaluation empiricist (harshness 4)
**Strengths.** (1) Reference-free operation is a real practical wedge; (2) the
blind leaderboard demo (contaminated model's winning claim flagged at Γ\*=0.173 vs
≥0.29 clean) is exactly the right validation shape; (3) reporting Γ\* as a curve
pre-empts the arbitrary-knob objection.
**Weaknesses.** (W2.1) All validation is synthetic and *circular* — injection
follows A1 by construction; nothing yet shows real memorization respects the
headroom channel. CONTAM-CTRL (LoRA injection) is necessary, not optional, and one
real-leaderboard case study must appear in Paper 1. (W2.2) ConStat neither cited
nor compared in v0.1. *(Fixed: §9 positioning + baseline.)* (W2.3) For the dominant
binary single-draw benchmarks the method reduces to "subtract πΛ" — a constant
shift; the interesting machinery needs multi-draw data. *(Response: correct — and
that is the design-sensitivity result: benchmarks should publish multi-draw
item-level results because identification sharpens; plus U3's stratified budgets
make the bound non-constant even in R2.)* **Soundness 2/5 (v0.1) → 4/5 (v0.2 plan),
confidence 4.**

### Reviewer 3 — benchmark practitioner / D&B track (harshness 3)
**Strengths.** (1) Γ\* is a two-line addition to any results table — adoption cost
near zero; (2) the frontier figure is immediately legible; (3) compute cost is
negligible, so anyone can run it.
**Weaknesses.** (W3.1) Who sets π? Practitioners need defaults with provenance.
*(Response: cutoff-stratified budgets set π=0 where provable; elsewhere report the
curve and the meta-analytic prior.)* (W3.2) Leakage granularity is datasets, not
items. *(Fixed: U2.)* (W3.3) "Why not just evaluate on fresh items?" *(Response:
fresh items are scarce, legacy claims still steer decisions, and freshness decays —
§9's temporal stratification makes freshness an input, not an alternative.)*
**Overall 4/5, confidence 3.**

### Meta-review
Convergent finding (R1.2 ≡ R2's realism concern ≡ R3.2): **v0.1's contamination
channel was too idealized** — both in sign (monotone) and in granularity
(item-level). Both are now fixed in the model rather than the rhetoric, which is
the right kind of fix. Biggest remaining risk to acceptance: **circular validation**
(W2.1) — the only weakness that requires new experiments rather than new writing.
A champion exists (R3) only if the real-data case study lands. **Decision-risk
(v0.1): borderline-reject. Projected (v0.2 + CONTAM-CTRL + one real case study):
borderline-accept to likely-accept at NeurIPS D&B; JMLR viable once the bridge
theorem and Γ̂\* asymptotics are written.**

### Prioritized fix list
| # | Fix | Tag | Status |
|---|---|---|---|
| 1 | Two-sided CSM (sign realism) | fix-now | **done** (code + tests) |
| 2 | ConStat positioning + baseline | fix-now | **done** (THEORY §9) |
| 3 | Group-level budgets (granularity realism) | fix-now | **done** (code + tests) |
| 4 | Model-stratified Λ + meta-analytic calibration sources | fix-now | **done** (spec'd in §9; harvest in P3) |
| 5 | CONTAM-CTRL LoRA injection (break circularity) | structural for Paper 1 | Phase 2, unchanged |
| 6 | One real-leaderboard case study (87.9M-observation item-level data) | structural for Paper 1 | Phase 4, de-risked |
| 7 | MSM bridge theorem + Γ̂\* asymptotics + noisy-knapsack correction | structural (theory) | Phase 1, now mandatory not optional |
| 8 | Honest R2 statement: product-only identification in binary regime | fix-now | write into THEORY §5 at next edit |

## 6. Stages 5–6 — Re-engineered contributions (v0.2)

C1. Estimand + identification-failure theorem for benchmark scores under latent
contamination. C2. **Two-sided** CSM±(Λ⁺,Λ⁻,π) with sharp identified sets; grouped
and cutoff-stratified budgets; proof that group structure and post-cutoff strata
only tighten. C3. Bridge theorem to the marginal sensitivity model (inherits
calibration semantics; answers "trivial math"). C4. Γ\* with two-sided frontier
π Λ(1+ρ)=Δ; estimation theory (consistency, bootstrap validity, selection-corrected
knapsack). C5. Meta-analytic Λ calibration from published dose-response + ConStat
estimates + matched cutoff pairs — **model-stratified**, uncertainty propagated.
C6. Ground-truth validation: synthetic + CONTAM-CTRL (real LoRA memorization
tests A1's channel shape externally). C7. Re-audit of real leaderboard claims
(item-level OLL data) with FDR control; ConStat as baseline. C8. Design
sensitivity: repeated draws and fresh-item fractions as identification-buying
design levers — a prescription, not just an audit. C9. `contamsens` + reporting
standard.

Dropped/demoted: "detection is impossible on closed models" as a headline
(overstated vs ConStat) → now "the only reference-free, claim-level instrument."

## 7. Stages 7–9 — Data, benchmarks, experiments (deltas only)

**Confirmed data:** [open-llm-leaderboard/results](https://huggingface.co/datasets/open-llm-leaderboard/results)
+ [HuggingFaceH4 evaluations](https://huggingface.co/datasets/HuggingFaceH4/open-llm-leaderboard-evaluations-results)
(item-level, ~87.9M observations); your TSFM atlas (1,632 cells, zero new compute);
PTB-XL/CODE-15/VitalDB via medtsfm loaders. **New baselines:** ConStat (strong),
CapBencher, KDS/CoDeC (fine-tune-access class), inference-time decontamination
(model-access class), Rank Intervals (sampling-only class) — one representative per
*assumption class*, which is the honest comparison structure. **New experiments:**
(E5) A1-shape test on CONTAM-CTRL — fit the empirical lift-vs-headroom curve, test
the multiplicative channel against additive and full-memorization alternatives;
(E6) ρ-sensitivity (two-sided stress); (E7) grouped-vs-item bound gap on real
source metadata; (E8) agreement study: does low Γ\* co-occur with ConStat flags on
the same models? (convergent validity — psychometrics reviewers love this and it
costs one figure).

## 8. Stage 12 — Scores (v0.1 → v0.2)

| Dimension | v0.1 | v0.2 |
|---|---|---|
| Novelty | 8.5 | 8.5 |
| Research-gap strength | 9 | 9 |
| Originality (conceptual) | 9 | 9 |
| Technical depth | 5.5 | **7.5** (bridge theorem + estimation theory now mandatory) |
| Experimental depth (planned) | 6 | **8** |
| Theoretical contribution | 6 | **7.5** |
| Practical significance | 8.5 | 9 |
| Industrial relevance | 8 | 8.5 |
| Societal impact | 7 | 7 |
| Dataset quality | 6 | **8.5** (item-level OLL confirmed) |
| Benchmark quality | 5 | **8** (assumption-class baseline design) |
| Publication readiness | 4 | 5.5 (theory phase still ahead) |
| Reviewer excitement | 7 | 8 |
| Funding attractiveness | 8 | 8.5 |
| Patent potential | 3 | 3 (methodology; keep open) |
| Q1 journal probability | 0.55 | **0.70** |
| Top-conference probability (D&B/main) | 0.45 | **0.60** |

## 9. Stage 13 — Verdict, risks, venues, extensions

**Verdict: proceed.** The idea is stronger after attack than before — the two
defects found were fixable in the *model* rather than the pitch, which is the
signature of a sound core. Convergence check per Stage 11: a further iteration
found no new structural weakness beyond the already-scheduled theory debt (fix
#7) and validation debt (fixes #5–6); recursion stops here.

**Threats to validity (ranked).** (1) A1 channel-shape mismatch with real
memorization — *the* falsifiable risk; E5 tests it, and if the channel is closer
to full-memorization-or-nothing, CSM still applies with Λ→1 on a smaller π (the
model family is closed under that reparametrization). (2) Λ-calibration
confounding by temporal drift — mitigate with difficulty-matched pairs + drift
negative controls (Idea 29 machinery). (3) Concurrent work — ingredients public;
ship theory+validation early on arXiv. (4) Product-only identification in R2 —
state honestly; design-sensitivity prescription is the constructive answer.

**Venues (updated).** Paper 1 (theory + validation + case study): NeurIPS D&B or
JMLR. Paper 2 (large re-audit): NeurIPS D&B / TMLR fast-track. The ConStat
comparison makes COLM plausible as a third home. Journal floor unchanged (TIFS/
TKDE). **Extensions** unchanged from the discovery document (§4.1K) plus one new:
Γ\*-style sensitivity for *judge* bias (same machinery, LLM-as-judge channel) —
a natural Paper 3 bridging to the systematic-uncertainty budget.

**Commercialization/patents:** low patent potential (open methodology is the
point); commercial value is in audit/assurance services and leaderboard adoption —
value flows from becoming the standard, which requires the permissive license
already chosen.

---

*Full literature links for this cycle are embedded above; the discovery-phase
citation base lives in `ELITE_Research_Discovery_2026.md` (Appendix). Simulated
reviews are archetypes, not predictions; real panels vary widely.*
