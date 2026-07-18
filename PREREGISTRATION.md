# Pre-registration — Confirmatory Contamination Re-Audit (Phase 4)

**v1.0-FREEZE (19 July 2026).** Frozen BEFORE any confirmatory Γ\* was
computed; the analysis code is frozen at the git tag `prereg-freeze-v1.0`
in this repository. The Phase-4 pilots (TSFM atlas; the 4-model ARC demo)
are exploratory, labeled as such, and their pairs are flagged in the
confirmatory corpus. To be filed on OSF verbatim (see OSF_FILING.md).

## 1. Primary hypothesis (H4)

**H4:** At the meta-analytically calibrated contamination strength (§4), at
least 25% of adjacent-pair leaderboard claims in the confirmatory corpus are
**not contamination-robust** (their Γ\* falls below the calibrated Λ median for
the winning model's stratum).

Falsification is informative either way: if fewer than 25% are non-robust, the
result is published as a *defence of current leaderboard practice* under the
same protocol.

## 2. Claim corpus (FROZEN)

- Source: the fully open Open LLM Leaderboard v1 archive
  (`open-llm-leaderboard-old/details_<org>__<model>` per-item parquet;
  the v2 details datasets are auth-gated and excluded). Fetched at analysis
  time by `scripts/run_confirmatory_audit.py`; snapshot choice per model =
  newest snapshot with a full item count (tiny debug re-runs skipped).
- Tasks (5, single-file, binary single-draw):
  `arc:challenge|25`, `hellaswag|10`, `truthfulqa:mc|0`, `winogrande|5`,
  `gsm8k|5`. Per-item score = `acc_norm` (fallback `acc`).
- Model list (16 candidates, fixed a priori; spans 7 providers and
  0.5B–70B; models missing a task's item-level file are dropped for that
  task with a logged reason, no substitutions):
  Llama-2-{7b,13b,70b}-hf, Meta-Llama-3-{8B,70B}, Mistral-7B-v0.1,
  Mixtral-8x7B-v0.1, Qwen1.5-{7B,14B,72B}, Yi-34B, gemma-7b, phi-2,
  SOLAR-10.7B-v1.0, gpt-j-6b, falcon-7b.
- Claims: all adjacent pairs in each task's per-task ranking among the
  models available for that task.
- Flags: pairs among {Llama-2-7b, Llama-2-13b, Mistral-7B-v0.1,
  Meta-Llama-3-8B} on ARC-Challenge were analysed in the exploratory demo
  (`p6_oll_demo.csv`) and are marked `pilot_overlap = true`; the primary H4
  statistic is reported both with and without them.

## 3. Estimator specification (FROZEN)

- Regime: single-draw binary items → simple bound at probability level
  (THEORY §5); the library's auto-regime must report "simple" for every
  claim (a "sharp" appearance is a protocol violation).
- Γ\* reported as a curve over π ∈ {0.05, 0.1, 0.2, 0.3}.
- **Primary decision rule (H4):** a claim is *non-robust* iff
  Γ\*(π = 0.1) < Λ_ref of its task stratum (§4). π = 0.1 reflects the
  archive-era contamination fractions documented for these benchmarks
  (e.g. 29.1% flagged MMLU items, 1–45% across benchmarks); 0.1 is the
  conservative low end, and the π-curve shows the sensitivity.
- Two-sided: primary ρ = 0; sensitivity ρ = 0.2 (dose-1 violation evidence,
  THEORY §11).
- Uncertainty: paired item bootstrap (2,000 replicates, seed 42);
  fragility p-values (CI-inversion); Benjamini–Hochberg FDR at q = 0.05
  (primary) with Benjamini–Yekutieli as the dependence-robust sensitivity.
  Per-model clean-score intervals at the parameter level (Imbens–Manski
  2004); set-level in the appendix.

## 4. Λ calibration (FROZEN constants)

- Prior table: `results/lambda_priors.csv` frozen at tag
  `prereg-freeze-v1.0`. **Quality policy:** calibration uses only rows with
  quality ∈ {high, medium}; low/corroboration/qualitative rows inform
  discussion only; CONTAM-CTRL 'measured' rows are concentrated-exposure
  ceilings and enter as upper-bound anchors, never central values.
- **Pooled reference (primary): Λ_ref = 0.355** — the median of the eight
  quality-eligible converted estimates
  {0.02, 0.08, 0.23, 0.31, 0.40, 0.40, 0.42, 0.47}.
- **Task strata (secondary):** HellaSwag: Λ_ref = 0.42 (median of its three
  same-benchmark rows {0.23, 0.42, 0.47}); all other audited tasks lack a
  same-benchmark quality-eligible prior and use the pooled 0.355. (MMLU is
  not in the audited task set; its 0.02 row enters only the pooled median.
  The GSM8K row is quality-low and excluded — GSM8K claims use the pooled
  reference and are additionally flagged as least-calibrated.)
- No post-hoc additions or edits after unblinding.

## 5. Outcomes and reporting

- Primary: fraction of claims non-robust at the stratum-median Λ (H4 test,
  exact binomial CI).
- Secondary: Γ\* distribution; robustness by margin size, benchmark, model
  family; agreement with ConStat flags where ConStat is computable
  (convergent-validity, E8).
- All numbers map to scripts via PROVENANCE.md; analysis code frozen at a
  tagged commit before unblinding.

## 6. What would change our conclusions

- If the A1 channel-shape test (Phase-2/CONTAM-CTRL, small-scale and LoRA)
  rejects the bounded-headroom envelope (observed lift envelope exceeding
  Λ(1−y\*) systematically at calibrated Λ), the audit is re-run under the
  full-memorization channel (Λ → 1 on reduced π) and both are reported.
- If item-level records prove unusable (coverage exclusions > 50% of models),
  the corpus falls back to the TSFM transfer atlas (continuous scores,
  regime R1, sharp knapsack) and this substitution is disclosed prominently.
