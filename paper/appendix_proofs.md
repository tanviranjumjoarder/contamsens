# Appendix A — Formal statements and proofs

*Companion to `manuscript.md`; every result is implemented in
`src/contamsens` and pinned by a unit test named in brackets.*

## A.1 Setup and notation

Items i = 1,…,n. For a fixed model, observed scores y = (y₁,…,y_n) ∈ [0,1]ⁿ,
clean (counterfactual) scores y\* ∈ [0,1]ⁿ, latent contamination
c ∈ {0,1}ⁿ. Estimand θ = n⁻¹Σᵢ y\*ᵢ; observed mean μ̂ = n⁻¹Σᵢ yᵢ.

**CSM(Λ, π)** (Λ, π ∈ [0,1]):
(A1) 0 ≤ yᵢ − y\*ᵢ ≤ cᵢ Λ (1 − y\*ᵢ) for all i;  (A2) n⁻¹Σᵢ cᵢ ≤ π.

Define, for Λ < 1, the **floor** ℓᵢ(Λ) = max{0, (yᵢ − Λ)/(1 − Λ)} and the
**deflation capacity** dᵢ(Λ) = yᵢ − ℓᵢ(Λ); for Λ = 1, ℓᵢ = 0, dᵢ = yᵢ.
Write k = ⌈πn⌉ ∧ n and let d₍₁₎ ≥ … ≥ d₍ₙ₎ be the sorted capacities.

**Lemma A.0 (capacity form).** For Λ ∈ (0,1),
dᵢ(Λ) = yᵢ if yᵢ ≤ Λ, and dᵢ(Λ) = Λ(1−yᵢ)/(1−Λ) if yᵢ > Λ. Moreover
dᵢ(Λ) = min{ yᵢ, Λ(1−yᵢ)/(1−Λ) }, hence dᵢ is the minimum of two affine
functions of yᵢ and is **concave in yᵢ**; it is continuous, maximized at
yᵢ = Λ with value Λ, and nondecreasing in Λ. *Proof.* Solving (A1) for
y\*ᵢ with cᵢ = 1: yᵢ ≤ Λ + (1−Λ)y\*ᵢ ⟺ y\*ᵢ ≥ (yᵢ−Λ)/(1−Λ); intersect
with y\*ᵢ ≥ 0. The min form follows by checking which branch binds at
yᵢ ⋚ Λ; the derivative of the right branch in Λ is (1−yᵢ)/(1−Λ)² ≥ 0. ∎
[`test_deflation_hand_computed`, `test_deflation_peak_at_lambda`]

## A.2 Proposition 1 (sharp identified set)

**Statement.** Under CSM(Λ, π), the identified set for θ given y is exactly
the interval  Θ(y) = [ μ̂ − B(Λ, π),  μ̂ ],  where
B(Λ, π) = n⁻¹ Σ_{j=1}^{k} d₍ⱼ₎(Λ).

**Proof.** *(⊆)* For any feasible (c, y\*), θ = μ̂ − n⁻¹Σ_{i: cᵢ=1}(yᵢ−y\*ᵢ).
Each term obeys 0 ≤ yᵢ − y\*ᵢ ≤ dᵢ(Λ) (Lemma A.0), and at most k indices
have cᵢ = 1; the sum of any ≤ k capacities is at most the sum of the k
largest. Hence θ ∈ [μ̂ − B, μ̂].
*(⊇, sharpness)* The upper endpoint is attained by c ≡ 0, y\* = y. The
lower endpoint is attained by setting cᵢ = 1 exactly on an argmax set S of
k capacities and y\*ᵢ = ℓᵢ(Λ) for i ∈ S, y\*ᵢ = yᵢ otherwise; feasibility
of (A1) holds by construction of ℓᵢ (the constructed lift equals the cap).
Intermediate values are attained by scaling y\*ᵢ continuously between ℓᵢ
and yᵢ on S (θ is continuous in that scaling), so Θ(y) is the full
interval. ∎ [`test_sharpness_attained`, `test_interval_contains_mu_and_is_ordered`]

**Corollary 1 (distribution-free width).** B(Λ, π) ≤ πΛ + Λ/n, with
n⁻¹ k max_i dᵢ ≤ (π + 1/n)Λ by Lemma A.0's maximum; in particular the
identified-set width is at most πΛ up to the ⌈·⌉ rounding term, and the
bound is achieved when k items sit at yᵢ = Λ.
[`test_simple_bound_dominates_sharp`]

**Corollary 2 (reductions).** B(0, π) = B(Λ, 0) = 0, so Θ collapses to
{μ̂}: sampling-only analyses (e.g. rank intervals) are the Λ = 0 special
case. [`test_reduction_at_zero`]

**Lemma A.1 (R2 degeneracy).** If yᵢ ∈ {0,1} for all i and Λ < 1, then
dᵢ(Λ) = 0 for all i, hence B(Λ, π) = 0, while B(1, π) = n⁻¹Σ_{j≤k} y₍ⱼ₎:
B(·, π) is identically zero on [0,1) with a jump at Λ = 1. Consequently the
per-item machinery is vacuous on single-draw binary data and the population
simple bound (width πΛ at the probability level) must be used.
*Proof.* d(0) = 0 (left branch), d(1) = 1·(1−1)/(1−Λ) = 0 (right branch);
at Λ = 1, d = y. ∎ [`test_sharp_bound_degenerates_on_binary`,
`test_ceiling_effect`]

## A.3 Proposition 2 (claim robustness) and Γ\*

**Statement (two-sided form).** For models A, B on shared items with margin
Δ̂ = μ̂_A − μ̂_B > 0, under CSM±(Λ, ρΛ, π) applied to both models the claim
"A ≻ B holds for the clean scores" is TRUE over the entire identified set
iff Δ̂ > B_A(Λ, π) + U_B(ρΛ, π), where U is the inflation analogue of B
built from uᵢ(Λ⁻) = min{1 − yᵢ, yᵢΛ⁻/(1−Λ⁻)}.

*Proof.* θ_A ≥ μ̂_A − B_A and θ_B ≤ μ̂_B + U_B, with both endpoints
attained (Prop. 1 and its mirror image applied to −y); the extremal joint
configuration is feasible because A's and B's contamination assignments are
unconstrained across models. Hence min(θ_A − θ_B) = Δ̂ − B_A − U_B. ∎
[`test_gamma_star_bisection_consistency`, `test_is_robust_rho_consistent_with_gamma_star`]

**Definition (Γ\*).** Γ\*(π) = inf{Λ ∈ [0,1] : B_A(Λ,π) + U_B(ρΛ,π) ≥ Δ̂},
with Γ\* = ∞ if no such Λ exists. Well-defined by continuity and
monotonicity of both maps in Λ (Lemma A.0 and its mirror). In the simple
regime the closed form is Γ\* = Δ̂ / (π(1+ρ)).
[`test_gamma_star_simple_closed_form`, `test_gamma_star_twosided_is_smaller`]

## A.4 Proposition 3 (selection-bounded bridge)

**Statement.** Let contamination be random with propensities qᵢ = P(cᵢ=1)
subject to (i) n⁻¹Σqᵢ = π and (ii) odds(qᵢ)/odds(qⱼ) ≤ Γ²_sel for all i,j.
Then the worst-case expected bias
B(Λ, π, Γ_sel) = max_q n⁻¹Σᵢ qᵢ dᵢ(Λ) satisfies:
(a) it is attained by a two-point propensity vector taking values
{q_lo, q_hi} with odds(q_hi) = Γ²_sel·odds(q_lo), the high value assigned
to the t largest capacities for some t;
(b) B(Λ, π, 1) = π d̄ (random contamination) and
B(Λ, π, ∞) = the Proposition-1 top-k bound;
(c) B is nondecreasing and continuous in Γ_sel.

*Proof.* (a) The objective is linear in q over the polytope defined by
(i)–(ii); extreme points of that polytope have coordinates taking at most
two distinct values with the odds constraint active between them (any three
distinct values admit a feasible transfer increasing the objective:
move mass toward larger dᵢ). Given the two values, assigning q_hi to the
largest capacities is optimal by rearrangement. (b) At Γ_sel = 1 the only
feasible q is constant = π. As Γ_sel → ∞, q_hi → 1 on t = k items and
q_lo → 0 elsewhere is feasible in the limit, recovering the deterministic
knapsack. (c) Enlarging Γ_sel enlarges the feasible set; continuity follows
from continuity of the extreme-point solution in Γ_sel. ∎
[`test_random_endpoint`, `test_adversarial_endpoint`,
`test_monotone_in_gamma_sel_and_bracketed`]

## A.5 Theorem 4 (consistency and rate of Γ̂\*)

**Statement.** Let items be iid with per-item scores having distribution F
(continuous case, regime R1), and let d(·;Λ) be as in Lemma A.0. Define
B(Λ) = E[d 1{d ≥ q_{1−π}(d)}]/1 + boundary term (the trimmed upper mean at
level π) and assume F_d is continuous with positive density at its
(1−π)-quantile for the relevant Λ, that Δ = μ_A − μ_B > 0, and that
B(Γ\*) = Δ with B strictly increasing at Γ\* ∈ (0,1). Then
(i) Γ̂\* → Γ\* a.s.; (ii) √n(Γ̂\* − Γ\*) ⇒ N(0, σ²) with
σ² = Var-functional of (Δ̂, B̂) via the delta method with slope 1/B′(Γ\*);
(iii) the paired nonparametric item bootstrap consistently estimates the
law in (ii).

*Proof sketch.* B̂(Λ) is a trimmed L-statistic with bounded, monotone
weight generator; under the quantile-density condition it is strongly
consistent and Hadamard differentiable at F tangentially to the uniform
ball (van der Vaart & Wellner, Lemma 22.10-type arguments), giving a
√n-CLT jointly with Δ̂ (a sample mean on the same items — the pairing
supplies the joint law). Pointwise a.s. convergence of the nondecreasing
functions B̂(·) to the continuous limit B(·) upgrades to uniform
convergence on [0,1] by Pólya's argument. The map (B, Δ) ↦ B⁻¹(Δ) is
differentiable at (B, Δ) with B′(Γ\*) > 0; the functional delta method
yields (ii), and Hadamard differentiability + Efron bootstrap consistency
for L-statistics yields (iii). ∎
*Empirical verification:* RMSE slope −0.518 vs theoretical −0.5 over
n ∈ [100, 25600], bias ≤ 0.0025 (`p1_t2_consistency.csv`, fig. f10).

## A.6 Lemma A.2 (finite-draw bias directions, E9)

With m Bernoulli draws per item, p̂ᵢ ~ m⁻¹Bin(m, pᵢ): (i) by Lemma A.0's
concavity and Jensen, E[d(p̂ᵢ)] ≤ d(pᵢ) pointwise (smoothing bias down);
(ii) the top-k selection over noisy capacities satisfies
E[max-k mean of d(p̂)] ≥ max-k mean of E[d(p̂)] (selection bias up). The
net sign is regime-dependent; measured: −40% (m = 2) to +4% (m = 10) to
<1% (m = 50), motivating the m ≥ 10 prescription and the simple-bound
fallback, which depends on p̂ only through the unbiased grand mean.
[`p1_t3_finite_draws.csv`; fig. f11]

## A.7 Lemma A.3 (Imbens–Manski critical value)

The equation Φ(c + w) − Φ(−c) = 1 − α has a unique root c(w) ∈
[z_{1−α}, z_{1−α/2}] for every w ≥ 0; c is continuous and nonincreasing in
w with c(0) = z_{1−α/2} and c(∞) = z_{1−α}. *Proof.* The left side is
strictly increasing in c and increasing in w; evaluate at the bracket
endpoints. ∎ Hence the parameter-level interval
[θ̂_lo − c·σ̂_lo, θ̂_hi + c·σ̂_hi] has asymptotic coverage ≥ 1 − α for
every θ in the identified set, with equality at the endpoints (Imbens &
Manski, 2004, Thm 1 conditions: endpoint estimators jointly asymptotically
normal — supplied here by Theorem 4's machinery — and identified-set width
consistently estimated). [`test_im_critical_value_limits`,
`test_im_worst_case_parameter_coverage`]

## A.8 Spillover-robust interval (U4)

If additionally |n⁻¹Σ_{i:cᵢ=0}(yᵢ − y\*ᵢ)| ≤ ε (bounded net drift on clean
items), every bound above extends by widening both endpoints by ε; the
proof of Prop. 1 goes through with the decomposition
θ = μ̂ − (contaminated deflation) − (clean-item drift). ε is estimable from
twin experiments (measured 0.003–0.30, growing with dose).
[`test_twosided_upper_moves_only_with_lam_minus`; `p2b_spillover_fix.csv`]
