import Mathlib

set_option linter.style.header false

/-!
# Tail-sequence lemmas for recurrence-preserving abstractions

This file deliberately proves only deterministic sequence statements. The
probabilistic interpretation of `S` and `T` as return-time survival functions,
and the controlled-policy quantifiers, are separate proof obligations.
-/

namespace RecurrenceTail

/-- Candidate algebraic core: an absolutely summable pairwise tail discrepancy
preserves summability of the two tail sequences. -/
theorem summable_iff_of_summable_abs_sub
    (S T : ℕ → ℝ)
    (hDiff : Summable (fun n ↦ |S n - T n|)) :
    Summable S ↔ Summable T := by
  have hSub : Summable (fun n ↦ S n - T n) := hDiff.of_abs
  constructor
  · intro hS
    exact (hS.sub hSub).congr (fun n ↦ by ring)
  · intro hT
    exact (hT.add hSub).congr (fun n ↦ by ring)

/-- If both tail sequences converge, a finite absolute pairwise discrepancy
forces the same limiting tail mass. For survival functions, this is equality
of non-return probabilities. -/
theorem limit_eq_of_summable_abs_sub
    (S T : ℕ → ℝ)
    (a b : ℝ)
    (hDiff : Summable (fun n ↦ |S n - T n|))
    (hS : Filter.Tendsto S Filter.atTop (nhds a))
    (hT : Filter.Tendsto T Filter.atTop (nhds b)) :
    a = b := by
  have hSub : Summable (fun n ↦ S n - T n) := hDiff.of_abs
  have hLimitSub :
      Filter.Tendsto (fun n ↦ S n - T n) Filter.atTop (nhds (a - b)) :=
    hS.sub hT
  have hZero := hSub.tendsto_atTop_zero
  have hEq : a - b = 0 := tendsto_nhds_unique hLimitSub hZero
  linarith

/-- The discrepancy controls the difference between the two finite tail sums.
For survival sequences, these sums are the corresponding finite mean return
times. -/
theorem abs_tsum_sub_le_tsum_abs_sub
    (S T : ℕ → ℝ)
    (hS : Summable S)
    (hT : Summable T)
    (hDiff : Summable (fun n ↦ |S n - T n|)) :
    |∑' n, S n - ∑' n, T n| ≤ ∑' n, |S n - T n| := by
  rw [← hS.tsum_sub hT]
  have hNorm : Summable (fun n ↦ ‖S n - T n‖) := by
    simpa only [Real.norm_eq_abs] using hDiff
  simpa only [Real.norm_eq_abs] using norm_tsum_le_tsum_norm hNorm

/-- A discounted tail transform cannot amplify an absolutely summable
pairwise discrepancy when the discount lies in `[0, 1]`. -/
theorem norm_tsum_geometric_mul_sub_le
    (S T : ℕ → ℝ)
    (γ : ℝ)
    (hγ0 : 0 ≤ γ)
    (hγ1 : γ ≤ 1)
    (hDiff : Summable (fun n ↦ |S n - T n|)) :
    ‖∑' n, γ ^ n * (S n - T n)‖ ≤ ∑' n, |S n - T n| := by
  apply tsum_of_norm_bounded hDiff.hasSum
  intro n
  rw [norm_mul, Real.norm_eq_abs, Real.norm_eq_abs, abs_pow, abs_of_nonneg hγ0]
  exact mul_le_of_le_one_left (abs_nonneg _) (pow_le_one₀ hγ0 hγ1)

/-- Boundary-rate version of `norm_tsum_geometric_mul_sub_le`: after the
standard `(1 - γ)` scaling, the discrepancy contracts at least linearly as
`γ ↑ 1`. This is an analytic inequality, not yet a return-time generating
function identity. -/
theorem norm_one_sub_mul_tsum_geometric_sub_le
    (S T : ℕ → ℝ)
    (γ : ℝ)
    (hγ0 : 0 ≤ γ)
    (hγ1 : γ ≤ 1)
    (hDiff : Summable (fun n ↦ |S n - T n|)) :
    ‖(1 - γ) * ∑' n, γ ^ n * (S n - T n)‖ ≤
      (1 - γ) * ∑' n, |S n - T n| := by
  have hCore := norm_tsum_geometric_mul_sub_le S T γ hγ0 hγ1 hDiff
  rw [norm_mul, Real.norm_eq_abs, abs_of_nonneg (sub_nonneg.mpr hγ1)]
  exact mul_le_mul_of_nonneg_left hCore (sub_nonneg.mpr hγ1)

/-!
## Cancellation counterexample

Let `τ₁ = 2` almost surely and let `τ₂` equal `1` or `3`, each with
probability `1/2`. Both means are `2`. Their survival sequences differ by
`1/2` at times `1` and `2`, with opposite signs, so their tail discrepancy is
`1`. The following identities machine-check the two algebraic facts used by
the counterexample.
-/

theorem cancellation_example_nonzero_tail_terms :
    |(1 : ℝ) - 1 / 2| + |(0 : ℝ) - 1 / 2| = 1 := by
  norm_num

theorem cancellation_example_generating_factorization (γ : ℝ) :
    γ ^ 2 - ((1 / 2 : ℝ) * γ + (1 / 2 : ℝ) * γ ^ 3) =
      -(1 / 2 : ℝ) * γ * (1 - γ) ^ 2 := by
  ring

end RecurrenceTail
