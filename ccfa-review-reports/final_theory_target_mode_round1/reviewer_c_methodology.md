# Reviewer C — Methodology and Deployability

## Score

- Methodology: **2.5 / 10**
- Confidence: **0.95**
- Recommendation: **Reject current adaptive-history certificate**

## Reproduced Finding

The exact-seed controlled experiment reproduces 100% one-second endpoint
containment, but the required true current-state containment is only:

- sudden velocity change: **10 / 200 = 5%**;
- sudden direction change: **58 / 200 = 29%**;
- stop-and-go: **48 / 200 = 24%**.

The abrupt generators use 19.5--27 m/s^3 pulses while the fast history model
assumes 2 m/s^3. A different latent trajectory can satisfy the narrow model and
measurement boxes, so LP feasibility does not identify the true model.

## Fatal Concerns

1. The old endpoint `contained` metric did not test the theorem premise
   `(p_t,v_t,a_t) in X_t`.
2. The robust future jerk term alone contributes about 8.66 m at one second;
   the resulting roughly 10 m radius masks the wrong initial box.
3. Robust reset rate is 0% in all abrupt regimes; the proposed monitor does not
   exercise the claimed fallback.
4. No integrated moving-obstacle closed loop executes the selected fallback.
5. One-obstacle LP P99 is already 31--39 ms, before perception, association,
   multi-obstacle estimation, proposal, exact verification, and actuation.

## Major Concerns

- Synthetic generator and estimator share the exact model and bounded noise.
- Fixed L4/L8, oracle-window, MHE/IMM, and compute-matched robust baselines are
  absent.
- Exact safe-candidate evidence has only 49 ordinary positive states and five
  hard positive states.
- Freeze, closed-loop path ratio, and energy are unmeasured for B5.
- Random B3 already matches B5 exact-subset recall.

## Minimum Score-Changing Experiment

A pre-registered full-stack benchmark would require adaptive, fixed
L2/L4/L8/L16, and robust no-history methods on identical moving-obstacle
streams, with pathwise current-state/tube containment, executed fallbacks,
collision, uncertified steps, completion, freeze, energy, and end-to-end P99.
This experiment is not justified until a valid observation-driven mode gate is
derived; the current gate has already failed its necessary premise.

