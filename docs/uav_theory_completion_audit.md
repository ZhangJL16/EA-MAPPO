# Requirement-by-Requirement Completion Audit

## Audit rule

This audit maps every numbered requirement in the theory-contribution-search
brief to current, inspectable evidence.  `VALIDATED` means the requested object
or check exists and was directly verified.  `REJECTED` means the branch was
attempted and falsified by proof, counterexample, prior art, or experiment.
`GATED_NOT_RUN` means a later experiment was explicitly conditional on a viable
theory candidate and was not run after that candidate failed.  The latter is
not promoted to positive evidence.

## Requirements 1--12: feasibility and backup

| # | Requirement | Status | Authoritative evidence |
| ---: | --- | --- | --- |
| 1 | Core sampled, bounded-input, multi-obstacle UAV question | VALIDATED | Problem and unresolved target in `docs/uav_theory_contribution_search.md` |
| 2 | Treat observed infeasibility as the primary defect | VALIDATED | Earlier 34 fallbacks and harder matched counts in `docs/uav_theory_experiment_validation.md` |
| 3 | Define joint feasibility margin `rho` | VALIDATED | Definition and implementation in `docs/uav_recursive_feasibility_derivation.md` and `review_bundle/safety/collision/feasibility.py` |
| 4 | Continuity, Lipschitzness, differentiability, Clarke behavior, dual, equivalence | VALIDATED | Propositions 1, 3, and 4 in `docs/uav_recursive_feasibility_derivation.md`; nonsmooth limits stated |
| 5 | Single-obstacle support closed form | VALIDATED | Proposition 2, symbolic/numerical support checks; 100,000 cases, max error `1.42e-14` |
| 6 | Multi-obstacle primal, dual, Farkas witness, low-dimensional certificate | VALIDATED | Minimax dual theorem, KKT recovery LP, and 100,000 primal-dual checks |
| 7 | Explore `rho` as feasibility barrier | REJECTED | Explicit next-step counterexample and invalid-theorem section |
| 8 | Prove noncircularity or reject | REJECTED | Circular feasibility-CBF counterexample `u>=0.75`, `u<=0.25` |
| 9 | Formula-level feasibility-CBF prior-art audit | VALIDATED | `docs/theory_novelty_feasibility_matrix.md` and `docs/uav_theory_novelty_attack.md` |
| 10 | Physically derive braking/backup set | REJECTED | Fixed-direction stopping-distance derivation and multi-obstacle failure boundary in `docs/uav_recursive_feasibility_derivation.md` |
| 11 | Energy-optimal certified backup | REJECTED | No invariant multi-obstacle backup domain was obtained, so energy optimization cannot repair its missing certificate |
| 12 | Separate backup-safety and energy-optimality theorems | VALIDATED AS NON-RESULT | Neither theorem is claimed; their different missing premises are explicit in `docs/uav_theory_contribution_search.md` |

## Requirements 13--29: energy selector and certificates

| # | Requirement | Status | Authoritative evidence |
| ---: | --- | --- | --- |
| 13 | Remove arbitrary `lambda=0.1` as theory | VALIDATED | Candidate A retained only as engineering baseline |
| 14 | Safety-first constrained energy optimization | VALIDATED | Two-stage definition and implementation |
| 15 | Preserve nominal navigation without arbitrary weighted safety reward | VALIDATED, THEN FALSIFIED | Feasibility-preserving progress floor is derived; closed-loop navigation collapses |
| 16 | Lexicographic two-stage filter and derived reserve | REJECTED | Pointwise construction is feasible, but no recursive reserve is derivable; 69/334 success and 15 fallbacks |
| 17 | Derive physical convex action energy | VALIDATED | `TelemetryCostModel` quadratic, units, and PSD conditions in energy derivation |
| 18 | Do not equate instantaneous and trajectory energy | VALIDATED | Analytical counterexample and empirical separation |
| 19 | Explore one-step Energy-to-Go upper bound | VALIDATED CONDITIONALLY | Descent-lemma form derived |
| 20 | Derive exact ZOH control Jacobian and second-order term | VALIDATED | `B=[0.5T^2I;TI]`, Hessian identity checked by SymPy and 100,000 smooth cases |
| 21 | Form physical one-step upper objective | VALIDATED CONDITIONALLY | Convex quadratic only under verified smoothness/affine feature assumptions |
| 22 | State theorem only on certified trust region | VALIDATED AS CONDITIONAL LEMMA | No deployed certificate is claimed |
| 23 | Audit deployed estimator smoothness | REJECTED FOR DEPLOYMENT | ReLU gradient jumps and goal-direction singularity preclude a useful global bound |
| 24 | Separate deterministic and statistical semantics | VALIDATED | Conformal coverage is explicitly not used as a deterministic Hessian/Bellman certificate |
| 25 | Derive safety-induced energy overhead | VALIDATED | Exact quadratic increment and spectral upper bound in `docs/uav_energy_optimal_filter_derivation.md` |
| 26 | Pointwise energy dominance only on common feasible set | VALIDATED | Proposition 2; no trajectory claim |
| 27 | Search for trajectory-energy theorem | REJECTED | Different state sequences and fixed-power accumulation break implication |
| 28 | Explore energy supersolution | REJECTED | Bellman inequality is not verified by MC/conformal prediction |
| 29 | Joint energy-certificate constraint if supersolution exists | GATED_NOT_RUN | Premise failed; adding the constraint would invent an uncertified certificate |

## Requirements 30--39: candidate theory, falsification, and novelty

| # | Requirement | Status | Authoritative evidence |
| ---: | --- | --- | --- |
| 30 | Attempt integrated RF-energy sampled-data filter | VALIDATED AS SEARCH, REJECTED AS THEORY | Branch inventory in contribution search |
| 31 | Build definitions/lemmas/theorems only where valid | VALIDATED | `docs/theory_candidate_v1.md` distinguishes valid, conditional, and invalid statements |
| 32 | Counterexample search before accepting theorems | VALIDATED | Five analytic families plus 100,000 randomized and 1,002 trajectory falsification |
| 33 | SymPy plus at least 100,000 numerical states | VALIDATED | `artifacts/uav_theory_candidate_validation_20260819/report_100k_all_final.json` |
| 34 | Explicit inter-sample theorem/search | VALIDATED AS VERIFIER, COVERED AS SYNTHESIS | Exact quartic verifier; 74 endpoint-safe/inter-sample-unsafe examples; no convex recursive synthesis theorem |
| 35 | Braking-aware barrier and novelty audit | VALIDATED, NOT NOVEL | Fixed-direction formula derived; joint 3D/multi-obstacle theorem rejected and covered by braking/viability literature |
| 36 | Consider combination theorem | REJECTED | No recursive-feasibility component survived, so no non-equivalent joint theorem exists |
| 37 | Search recent 2025--2026 literature | VALIDATED | Primary-source matrix includes L4DC/ICML/CoRL and 2025--2026 arXiv work |
| 38 | Audit named newest work | VALIDATED WITH ONE LIMITATION | Backup, LSE-CBF, predictive, bandit, performance, sampling-aware, and ATOM-CBF were verified; no primary `LatentCBF L4DC 2026` source was found, so no claim relies on it |
| 39 | Apply strict novelty standard | VALIDATED | Formula-level matrix and independent reviewer score `2.34/5`; novelty false |

## Requirements 40--57: theorem-directed experiments

| # | Requirement | Status | Authoritative evidence |
| ---: | --- | --- | --- |
| 40 | Measure feasibility rather than success alone | VALIDATED | Margins, violations, infeasible steps, fallbacks, and solve tails reported |
| 41 | Test endpoint versus inter-sample safety | VALIDATED | Exact quartic check and swept-segment rollout collision metric |
| 42 | Separate instantaneous, one-step, and total energy | VALIDATED | Dedicated guarantee-separation section |
| 43 | Keep standard and Candidate A baselines | VALIDATED | Matched 334-scenario comparison |
| 44 | Do not spend resources on long CMDP yet | VALIDATED | No long CMDP run started |
| 45 | Grade contribution level honestly | VALIDATED | Only Level 4 engineering contribution remains |
| 46 | Prefer one defensible theorem | VALIDATED | Search stops rather than multiplying weak claims |
| 47 | Attack recursive margin first, energy upper bound second | VALIDATED | Both branches attempted in the requested order and rejected/conditioned |
| 48 | Paper-style theory document | VALIDATED | `docs/theory_candidate_v1.md` |
| 49 | Explain each assumption and failure without it | VALIDATED | Assumption section includes model, actuation, geometry, active set, energy, smoothness |
| 50 | Check physical units | VALIDATED | HOCBF slack, stopping distance, energy, and Taylor terms are annotated |
| 51 | State complexity dependence and measure 20 Hz | VALIDATED | Single obstacle/exact quartic `O(1)`, all quartics `O(K)`, joint dual dense solve up to `O(K^3)` per iteration, two-stage filter uses two serial solves; timing tails reported |
| 52 | Run symbolic, toy, 100k, 1000 trajectories before longer rollouts | VALIDATED THROUGH FAILURE GATE | Mandatory stages completed; 10k/50k were not run because candidate already failed navigation, energy, feasibility, and 20 Hz gates |
| 53 | Generate `HARD_FEASIBILITY_SET` | VALIDATED | 100,000 physical-state search and 1,000 retained hardest states in `artifacts/uav_hard_feasibility_set_20260819_v3/` |
| 54 | Reproduce 34 fallback type and apply proposed method on same/harder states | VALIDATED WITH PROVENANCE LIMITATION | Exact historical states were not logged; on 1,000 targeted harder states `method_audit.json` gives `0/1000` certified actions for all selectors sharing the sampled hard set |
| 55 | Distinguish certified backup from uncertified fallback | VALIDATED | Candidate has 15 uncertified fallbacks and no certified backup claim |
| 56 | Prevent uncontrolled energy regression | FAILED BY CANDIDATE | Lexicographic candidate is `+235.7%` versus standard and `+249.9%` versus Candidate A |
| 57 | Preserve navigation performance | FAILED BY CANDIDATE | 69/334 success, path ratio 3.456; freeze/detour behavior explicitly rejected |

## Requirements 58--61: claim downgrade and deliverables

| # | Requirement | Status | Authoritative evidence |
| ---: | --- | --- | --- |
| 58 | Automatic publication-claim downgrade | VALIDATED | `CLAIM_LEVEL = ENGINEERING_CONTRIBUTION_ONLY` |
| 59 | Independent novelty attack | VALIDATED | `docs/uav_theory_novelty_attack.md`; closest-formula rejection and AC synthesis |
| 60 | Produce six named final documents | VALIDATED | All six required `docs/uav_theory_*.md` files exist and are nonempty |
| 61 | Use exact final reporting fields | PENDING FINAL RESPONSE | Final handoff must use the requested field order and state `FORMAL 500K: NOT STARTED` |

## Completion decision

The exploration requirements are complete only after the hard-state artifact is
validated, the final documents are synchronized to its numbers, and regression
tests pass.  The scientific result is deliberately negative:

- no recursive-feasibility theorem was proved;
- no finite-horizon energy theorem was proved;
- no non-equivalent novelty survived the prior-art attack;
- no formal 500k run was started.

The negative result closes the requested search; it does not make the proposed
filter a theory contribution.

## Final verification record

- `tests/test_uav_safety_energy_filter.py`: `63 passed`.
- `review_bundle/tests/new_route`: `58 passed` with 19 deprecation warnings.
- Root `tests`: `293 passed`, 32 subtests passed, and one load-sensitive legacy
  certified-acceptance subtest reported `WATCHDOG_DEADLINE` instead of the
  expected stale-bundle reason during the full suite.
- The isolated legacy acceptance test then passed: `1 passed, 9 subtests
  passed`.  No legacy certified-route code was modified to mask the timing
  behavior.
- `git diff --check`, Python compilation, required-document existence, and
  hard-state artifact integrity all passed.
