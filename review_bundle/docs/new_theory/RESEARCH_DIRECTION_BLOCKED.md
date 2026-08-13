# RESEARCH_DIRECTION_BLOCKED

Date: 2026-08-13

## Decision

The current formulation of **Data-Driven Dual-Timescale Safety for Persistent UAVs** is blocked as an ICLR-level theory contribution.

This is a novelty decision, not a statement that the application is useless or that the conditional mathematics is false.

## Reformulations Attempted

1. **Independent dual-timescale critics:** collision risk plus distributional energy-to-charger and one-way stopping.
2. **Coupled dual-budget operator:** remaining collision budget induces the charger policy and a defective energy-return law.
3. **Filter-induced target transport:** admission-boundary visitation controls return-law drift and calibration debit across filter versions.
4. **Final data-identification view:** commitment-censored potential returns require randomized probes, positivity, and propensity-aware estimation.

The initial formulation plus three major redesigns were reviewed in four hostile rounds. The final blind panel scored theory `8/10`, novelty `3/10`, methodology `6/10`, and AC `4/10`.

## Prior Work That Covers the Core

- Budget-conditioned policies and Bellman operators: https://arxiv.org/abs/1903.01004
- Multiple distributional safety constraints: https://arxiv.org/abs/2301.10923
- Quantile constraints on cumulative cost: https://proceedings.neurips.cc/paper_files/paper/2022/hash/2a07348a6a7b2c208ab5cb1ee0e78ab5-Abstract-Conference.html
- Policy-coupled conformal decisions: https://arxiv.org/abs/2607.02206
- Off-policy return-distribution/risk estimation: https://proceedings.mlr.press/v151/huang22b.html
- Conformal target-policy return prediction: https://proceedings.mlr.press/v206/zhang23c.html
- Decision-making with selectively observed outcomes: https://proceedings.mlr.press/v139/wei21a.html
- Safe return-to-base filtering: https://arxiv.org/abs/2501.02620

Together these works cover the general mathematical mechanisms more strongly than the present specializations.

## Why the Remaining Formulation Is Insufficient

- Budget augmentation is known BMDP machinery.
- Distributional energy cost and tail thresholds are covered by distributional/constrained RL.
- Collision filtering and calibrated action selection are known.
- Return-law drift under policy changes follows generic coupling/occupancy perturbation.
- Coverage loss under TV shift is standard.
- Missing commitment outcomes, positivity, randomized logging, and inverse propensity estimation are standard selective-label/OPE ideas.
- Charger semantics, repeated sorties, and one-way commitment produce a clear application, but not a new central theorem.

The method can still be experimentally valuable, but its current theory can be summarized as a careful composition of known frameworks.

## What Would Be Needed to Restart

At least one of the following must appear:

1. a genuinely new finite-sample theorem for adaptive, selected, sequential charger-return calibration that is not implied by policy-coupled conformal/OPE;
2. a new minimax lower/upper bound specific to endogenous safety-filter updates that changes algorithm design beyond generic kernel perturbation;
3. a nontrivial active-probing result that jointly optimizes task throughput and distributional return identification under sequential safety budgets, improving known selective-label/safe-evaluation theory;
4. completed causal evidence strong enough to reposition the work as an empirical robotics/application paper rather than a theory paper.

No such result or evidence currently exists in the repository.

## Research Integrity Boundary

- `THEORY_ICLR_READY = FALSE`.
- `PAPER_ICLR_READY = FALSE`.
- The running E1 experiment is untouched and cannot resolve the novelty blocker because it tests only the collision-free energy-estimation special case.
- Unfinished experiments support no positive empirical claim.
- The old certified/recoverability route remains superseded and must not be reintroduced.
