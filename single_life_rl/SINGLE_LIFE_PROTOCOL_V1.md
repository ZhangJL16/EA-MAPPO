# Single-Life Protocol v1

Status: pre-outcome specification, to be locked by `freeze_protocol.py` with source,
configuration, job-plan, physical dependency and Git hashes. No production result
may alter any parameter or instance. Changes after freezing require a new version
and preservation/invalidation explanation; no silent patching of scientific code.

## Safety and scope

Strict safety: sup_M P_M(tau_C<infinity)=0, used for the impossibility theorem.
Learning safety: sup_M P_M(tau_C<infinity)<=delta for the full learning-and-deployment
life. Primary delta=.05; delta_epi=delta, delta_alea=0. Physical regeneration never
renews the statistical risk budget. Noise affects observations, not conditional
action safety. An anytime-valid likelihood-ratio confidence sequence and robust
menu enforce the lifetime theorem; empirical frequencies are not its proof.
Wrong-model exclusion, wrong certification and actual catastrophe are distinct.
Unsafe baselines are explicitly outside the guarantee.

The finite-menu theorem and staged-family bounds, with proofs and limitations,
are in paper/THEORY.md. Do not resurrect the rejected zero-risk iff or the original
all-pair Gamma. Stage Gamma distinguishes currently observable blocks.

## Fixed synthetic suites

Exact integers, seeds and method lists are in configs/protocol_v1.json.
Each independent run is one life with absorbing catastrophe, never a reset/retry.
All probes return to the same regenerative state if conditionally safe. Baseline
reward .2 and probe reward .2 per completed cycle; probe duration 1 except random
trees. Correct specialized exploit reward 1 and duration 1; incorrect exploit or
wrong-prefix probe causes catastrophe at the end of its duration, reward 0.
After catastrophe cumulative reward is held constant until T. A partially completed
cycle earns no reward. epsilon=0; known deterministic model-specific safety.

- Binary: kappa={0,.02,.04,.08,.16}, both true bits, 200 seeds each, T=20000.
  Oracle, RobustOnly, StaticSafeID, SafeRefine, UnsafeMLE: 10000 primary runs.
- Recursive: (kappa1,kappa2) in {.04,.08}^2, all four true models, 100 seeds,
  T=30000. Same five methods: 8000 runs.
- Nuisance: m={0,4,8,16,32}, kappa=.08, 100 seeds, T=50000. SafeRefine and
  FullModelID: 1000 runs. Relevant bits follow the recursive unlock. Each nuisance
  bit has an always-safe independent Bernoulli probe with the same kappa, reward
  and duration. Truth is a fixed seeded 34-bit vector, prefix-shared across m.
  Both methods prioritize relevant bits, use identical per-coordinate lifetime
  budgets and information design; only FullModelID continues to singleton.
- Random: 100 complete depth-3 binary trees; randomized leaf labels, each internal
  node kappa Uniform(.04,.12), duration uniform integer 1..5. Fixed tree seed
  20260922. Each node probe safe exactly on its descendant leaves. Each of 50
  seeds chooses a true leaf uniformly. SafeRefine, StaticSafeID, FullModelID,
  Oracle: 20000 runs, T=30000. Topology is a fixed balanced binary tree with random
  labels/channel parameters, not a claim of arbitrary graph diversity.
- Sensitivity: binary SafeRefine only, all the same instances, delta=.1,.01,.001
  in addition to primary .05: 6000 additional runs. No extra sensitivity suites.

Relevant tree bit j uses alpha_j=delta/2^(j+1); binary uses alpha_0=delta.
For m nuisance bits, keep the relevant budgets delta/2 and delta/4 and split
the remaining delta/4 uniformly: each nuisance bit gets delta/(4m).
A bit is certified at absolute cumulative log-likelihood >=log(1/alpha_j).
There is no fixed-time CI or per-cycle delta reset. Already certified bits retain
their evidence and budget. Sampling uses independent named probe streams with
fixed seed derivation, shared across methods. No adaptive reseeding.
When no informative robust-safe probe remains and no common policy exists, report
UNIDENTIFIABLE_WITH_AVAILABLE_MENU and use robust baseline for the remainder.
Greedy unsafe foil chooses the lexicographically first MLE (all zero bits before
observations) and immediately exploits. This weak but transparent foil is fixed;
no outcome-driven exploration warmup.

## Fixed UAV semi-empirical suite

Generate exactly 200 legal service locations with numpy default_rng(20260921),
using the existing legal_position sampler and frozen map. No replacement for
navigation failures. Pilot capacity 1e9. Reuse unchanged frozen SAC, HOCBF,
collision recovery, energy model and 4000-policy-step per-leg timeout. Fly
charger→task→charger physically, stopping velocity at service with the existing
service semantics; no reset between outbound and return. Independent library
missions start at the canonical charger. Record every success/failure, duration,
actual energy and unified contact count. Pilot data generation is separate from
formal learning and is not itself a real single-life safety guarantee.

Use successful missions only as deterministic cycle arms; retain excluded timeout
records. B is linear-interpolated q60 of their energy, r=B/109.05. This rule is
fixed before library measurements; materialized B/library are hashed before
learning runs. No successes => UAV suite structurally unavailable, recorded for
all planned jobs; never draw replacements or stop other suites.
Theta={.85,.925,1,1.075,1.15}; true energy=theta E; arm safe iff theta E < B.
Energy sensor Y~Normal(theta E,(.03 E)^2). Exact duration T_i+theta E_i/r and reward
are also observable; do not hide the clock or claim Gaussian-only KL complexity.
Model support can shrink after one cycle. All-safe same optimal arm means policy
certified at time zero. The learner uses the same robust-menu/common-answer/
likelihood principles; exact-support observations are zero-error exclusions.

100 seeds per theta, T=20000 seconds. Oracle, WorstCaseRobust, SafeRefine,
CertaintyEquivalent: 2000 runs. Oracle objective is max safe 1/cycle-duration;
WorstCaseRobust optimizes theta_max; CE uses lexicographic minimum-theta MLE ties.
SafeRefine chooses shortest worst-case safe cycle if it needs information: each
arm's exact clock provides singular information. No new energy model is fitted.

Library E,T do not specify within-leg depletion timing. Formal semi-Markov arms
place success/catastrophe at scheduled cycle completion; unfinished cycles at T
are censored with no reward. This is an explicit coarse abstraction, not physical
catastrophe timing, persistent queue dynamics or deployable UAV safety validation.
Regenerative cycles count measured flight time plus charging; there is no free
instant refill in the time accounting.

## Primary metrics and complete analysis

Total 47000 learning jobs plus 200 physical library missions. Primary metrics:
finite-window catastrophe indicator (calibration of lifetime theorem),
T*rho_oracle - reward(T), time to policy certification, exploration time.
Auxiliary: safe probe count, cycles, version cardinality, number of candidate
policy answers (NOT equivalence classes of overlapping sets), unlocked experiments,
true model exclusion, per-stage likelihood/evidence and full-model ID time.

Report 95% Wilson binomial intervals for fixed-condition cells. Pooled heterogeneous
cells are descriptive only. Report certification fractions and restricted mean
min(T_cert,T), treating all noncertified lives including failures as censored at T.
Never exclude failure runs or silently condition a scaling claim on certification.
Plots: binary inverse-information vs restricted time; log(1/delta) vs restricted
time; nuisance dimension vs exploration; random-tree path complexity vs time.
No visual threshold or observed curve triggers a new run or a parameter change.

## Execution and provenance

From repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /mnt/workspace/zjl-exp/.venv/bin/python \
  -m single_life_rl.scripts.run_all \
  --config single_life_rl/configs/protocol_v1.json --workers 16
```

The entry freezes/verifies config, Git, sources, physical dependencies and all jobs;
executes library→binary→recursive→nuisance→random→UAV in that order; waits for all
workers; verifies coverage; then collects/analyzes once. No scientific CLI overrides.
Output location is operational only. Repeating the same entry resumes immutable
results. Each physical worker has an exclusive lock and checksummed immutable
midflight checkpoints. Synthetic jobs are atomic independent lives: an interrupted
unfinished job restarts with identical seed/contract, never removes a completed
failure. Driver errors stop execution and retain checkpoints; no automatic retuning.
Operational health checks may inspect finiteness, counts and errors, not performance
for experiment selection. No additional experiments, neural training or legacy
persistent-UAV changes are authorized by these results.
