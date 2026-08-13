> **LEGACY / SUPERSEDED:** Historical research provenance only. This file is not an active method, theorem, implementation, test, or README authority. The active route is in docs/new_theory/.

# EA-MAPPO Research Results Summary

Date: 2026-08-11  
Scope: repository artifacts available in this snapshot; synthetic/software evidence unless stated otherwise.

## Executive conclusion

The project has one completed learning result, one conditionally coherent safety/runtime line, and several incomplete research hypotheses:

1. **Completed learning result:** Standard SB3 SAC solves the open random-goal navigation baseline after 1M steps on three training seeds.
2. **Conditional theory/runtime result:** Persistent Generator-SAC now has a detailed mixed continuous/atomic Bellman formulation, verified-support runtime, frozen recovery authority, charging/departure/task-resume lifecycle, and extensive adversarial software regressions. These establish conditional synthetic/software properties only, not physical UAV safety.
3. **Negative empirical result:** the learned Generator residual has not shown a material advantage over the task-aware verified center. Current fixtures are center-dominated.
4. **Incomplete charging result:** neither the historical 1x nor current 2x policy-only studies establish reliable autonomous charge-return-resume behavior. Certified execution prevents sampled collision/stranding in early pilots but has not completed tasks in the available checkpoints.
5. **Unvalidated hypotheses:** the dual learned fields and recovery teacher are implemented/protocolized, but have no completed causal evidence.
6. **Submission status:** the bundle is not ICLR-ready. Learned contribution, novelty differentiation, final exact-hash proof review, a green current full suite, and physical/HIL evidence remain open.

## 1. Completed Standard SAC navigation baseline

Source: `artifacts/phase1_sb3_sac_1m_gpu/seed*/`.

Each of three seeds trained for 1,000,000 steps. Deterministic held-out evaluation used five environment streams and 25,000 steps per training seed.

| Metric | Seed 0 | Seed 1 | Seed 2 | Mean ± sample SD |
| --- | ---: | ---: | ---: | ---: |
| Held-out tasks / 1,000 steps | 26.52 | 27.00 | 26.72 | 26.747 ± 0.241 |
| Held-out collision rate | 0 | 0 | 0.00012 | 0.000040 ± 0.000069 |
| Held-out median steps / completed goal | 37 | 37 | 37 | 37 ± 0 |
| Successful held-out streams | 5/5 | 5/5 | 5/5 | 100% |
| Training tasks completed | 11,215 | 14,939 | 14,415 | 13,523 ± 2,016 |
| Final 50k training tasks / 1,000 steps | 25.58 | 26.70 | 26.12 | 26.133 ± 0.560 |

Supported claim: Standard SAC learns the open navigation task in this synthetic environment. This baseline does not include persistent charging or certify the primary Generator-SAC method.

## 2. Energy and charging experiments

### Historical 1x study

Source: `artifacts/phase2_sb3_sac_energy_open_1m/`.

The runs were interrupted for a protocol change. Latest available training checkpoints are 300k, 300k, and 200k steps for seeds 0, 1, and 2. Training throughput is only 0.123, 0.123, and 0.050 tasks per 1,000 steps. Stranded-episode rates are 0.983, 1.000, and 0.975. Logged charging sessions can complete and record same-task resume, but the held-out policy does not autonomously produce stable complete cycles: seed-0 deterministic full-SOC evaluation at 300k completes 4 tasks over 25k steps, performs no charging sessions, and strands in every stream.

Conclusion: charging mechanics work in exercised traces, but autonomous persistent charging is not learned.

### Current 2x early checkpoints

These directories are named `1m` because that is the requested budget; the available artifacts are not 1M completions.

| Condition | Available seed/checkpoint | Training tasks / 1k | Collision rate | Stranded episode rate | Training charge/resume observation |
| --- | --- | ---: | ---: | ---: | --- |
| Unguided | seed 0 / 100k | 0.01 | 0.01246 | 1.0 | no charging sessions |
| Recovery-guided | seed 0 / 50k | 0.02 | 0.03822 | 1.0 | no charging sessions |
| Certified execution | seed 1 / 10k | 0 | 0 | 0 | 31 sessions; charge/resume rate 0.968 |
| Certified execution | seed 2 / 10k | 0 | 0 | 0 | 19 sessions; charge/resume rate 1.0 |

Policy-only held-out evaluations remain poor: they usually complete zero tasks and strand at rate 1.0. Available system-with-kappa held-out evaluations have zero sampled collisions and zero stranding, but also zero completed tasks; most recorded charging sessions do not complete within the evaluation window. Thus certified execution is currently a safety/lifecycle diagnostic, not a performance result.

The original seed-0 unguided/guided processes were archived under `*_interrupted_concurrent_code_20260811_154126` after a source-mutation guard fired. Those artifacts are provenance records and are invalid for formal cross-condition comparison.

## 3. Generator-SAC contribution result

Source: `artifacts/rl_contribution/`, `DERIVATION_PACKAGE.md`, and `artifacts/generator_center_ablation/results.json`.

Across five seeds and 20 evaluation episodes per seed, Center-Only, Random-in-Generator, and Generator-SAC all obtain task and return success 1.0 with zero sampled collisions in the open and obstacle synthetic missions. Generator-SAC exactly matches the 226-step Center-Only open result and improves the obstacle mean by only 0.15 step, about 0.00040 m path length, and 0.00164 synthetic energy units.

On 20 held-out certified scenarios, Generator-SAC and Center-Only have the same success pattern: open 1.0, obstacle 1.0, narrow 0, energy-tight 0.20; return success is 1.0 throughout.

Supported interpretation: the task-aware verified center supplies almost all demonstrated competence. The learned residual is mathematically and operationally integrated, but a meaningful residual-learning benefit is not established.

## 4. Conditional safety, energy, and lifecycle results

The current method line separates:

- verified task support from the frozen recovery controller;
- continuous Generator actions from atomic kappa/hold/fail-closed Bellman branches;
- current swept-tube and within-step energy-prefix checks from endpoint-only checks;
- lower battery energy from nominal energy in return/departure gates;
- command publication coverage from post-step hybrid successor classification;
- task identity from task-completion claims.

The synthetic 2x energy domination artifact checks every cell analytically, rather than by random sampling:

| Scenario | Nonterminal cells | Terminal cells | Minimum one-step slack | Violations |
| --- | ---: | ---: | ---: | ---: |
| Open | 88,144 | 449 | 0.0031119521 | 0 |
| Obstacle | 47,228 | 298 | 0.0031119519 | 0 |
| Energy-tight | 88,144 | 449 | 0.0031119521 | 0 |

Source: `artifacts/theory/two_x_energy_domination_action_rule_v2.json`.

This supports the synthetic atlas-energy contract under its declared model. It does not establish aircraft calibration, physical disturbance bounds, WCET, actuator-bus atomicity, HIL behavior, or real-flight safety.

## 5. Proof and software status at this snapshot

The recorded `PROOF_AUDIT.md` is a conditional same-family PASS for an older exact hash tuple. The current canonical files have different hashes, so that audit must not be presented as a current-hash proof verdict. A fresh proof audit is required after the repository is frozen.

Current validation performed during this summary:

- `python3 -m compileall -q review_bundle`: PASS.
- terminal `FAILURE`/`SUCCESS` phases are now explicitly ineligible for covered kappa publication; the focused regression passes.
- the current 2x certificate was deterministically regenerated and rebound to the current recovery manifest/atlas/terminal/kappa versions.
- the former seed-0 `post-departure-not-in-R_RL` lifecycle failure is repaired at the software-contract scope. Root cause was a two-part handoff defect: the scalar battery gate opened before the action-specific robust successor retained the normal switching margin, and terminal-set overlap erased the support-selected normal successor identity after mode change. Runtime now keeps charging under an exact certified hold until the full handoff gate passes, commits the selected candidate normal-authority cell, and transactionally reads back the realized successor before changing to `TASK_RL`. The exact full lifecycle and direct departure-successor regressions pass. The old artifact field is retained for compatibility but means `post_departure_normal_authority`, not implementation of state-level `R_RL`.
- a complete current-snapshot test suite and fresh exact-hash hostile review remain pending.

Historical documents that mention a prior `440-test` pass describe earlier byte snapshots. They are useful provenance but do not replace the current rerun.

## 6. Dual fields and recovery teacher

The dual-field code implements task-independent features and training-only proposal ranking inside an already verified Generator support. The fields do not participate in the safety certificate. There is no offline label package, trained calibrated checkpoint, shadow-mode result, or causal ablation; therefore no efficiency claim is supported.

The recovery-teacher protocol separates replay prefill and actor warm start conceptually, but the no-guidance / prefill-only / warm-start-only / both factorial is incomplete. Available guided and unguided 2x checkpoints are different ages and only one seed, so they cannot establish teacher benefit.

## 7. Claim ledger

| Claim | Verdict |
| --- | --- |
| Standard SAC solves open synthetic navigation at 1M | **Supported** |
| Synthetic 2x atlas one-step and cumulative energy bounds pass | **Supported conditionally** |
| Current runtime embodies the intended authority branches | **Supported narrowly by targeted lifecycle regressions; complete current-hash suite pending** |
| Learned Generator residual improves over center-only | **Not supported** |
| Policy autonomously learns reliable charging and task resume | **Not supported** |
| Dual learned fields improve data efficiency | **Not supported** |
| Recovery teacher improves 2x learning | **Not supported** |
| Current theorem snapshot has an exact-hash proof PASS | **Not supported; recorded audit is stale** |
| Physical UAV safety is established | **Not supported** |
| Current bundle is ICLR-ready | **Not supported** |

## 8. Next decisive experiments and gates

1. Freeze the source, run the complete suite, and repeat the exact-hash proof and three-role method review.
2. Decide whether to implement the optional energy-augmented state-level `R_RL`; do not relabel the current candidate-kernel handoff as that stronger result.
3. Build a non-saturated certified family where Center-Only is feasible but suboptimal; rerun matched Center / Random-in-Generator / Generator-SAC ablations.
4. Complete multi-seed 2x policy-only and system-with-kappa studies with equal checkpoints and full/low-SOC evaluations.
5. Run the teacher 2x2 factorial and a matched no-field / separate-field / coupled-field ablation before retaining either as a paper contribution.
6. Treat calibration, deployment timing, HIL, and real flight as independent gates rather than consequences of software tests.

## Upload snapshot

Before this summary was added, the untracked research snapshot contained 213 files and 40,699,654 bytes. The largest untracked file was 3,874,032 bytes, below GitHub's 100 MB single-file limit. It includes partial checkpoints, raw JSON/JSONL evaluations, interrupted-run provenance, theory artifacts, certificates, scripts, tests, and review reports. Interrupted and smoke artifacts are retained for provenance and explicitly excluded from formal performance claims above.
