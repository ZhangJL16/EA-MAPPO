# R3 SAC Accelerated-Implementation Reproduction

## Objective

Retrain R3 SAC from random initialization at seed 0 while preserving its frozen
training and 500-task evaluation contract. The only intended implementation
change is the array/native execution of the same ordinary top-16 HOCBF
constraints and projection problem.

## Frozen baseline

- Baseline artifact: `artifacts/jseb_navigation_repair_r3r4_uv_seed0_20260828_193904/R3`
- Baseline git SHA: `7e30d8f5b6928bbb505b9eaa3873c1791590400b`
- Seed / environments / budget: `0 / 8 / 500000`
- Training vectorization: `SubprocVecEnv`, `forkserver`, one native thread per worker
- SAC gradient schedule: `gradient_steps=-1`
- Observation/safety: structured 2 x 8 x 128 LiDAR, ordinary HOCBF, top K = 16
- R3 bridge: recent-window 512, trust region 0.35, shield loss weight 0.10
- Evaluation: 500 immutable tasks, seed 170001, six parallel workers
- Source-evaluation SHA-256: `ca4ae3a11b129948a4c4b22a27973da74e8ad006c69a08b3c6343b2e14172a6e`
- Selection-task SHA-256: `ead4b79839f0af425896ef27d455d4d6ea78afd60cda7e28e6f3ffbef1cd07c9`

## Baseline target

The original R3 result achieved 96% success and mean path ratio 1.188026 over
500 tasks, but did not pass the safety gate: it recorded 2,242 obstacle-collision
steps and four boundary-contact episodes. Reproduction means measuring whether
the accelerated implementation recovers this distribution; it does not relabel
the original run as fully gate-passing.

## Preflight evidence

- Frozen-contract checks: 13/13 passed.
- Focused parallel-environment, Jacobian-SAC, and HOCBF tests: 110 passed.
- CUDA/Subproc smoke: exact 2,000-transition budget and five-task terminal
  evaluation completed with both checkpoints present.

## Active run

- Artifact: `artifacts/r3_sac_accelerated_reproduction_seed0_500k_formal500_20260902_v1`
- Process group / trainer PID at launch validation: `204721 / 204724`
- First scheduled log: 8,000 transitions, 2,992 gradient updates, finite reward,
  zero obstacle/boundary contacts, and 0.167875 HOCBF intervention rate.
- Native activation: all eight forkserver workers mapped `_qp_native.so`.

The run trains from scratch to 500,000 transitions, saves checkpoints every
50,000 transitions, and then executes the immutable 500-task evaluation.
Runtime output, source hashes, command, checkpoint hash, training audit, and
evaluation records remain under the versioned artifact directory.
