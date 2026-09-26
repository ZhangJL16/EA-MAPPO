# ARM64/CPU execution of the next paired 2D diagnostic

Date: 2026-09-26. The 2D research baseline and archived results are the
remote branch at `5a6a561`. The active 3D return matrix is local and runs
independently. This document implements the next experiment specified in
`SERVER_2D_RESEARCH_HANDOFF_20260926.md`; it does not revise the previous
CUDA experiments or reinterpret their results.

The two conditions start from the same frozen v4 final **policy weights**,
SHA-256 `71ca7c3382f4fb994f90a1334b70fcece5ea5fdf1cb98b3319e069b696858429`.
Each creates a fresh CPU PPO optimizer and starts its own timestep count at
zero. Both train with the same shielded plant on maps 8–31, seed 101, for
25,600 decisions, with 512-step rollout/checkpoint intervals, gamma 1,
GAE lambda 1, the same 13-action mode mask and other v4 PPO settings.
Control retains the existing 0.2 forced-return penalty. Feedback adds 1.8
per forced return and 2.0 per rejected departure, without changing the
plant, safety filter, energy dynamics, or ordinary QP intervention reward.

Only the final 25,600-step checkpoints are evaluated. Each condition runs
paired, deterministic shielded and raw-policy diagnostics on validation maps
32–39. Raw replay is a failure diagnostic, not a deployable policy. The
fresh holdout maps 56–71 remain untouched. A 512-step startup checkpoint
can be run with `--max-new-timesteps 512`; its metrics are not an experiment
outcome. The full launcher resumes those checkpoints and runs every remaining
job without choosing settings from intermediate results.

The ARM64 host uses an isolated workspace environment at
`../.venv-2d-cpu/`, separate from the running 3D environment. It has the
CPU-only PyTorch 2.7.1 wheel, SB3 2.8.0, Gymnasium 1.2.3, NumPy 1.26.4,
SciPy 1.17.1, and OSQP 1.1.3. Each training and evaluation entrypoint
selects `device="cpu"` explicitly. The launcher rejects a CUDA-enabled
wheel, hides accelerators, and limits numerical libraries to one CPU thread
per process. It records architecture, dependency versions, source hashes,
parent checkpoint hash, and calibration hash in each output manifest.

Full run from the repository root:

```bash
bash scripts/run_2d_shield_feedback_arm_cpu.sh
```

Results are written only under
`artifacts/dual_constraint_2d_shield_feedback_arm_cpu_20260926/`.
Checkpoints are atomic and resume from the last completed 512-step update.
As in the historical PPO code, a resumed process loads model and optimizer
but starts a new simulator episode, so replay is resumable rather than
bitwise identical to an uninterrupted run. The two conditions must retain
separate manifests and output directories.
