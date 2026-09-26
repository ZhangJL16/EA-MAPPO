# ARM CPU 16-worker paired 2D safety-feedback experiment

Date: 2026-09-26. This is a new execution protocol prompted by the request to
use 16 CPU processes. The earlier single-environment run under
`artifacts/dual_constraint_2d_shield_feedback_arm_cpu_20260926/` is retained
only as a startup diagnostic. Its checkpoints are **not** mixed with this run.
Changing PPO's vectorization changes rollout boundaries and optimization, so
the parallel run is not a bitwise continuation of the single-environment run.

The scientific comparison remains the frozen handoff's two conditions. Both
start from the same SHA-256-verified v4 final **policy weights**, with fresh
optimizers. Both use the same shielded plant, 13-action mode mask, training
maps 8–31, seed 101, 600-second terminal, gamma 1, GAE lambda 1, and the same
PPO network, batch size, epochs, learning rate, clipping, and entropy setting.
Control retains the existing -0.2 forced-return penalty. Feedback adds -1.8
per forced-return takeover and -2.0 per rejected departure, with no extra
penalty for ordinary QP correction.

Each condition has eight independent `SubprocVecEnv` workers, seeded 101–108,
each drawing from the same 24 training maps. The two conditions run
concurrently: **16 environment worker processes** plus two CPU learner
processes. Each worker collects 64 high-level decisions per PPO update; each
condition therefore still has 512 transitions per update and 25,600 total
transitions. This vectorized sampling design is fixed before inspecting any
parallel result. All numerical libraries and PyTorch are restricted to one
thread per process, and the launcher rejects a CUDA-enabled PyTorch wheel.

Both conditions save model, optimizer, status, and episode records after each
512-transition update. An interrupted condition resumes from its own latest
checkpoint; it starts new simulator episodes and is not bitwise identical to
uninterrupted execution. A single 512-transition update per condition is the
startup smoke; its reward and outcome are not used to alter this protocol.

Only the final 25,600-transition checkpoint of each condition is evaluated.
Validation uses maps 32–39 in shielded and unshielded diagnostic modes, with
at most 16 independent CPU evaluation processes. The raw mode is not a
deployable policy. Holdout maps 56–71 remain untouched. No intermediate
checkpoint or validation outcome selects parameters, seeds, maps, or workers.

Run from the repository root:

```bash
bash scripts/run_2d_shield_feedback_arm_cpu_16workers.sh
```

Outputs are confined to
`artifacts/dual_constraint_2d_shield_feedback_arm_cpu_16workers_20260926/`.
This is a paired diagnostic of safety-feedback learning in the existing
single-target environment; it does not add an order queue or claim that
parallel PPO is directly comparable to the earlier one-environment run.
