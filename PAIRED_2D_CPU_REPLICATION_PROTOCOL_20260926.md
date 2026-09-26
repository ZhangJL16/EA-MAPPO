# Independent-seed CPU replication panel

Date: 2026-09-26. This panel uses otherwise idle CPU cores while the separate
16-environment-worker paired experiment runs. It is fixed before inspecting
either experiment's final outcomes. It does **not** pool trajectories or
checkpoints with the vectorized experiment.

Five paired seeds, 102–106, each run control and feedback, for ten independent
single-environment CPU learners. Each starts from the same verified v4 policy
weights, uses a new optimizer, and trains for 25,600 decisions on maps 8–31.
The two conditions within each seed share every setting except the extra
safety-feedback penalties specified in
`SERVER_2D_RESEARCH_HANDOFF_20260926.md`. Each PPO update contains one
512-decision rollout, with the frozen v4 network and other PPO settings.
Every 512 decisions is saved atomically; interrupted jobs resume from their
own checkpoint, beginning a fresh simulator episode. Seeds and conditions
have separate manifests, output directories, and logs. No intermediate
outcome selects a seed, map, reward, or checkpoint.

Only final checkpoints are validated on maps 32–39 in shielded and raw
diagnostic modes. The 160 independent evaluation jobs may use at most 16 CPU
processes at once. Holdout maps 56–71 remain untouched. This is a replication
panel for the **single-environment** PPO design; report it separately from
the vectorized primary diagnostic because their rollout boundaries differ.

Run from the repository root:

```bash
bash scripts/run_2d_shield_feedback_cpu_replicates.sh
```

Outputs are confined to
`artifacts/dual_constraint_2d_shield_feedback_cpu_replicates_20260926/`.
