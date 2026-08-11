# Certified Execution Training Loop

This track restores the theorem-facing online execution path as the main energy experiment.

- The clean 77-dimensional absolute-coordinate actor observation is unchanged.
- Standard Stable-Baselines3 SAC proposes normalized physical acceleration in `[-1, 1]^3`.
- The environment maps each normalized SAC action through the current verified support as `a = c + G @ u`.
- The frozen κ takes over when normal support is unavailable or the certified energy trigger closes normal authority.
- At the charger, certified hold remains active until both the departure-energy gate and the first departure action certificate pass.
- The original pending task goal is not replaced by recovery, charging, hold, or departure.
- SB3 replay stores its own normalized support coordinate; rewards and successors come from the physical action produced by the certified adapter.
- There is no recovery-teacher replay prefill and no supervised actor warm start.
- `learning_starts=5000` remains the standard SAC replay-population setting; it is not demonstration warmup.

The formal run uses `flight_energy_multiplier=2.0` and a certificate artifact bound to the exact current runtime dependencies. A stale 1x or stale 2x artifact fails closed.

Run one seed:

```bash
cd review_bundle
PHYSICAL_GPU_INDEX=0 bash scripts/run_one_sb3_energy_2x_certified_execution_1m.sh 0
```

Launch all seeds in separate tmux sessions:

```bash
cd review_bundle
PHYSICAL_GPU_INDEX=0 bash scripts/launch_sb3_energy_2x_certified_execution_1m.sh
```

The launcher refuses to overwrite `artifacts/phase2_sb3_sac_energy_open_1m_2x_certified_execution/seed*`.
