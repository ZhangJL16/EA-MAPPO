# UAV Energy Delivery V3

The current energy experiment is an obstacle-free, pre-safety stage. Navigation
and safety readiness are intentionally separate:

- `navigation_energy_ready` checks navigation success, distance-bucket success,
  and path ratio. It gates battery calibration, Energy TD, and Phase 2.
- `navigation_safety_ready` checks boundary-contact rate. It is recorded but does
  not block the current energy experiment because the Safety Module is disabled.

Current artifact metadata must state:

```text
safety_module_enabled = false
obstacles_enabled = false
lidar_enabled = false
cbf_enabled = false
energy_experiment_stage = obstacle_free_pre_safety
energy_td_policy_context = frozen_navigation_policy_without_cbf
requires_retraining_after_safety_layer = true
```

Boundary-contact trajectories remain in Goal-conditioned Energy TD data because
the estimator models the realized frozen-policy trajectory and its realized
TelemetryCostModel energy. Held-out TD results are reported for all trajectories,
clean trajectories, and boundary-contact trajectories.

When a future Safety Module changes nominal actions into CBF-filtered executed
actions, trajectory, acceleration, and energy distributions change. Energy TD
must then be retrained or fine-tuned on executed safe trajectories; the current
critic is not claimed to transfer unchanged.
