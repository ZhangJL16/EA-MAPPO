# Recoverable Safety Experiment Protocol

## Frozen Infrastructure

The diagnostic reuses without training or retuning:

- frozen SAC navigation checkpoint;
- existing sampled-data second-order HOCBF filter;
- Candidate A behavior already present in the filter stack;
- repository UAV limits and timing;
- existing primitive obstacle representation.

It does not alter energy, task switching, or prior experiments.

## Runs

### Hand scenarios

`artifacts/uav_recoverable_safety_hand_v4_20260820/`

- 10 hand cases × 4 perception modes;
- 3,200 steps;
- 25 deduplicated events;
- validates every required family and record schema.

### Random current-frame search

`artifacts/uav_recoverable_safety_random1000_seed0_v2_20260820/`

- command stored in `run_manifest.json`;
- 10 hand cases plus 1,000 randomized cases;
- 80,800 steps;
- 556 events in 555 rollouts.

### Matched perception comparison

`artifacts/uav_recoverable_safety_matched250x4_seed1_v3_20260820/`

- command: `.venv/bin/python scripts/run_uav_recoverable_safety_diagnostics.py --output-dir artifacts/uav_recoverable_safety_matched250x4_seed1_v3_20260820 --random-scenarios 250 --seed 1 --workers 8`;
- 260 scenario instances, including ten hand cases;
- identical scenario instances evaluated under four perception modes;
- 1,040 rollouts and 83,200 filter steps;
- no neural training.

## Metrics Implemented

Safety and filter behavior:

- collision and near-collision steps;
- minimum clearance;
- infeasible and perceived-feasible/true-infeasible steps;
- fallback and constraint-violating fallback steps;
- boundary contacts.

Recoverability:

- strengthened and raw `rho`;
- positive-to-negative entries;
- one-step sampled counterfactual action;
- dropout, reappearance, stale sensing, braking and multi-obstacle flags.

Performance:

- net progress toward the scenario goal.

Compute:

- filter and `rho` mean, median, P90, P95, P99 and maximum latency;
- sampled recovery-search mean, median, P95 and maximum latency.

Perception:

- visible IDs, stale ages and expanded obstacle radius through the serialized sensing history.

## Compute Results

| Component | Median | P90 | P95 | P99 | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| Existing filter | 1.87 ms | 25.07 ms | 57.03 ms | 100.42 ms | 291.70 ms |
| Exact `rho` diagnostic | 9.41 ms | 54.37 ms | 106.82 ms | 311.81 ms | 1,112.35 ms |
| Combined diagnostic | 15.16 ms | 81.75 ms | 132.09 ms | 325.92 ms | 1,118.07 ms |
| Sampled recovery oracle | 237.16 ms | — | 1,166.14 ms | — | 3,970.03 ms |

Mean combined latency is under 50 ms, but tail latency is not. A real-time claim based only on the mean would be invalid.

## Reproducibility

Each artifact stores:

- exact command;
- git SHA;
- script source hash;
- frozen checkpoint path and SHA-256;
- seed and diagnostic configuration;
- scenario summary and raw hard states.

The final diagnostic sources used for the schema-complete runs hash to:

- `experiments/uav_recoverable_safety/diagnostics.py`: `021b7109aff79f9a89bdb545ece01d2403226b2fe94dc00f84b34536a2168c45`;
- `scripts/run_uav_recoverable_safety_diagnostics.py`: `036de1b0f48a90958f58cb315aa5b70085563e21d9ff0782e1c66afdd71653fa`;
- frozen SAC checkpoint: `df9a4354420a1a043b60d704292c6fe06ddd2127e937d3d9c72278b287c1ff0a`.

Parallel workers load the same frozen policy with CPU inference and evaluate disjoint scenario shards. Scenario generation is seed-reproducible.

## Closed Experimental Claims

- The positive-to-negative margin phenomenon is stable.
- Constraint persistence substantially reduces dropout-specific events.
- Simple bounded-acceleration inflation does not dominate constant-velocity propagation.
- The current experiment does not establish collision reduction, task success, energy benefit, or 20 Hz recoverability control.

## Why Stage 4 Was Not Started

The required novelty gate failed before a proposed controller existed. Routes A and B match stronger prior theory, Route C is not the observed dominant mechanism, and the diagnostic oracle misses the 20 Hz tail deadline. Per the research stop rules, no trained method or long multi-seed experiment was launched.
