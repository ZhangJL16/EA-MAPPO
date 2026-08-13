# Stage B 16x16 Navigation Transfer Gate

## Decision

`NAVIGATION_TRANSFER_VALID = FALSE`

The frozen 4x4 SAC checkpoint completed only 1 of 150 preregistered open-world 16x16 navigation sorties. Per the Stage B gate, no 16x16 return-to-go dataset was collected, no Energy Critic was evaluated or updated, and no switching experiment was run.

## Immutable Inputs

- Frozen checkpoint: `artifacts/phase1_sb3_sac_1m_gpu/seed0/checkpoint_step_1000000.zip`
- Frozen checkpoint SHA-256: `56f50be556888a2b567ea76b3c9726319187802e1327a78db48082de195c6fce`
- Scenario: `envs/navigation/scenarios/target_open_16x16.json`
- Obstacles: none
- Observation/action dimensions: 77/3
- SAC retraining or checkpoint modification: none
- Navigation sorties: 150, seed 0, stratified over seven distance bins
- Per-sortie limit: 800 steps
- Operational energy: telemetry only; exhaustion was not enforced

## Preregistered Gate Results

| Check | Required | Observed | Pass |
|---|---:|---:|---:|
| Overall completion | >= 0.90 | 0.0067 (1/150) | No |
| Task-goal completion | >= 0.85 | 0.0000 (0/75) | No |
| Charger-goal completion | >= 0.85 | 0.0133 (1/75) | No |
| Distance >= 8 m completion | >= 0.80 | 0.0000 (0/63) | No |
| Median completed-path efficiency | >= 0.50 | 0.9724, but only one completion | Not informative |
| Boundary-contact step rate | <= 0.10 | 0.0526 | Yes |

Overall timeout was 0.9933, mean final goal distance was 5.7469 m, and mean progress fraction was -0.5233. The failure is not restricted to insufficient rollout length: the short 0-2 m bin had zero completions and several sorties moved substantially farther from their goals.

## Distance-Stratified Results

| Initial distance | Sorties | Completion | Mean steps | Mean final distance (m) |
|---|---:|---:|---:|---:|
| 0-2 m | 22 | 0.0000 | 800.0 | 6.1330 |
| 2-4 m | 22 | 0.0455 | 766.4 | 5.9411 |
| 4-6 m | 22 | 0.0000 | 800.0 | 5.8012 |
| 6-8 m | 21 | 0.0000 | 800.0 | 5.7679 |
| 8-10 m | 21 | 0.0000 | 800.0 | 5.6759 |
| 10-12 m | 21 | 0.0000 | 800.0 | 5.3234 |
| >12 m | 21 | 0.0000 | 800.0 | 5.5557 |

## Root-Cause Diagnosis

Dimensional checkpoint compatibility did not imply semantic transfer compatibility.

1. **Sensor-scale shift.** LiDAR distance is divided by a fixed 2 m sensor range, while position and goal are divided by world size. Under 4x-matched source/target coordinates, the non-LiDAR state components align, but the LiDAR-distance observations differ because a 2 m ray covers a much larger fraction of the 4x4 map than of the 16x16 map.
2. **Policy response changed.** Across 122 legal matched coordinate pairs, the mean source/target LiDAR-observation L2 difference was 2.9990 and the mean action L2 difference was 1.3590. Mean initial action alignment with the goal dropped from 0.9482 in the 4x4 source observation to 0.5037 in the 16x16 target observation.
3. **Normalized transition-scale shift.** Physical `v_max`, `a_max`, and `dt` are unchanged, but XY position is normalized by map width. The same physical motion therefore changes normalized XY position four times more slowly in 16x16 than in 4x4.
4. **Normalized terminal-radius shift.** The fixed 0.2 m goal radius is 0.05 of a 4 m axis but only 0.0125 of a 16 m axis. The target terminal set is four times narrower in the policy's normalized XY coordinates.

These jointly change both the observation distribution and the normalized transition/terminal semantics seen by the frozen actor. The experiment therefore does not isolate Energy Critic transfer.

## Energy-Transfer Consequences

- 16x16 return trajectories collected: 0
- Valid 16x16 RTG labels: 0
- B0/B1/B2/B3 zero-shot metrics: not measured
- Source-to-target Energy Critic deterioration ratios: not measured
- Target-only or source-pretrained adaptation: not run
- Negative transfer: not assessable
- `ENERGY_BOOTSTRAP_TRANSFER_HAS_VALUE = UNCLEAR`
- `ENERGY_CRITIC_DOMAIN_SHIFT_EXISTS = UNMEASURED`
- `NAVIGATION_DOMAIN_SHIFT_EXISTS = TRUE`

Running energy estimation after this gate failure would confound frozen-policy failure with critic transfer. Autonomous switching and obstacle experiments must remain blocked.

## Next Minimal Experiment

Do not train Energy Critics yet. First determine whether navigation transfer can be made meaningful without altering the frozen checkpoint by evaluating a checkpoint-compatible scale protocol, such as preserving source-normalized sensor, transition, and terminal semantics through an environment coordinate wrapper. That would be a new protocol decision and must be tested separately; it is not part of the present bounded run.

## Artifacts

- `artifacts/energy_transfer/stage_b1_navigation_seed0_20260814/config.json`
- `artifacts/energy_transfer/stage_b1_navigation_seed0_20260814/raw_navigation.jsonl`
- `artifacts/energy_transfer/stage_b1_navigation_seed0_20260814/results.json`
- `artifacts/energy_transfer/stage_b1_navigation_seed0_20260814/scale_mismatch_diagnostic.json`
- `artifacts/energy_transfer/stage_b1_navigation_seed0_20260814/COMPLETED.json`
