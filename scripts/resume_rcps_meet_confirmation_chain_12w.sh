#!/usr/bin/env bash
set -euo pipefail

cd /home/zjl/mappo

UV=(uv run --python .venv/bin/python)
PROTOCOL=docs/SCENE_GROUPED_RCPS_MEET_PROTOCOL_V1.md
FREEZE=artifacts/certified_meet_fusion_freeze_150scenes_20260904_v2
CAL_SOURCE=artifacts/rcps_meet_calibration_nominal_450scenes_20260904_v1
CAL_FORK=artifacts/rcps_meet_calibration_fork_450scenes_20260904_v1
CALIBRATION=artifacts/rcps_meet_calibration_450scenes_20260904_v1
TEST_SOURCE=artifacts/rcps_meet_confirmation_nominal_150scenes_20260904_v1
TEST_FORK=artifacts/rcps_meet_confirmation_fork_150scenes_20260904_v1
TEST_EVAL=artifacts/rcps_meet_confirmation_150scenes_20260904_v1

"${UV[@]}" scripts/run_lidar_forked_action_data_gate.py \
  --source "$CAL_SOURCE" \
  --protocol docs/FORKED_ACTION_CAUSAL_DATA_GATE_PROTOCOL.md \
  --output-dir "$CAL_FORK" \
  --device cuda \
  --num-scenes 450 \
  --anchors-per-scene 4 \
  --num-workers 12 \
  --hold-steps 8 \
  --perturbation 0.45 \
  --resume

"${UV[@]}" scripts/calibrate_rcps_meet_threshold.py \
  --data-dir "$CAL_FORK" \
  --freeze-dir "$FREEZE" \
  --protocol "$PROTOCOL" \
  --output-dir "$CALIBRATION" \
  --device cuda \
  --num-scenes 450 \
  --task-seed 510001 \
  --world-seed 520001 \
  --alpha 0.05 \
  --delta 0.05 \
  --grid-size 2001

.venv/bin/python -c "import json,sys; r=json.load(open('$CALIBRATION/RESULT.json')); sys.exit(0 if r['promotable'] else 2)"

"${UV[@]}" scripts/run_grouped_counterfactual_safe_energy_gate.py \
  --artifact artifacts/jseb_navigation_repair_r3r4_uv_seed0_20260828_193904/R3 \
  --checkpoint artifacts/jseb_navigation_repair_r3r4_uv_seed0_20260828_193904/R3/phase1_navigation/checkpoint_transition_500000.zip \
  --battery-calibration artifacts/r3_fixed_baseline_energy_chain_recalibrated_v2/battery_calibration.json \
  --protocol "$PROTOCOL" \
  --output-dir "$TEST_SOURCE" \
  --device cuda \
  --num-scenes 150 \
  --num-workers 12 \
  --task-seed 610001 \
  --world-seed 620001 \
  --intervention-seed 630001 \
  --model-seed 640001 \
  --max-policy-steps 4000 \
  --nominal-source-only

"${UV[@]}" scripts/run_lidar_forked_action_data_gate.py \
  --source "$TEST_SOURCE" \
  --protocol docs/FORKED_ACTION_CAUSAL_DATA_GATE_PROTOCOL.md \
  --output-dir "$TEST_FORK" \
  --device cuda \
  --num-scenes 150 \
  --anchors-per-scene 4 \
  --num-workers 12 \
  --hold-steps 8 \
  --perturbation 0.45

"${UV[@]}" scripts/evaluate_rcps_meet_confirmation.py \
  --data-dir "$TEST_FORK" \
  --freeze-dir "$FREEZE" \
  --calibration-dir "$CALIBRATION" \
  --protocol "$PROTOCOL" \
  --output-dir "$TEST_EVAL" \
  --device cuda \
  --num-scenes 150 \
  --task-seed 610001 \
  --world-seed 620001 \
  --bootstrap-draws 10000 \
  --seed 650001
