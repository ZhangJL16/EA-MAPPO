#!/usr/bin/env bash
set -euo pipefail

cd /home/zjl/mappo

UV=(uv run --python .venv/bin/python)
SOURCE=artifacts/certified_meet_fresh_nominal_source_75scenes_20260904_v1
FORK=artifacts/certified_meet_fresh_fork_75scenes_20260904_v1
EVAL=artifacts/certified_meet_fresh_confirmation_75scenes_20260904_v1

"${UV[@]}" scripts/run_grouped_counterfactual_safe_energy_gate.py \
  --artifact artifacts/jseb_navigation_repair_r3r4_uv_seed0_20260828_193904/R3 \
  --checkpoint artifacts/jseb_navigation_repair_r3r4_uv_seed0_20260828_193904/R3/phase1_navigation/checkpoint_transition_500000.zip \
  --battery-calibration artifacts/r3_fixed_baseline_energy_chain_recalibrated_v2/battery_calibration.json \
  --protocol docs/CERTIFIED_MEET_FUSION_CONFIRMATION_PROTOCOL_V2.md \
  --output-dir "$SOURCE" \
  --device cuda \
  --num-scenes 75 \
  --num-workers 8 \
  --task-seed 420001 \
  --world-seed 430001 \
  --intervention-seed 440001 \
  --model-seed 450001 \
  --max-policy-steps 4000 \
  --nominal-source-only

"${UV[@]}" scripts/run_lidar_forked_action_data_gate.py \
  --source "$SOURCE" \
  --protocol docs/FORKED_ACTION_CAUSAL_DATA_GATE_PROTOCOL.md \
  --output-dir "$FORK" \
  --device cuda \
  --num-scenes 75 \
  --anchors-per-scene 4 \
  --num-workers 8 \
  --hold-steps 8 \
  --perturbation 0.45

"${UV[@]}" scripts/evaluate_certified_meet_fusion.py \
  --data-dir "$FORK" \
  --freeze-dir artifacts/certified_meet_fusion_freeze_150scenes_20260904_v2 \
  --protocol docs/CERTIFIED_MEET_FUSION_CONFIRMATION_PROTOCOL_V2.md \
  --output-dir "$EVAL" \
  --device cuda \
  --bootstrap-draws 10000 \
  --seed 450001
