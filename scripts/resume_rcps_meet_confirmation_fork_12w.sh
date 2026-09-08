#!/usr/bin/env bash
set -euo pipefail

cd /home/zjl/mappo

UV=(uv run --python .venv/bin/python)
PROTOCOL=docs/SCENE_GROUPED_RCPS_MEET_PROTOCOL_V1.md
FREEZE=artifacts/certified_meet_fusion_freeze_150scenes_20260904_v2
CALIBRATION=artifacts/rcps_meet_calibration_450scenes_20260904_v1
TEST_SOURCE=artifacts/rcps_meet_confirmation_nominal_150scenes_20260904_v1
TEST_FORK=artifacts/rcps_meet_confirmation_fork_150scenes_20260904_v1
TEST_EVAL=artifacts/rcps_meet_confirmation_150scenes_20260904_v1

# Do not resume if the upstream risk certificate has changed.
.venv/bin/python -c "import hashlib,json,pathlib,sys; d=pathlib.Path('$CALIBRATION'); r=json.load(open(d/'RESULT.json')); sha=lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest(); ok=r.get('status')=='FROZEN_FOR_RCPS_CONFIRMATION' and r.get('promotable') is True and sha(r['protocol_document'])==r['protocol_sha256'] and sha(pathlib.Path(r['freeze_dir'])/'frozen_meet.pt')==r['frozen_checkpoint_sha256'] and sha(d/'risk_curves.npz')==r['risk_curves_sha256']; sys.exit(0 if ok else 2)"

"${UV[@]}" scripts/run_lidar_forked_action_data_gate.py \
  --source "$TEST_SOURCE" \
  --protocol docs/FORKED_ACTION_CAUSAL_DATA_GATE_PROTOCOL.md \
  --output-dir "$TEST_FORK" \
  --device cuda \
  --num-scenes 150 \
  --anchors-per-scene 4 \
  --num-workers 12 \
  --hold-steps 8 \
  --perturbation 0.45 \
  --resume

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
