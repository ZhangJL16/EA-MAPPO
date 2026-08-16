#!/usr/bin/env bash
cd /home/zjl/mappo
exec env PYTHONUNBUFFERED=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/train_uav_energy_delivery_sac.py \
  --output-dir "artifacts/uav_energy_delivery_v3_energy_formal_20260816_175614" \
  --device cuda \
  --seed 0 \
  --num-envs 8 \
  --gradient-steps -1 \
  --phase1-transition-budget 500000 \
  --phase1b-transition-budget 500000 \
  --phase2-transition-budget 500000 \
  --phase1-episode-max-steps 4000 \
  --phase2-episode-max-steps 20000 \
  --eval-freq-transitions 50000 \
  --checkpoint-freq-transitions 50000 \
  --eval-navigation-tasks 500 \
  --battery-calibration-tasks 500 \
  --battery-validation-runs 100 \
  --target-nominal-endurance-minutes 30 \
  --energy-reserve-fraction 0.10 \
  --run-td-pretraining \
  --run-phase2 \
  --resume-after-phase1-checkpoint artifacts/uav_energy_delivery_v3_formal_20260816_004619/phase1_navigation/checkpoint_transition_500000.zip \
  --source-phase1-artifact artifacts/uav_energy_delivery_v3_formal_20260816_004619
