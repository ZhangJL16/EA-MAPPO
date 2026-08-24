#!/usr/bin/env bash
set -euo pipefail

cd /home/zjl/mappo

if [[ -n "$(git status --porcelain)" ]]; then
  echo "formal JSEB launch requires a clean worktree" >&2
  exit 2
fi

STAMP="$(date -u +%Y%m%d_%H%M%S)"
OUT="artifacts/jacobian_safety_energy_static_1m_${STAMP}"
SESSION="jseb_1m_${STAMP}"
PYTHON="/home/zjl/mappo/.venv/bin/python"

mkdir -p "$OUT"

CMD=(
  "$PYTHON" -u scripts/run_jacobian_safety_energy_1m.py
  --output-dir "$OUT"
  --phase1-transitions 500000
  --phase2a-transitions 100000
  --phase2b-transitions 300000
  --phase2c-transitions 100000
  --device cuda
  --seed 0
  --ablation D
  --num-envs 8
  --lidar-enabled
  --lidar-horizontal-sectors 128
  --lidar-vertical-sectors 8
  --lidar-range 100
  --num-obstacles 24
  --obstacle-radius-min 50
  --obstacle-radius-max 120
  --hocbf-enabled
  --hocbf-top-k 16
  --projection-geometry-enabled
)

printf '%q ' "${CMD[@]}" > "$OUT/exact_command.txt"
printf '\n' >> "$OUT/exact_command.txt"
git rev-parse HEAD > "$OUT/git_sha.txt"
git status --porcelain > "$OUT/git_status.txt"
printf '%s\n' "$SESSION" > "$OUT/tmux_session.txt"

tmux new-session -d -s "$SESSION" \
  "cd /home/zjl/mappo && PYTHONPATH=. ${CMD[*]} 2>&1 | tee '$OUT/run.log'"

printf 'session=%s\noutput=%s\n' "$SESSION" "$OUT"
