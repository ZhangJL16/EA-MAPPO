#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "usage: $0 unguided|recovery_guided SEED OUTPUT_DIR PHYSICAL_GPU_INDEX [NAVIGATION_TEACHER_MODEL]" >&2
  exit 2
fi
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
mode="$1"; seed="$2"; output_dir="$3"; gpu="$4"; navigation_teacher="${5:-}"
python_bin="${PYTHON_BIN:-$root/.venv/bin/python}"
certificate="$root/certificates/random_persistent_open_energy_x2.json"
[[ -f "$certificate" ]] || { echo "missing 2x certificate: $certificate" >&2; exit 3; }
[[ ! -e "$output_dir" ]] || { echo "refusing to overwrite $output_dir" >&2; exit 4; }
mkdir -p "$output_dir"
command=(env CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$python_bin" -u scripts/train_sb3_energy_sac_2x.py
  --guidance-mode "$mode" --scenario random_persistent_open.json --seed "$seed" --steps 1000000
  --flight-energy-multiplier 2.0 --certificate-artifact "$certificate" --device cuda:0
  --output-dir "$output_dir")
if [[ "$mode" == "recovery_guided" ]]; then
  [[ -n "$navigation_teacher" && -f "$navigation_teacher" ]] || {
    echo "recovery_guided requires an existing navigation teacher checkpoint" >&2
    exit 5
  }
  command+=(--navigation-teacher-model "$navigation_teacher")
elif [[ "$mode" != "unguided" ]]; then
  echo "unknown guidance mode: $mode" >&2
  exit 6
fi
printf '%q ' "${command[@]}" > "$output_dir/command.txt"; printf '\n' >> "$output_dir/command.txt"
"${command[@]}" >> "$output_dir/train.log" 2>&1 &
pid=$!
printf '%s\n' "$pid" > "$output_dir/pid.txt"
set +e
wait "$pid"
status=$?
set -e
printf '%s\n' "$status" > "$output_dir/exit_code.txt"
exit "$status"
