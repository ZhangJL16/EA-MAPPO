#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python="${TWO_D_CPU_PYTHON:-$repo_root/../.venv-2d-cpu/bin/python}"
output="$repo_root/artifacts/dual_constraint_2d_shield_feedback_arm_cpu_20260926"

if [[ ! -x "$python" ]]; then
    echo "Missing isolated ARM CPU Python: $python" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES=-1
export HIP_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$repo_root"
"$python" - <<'PY'
import platform
import torch
import gymnasium
import stable_baselines3
if platform.machine().lower() not in ("aarch64", "arm64"):
    raise SystemExit("The frozen 2D run requires ARM64")
if torch.version.cuda is not None or torch.cuda.is_available():
    raise SystemExit("The isolated 2D environment must use a CPU-only PyTorch wheel")
print("ARM CPU runtime:", platform.machine(), torch.__version__,
      stable_baselines3.__version__, gymnasium.__version__)
PY

for condition in control feedback; do
    "$python" -u -m learning2d.train_shield_feedback_cpu \
        --condition "$condition" --output "$output/$condition/training"
done

for condition in control feedback; do
    for map_id in 32 33 34 35 36 37 38 39; do
        for mode in shielded raw; do
            "$python" -u -m learning2d.evaluate_shield_feedback_cpu \
                --mode "$mode" --map-id "$map_id" \
                --training-output "$output/$condition/training" \
                --output "$output/$condition/validation/$mode/map_$(printf '%03d' "$map_id")"
        done
    done
done
