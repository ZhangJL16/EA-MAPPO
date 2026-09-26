#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python="${TWO_D_CPU_PYTHON:-$repo_root/../.venv-2d-cpu/bin/python}"
output="$repo_root/artifacts/dual_constraint_2d_shield_feedback_arm_cpu_16workers_20260926"

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

mkdir -p "$output"
exec 9>"$output/launcher.lock"
if ! flock -n 9; then
    echo "The 16-worker paired launcher is already running" >&2
    exit 1
fi

cd "$repo_root"
"$python" - <<'PY'
import platform
import torch
if platform.machine().lower() not in ("aarch64", "arm64"):
    raise SystemExit("The frozen parallel run requires ARM64")
if torch.version.cuda is not None or torch.cuda.is_available():
    raise SystemExit("The parallel run requires a CPU-only PyTorch wheel")
print("ARM CPU 16 environment workers:", platform.machine(), torch.__version__)
PY

for condition in control feedback; do
    mkdir -p "$output/$condition/training"
    "$python" -u -m learning2d.train_shield_feedback_parallel_cpu \
        --condition "$condition" --output "$output/$condition/training" \
        > "$output/$condition/training/run.log" 2>&1 &
    if [[ "$condition" == control ]]; then
        control_pid=$!
    else
        feedback_pid=$!
    fi
done

training_failed=0
if ! wait "$control_pid"; then training_failed=1; fi
if ! wait "$feedback_pid"; then training_failed=1; fi
if ((training_failed)); then
    echo "Paired training failed; inspect condition-specific run.log files" >&2
    exit 1
fi

wait_batch() {
    local pid
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then evaluation_failed=1; fi
    done
    pids=()
}

pids=()
evaluation_failed=0
for condition in control feedback; do
    for map_id in 32 33 34 35 36 37 38 39; do
        for mode in shielded raw; do
            destination="$output/$condition/validation/$mode/map_$(printf '%03d' "$map_id")"
            mkdir -p "$destination"
            "$python" -u -m learning2d.evaluate_shield_feedback_parallel_cpu \
                --mode "$mode" --map-id "$map_id" \
                --training-output "$output/$condition/training" \
                --output "$destination" > "$destination/run.log" 2>&1 &
            pids+=("$!")
            if ((${#pids[@]} == 16)); then wait_batch; fi
        done
    done
done
if ((${#pids[@]})); then wait_batch; fi
if ((evaluation_failed)); then
    echo "At least one validation job failed; inspect per-map run.log files" >&2
    exit 1
fi

echo "Completed both 25,600-transition CPU conditions and all 32 validation jobs"
