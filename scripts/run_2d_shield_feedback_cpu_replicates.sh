#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python="${TWO_D_CPU_PYTHON:-$repo_root/../.venv-2d-cpu/bin/python}"
output="$repo_root/artifacts/dual_constraint_2d_shield_feedback_cpu_replicates_20260926"
seeds=(102 103 104 105 106)

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
    echo "The CPU replication launcher is already running" >&2
    exit 1
fi

cd "$repo_root"
"$python" - <<'PY'
import platform
import torch
if platform.machine().lower() not in ("aarch64", "arm64"):
    raise SystemExit("The replication panel requires ARM64")
if torch.version.cuda is not None or torch.cuda.is_available():
    raise SystemExit("The replication panel requires a CPU-only PyTorch wheel")
print("CPU-only replication seeds 102–106:", platform.machine(), torch.__version__)
PY

wait_batch() {
    local pid
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then failed=1; fi
    done
    pids=()
}

pids=()
failed=0
for seed in "${seeds[@]}"; do
    for condition in control feedback; do
        destination="$output/seed_$seed/$condition/training"
        mkdir -p "$destination"
        "$python" -u -m learning2d.train_shield_feedback_replicates_cpu \
            --seed "$seed" --condition "$condition" --output "$destination" \
            > "$destination/run.log" 2>&1 &
        pids+=("$!")
    done
done
wait_batch
if ((failed)); then
    echo "At least one replication training job failed; inspect per-job run.log" >&2
    exit 1
fi

failed=0
for seed in "${seeds[@]}"; do
    for condition in control feedback; do
        for map_id in 32 33 34 35 36 37 38 39; do
            for mode in shielded raw; do
                destination="$output/seed_$seed/$condition/validation/$mode/map_$(printf '%03d' "$map_id")"
                mkdir -p "$destination"
                "$python" -u -m learning2d.evaluate_shield_feedback_replicates_cpu \
                    --mode "$mode" --map-id "$map_id" \
                    --training-output "$output/seed_$seed/$condition/training" \
                    --output "$destination" > "$destination/run.log" 2>&1 &
                pids+=("$!")
                if ((${#pids[@]} == 16)); then wait_batch; fi
            done
        done
    done
done
if ((${#pids[@]})); then wait_batch; fi
if ((failed)); then
    echo "At least one replication validation job failed; inspect per-map run.log" >&2
    exit 1
fi

echo "Completed ten paired-seed CPU trainings and all 160 validation jobs"
