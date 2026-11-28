#!/usr/bin/env bash
set -eo pipefail
cfg=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$cfg/../.." && pwd)
root="$repo/run_dir/${cfg##*/}"
point=${1:?supply a training-statistics sample name}
case "$point" in quarter|half|double|double_50epochs) ;; *) exit 2 ;; esac
out="$root/$point"
mkdir -p "$out"
exec >> "$out/workflow.log" 2>&1
trap 'code=$?; date -u; echo "exit $code"; echo "$code" > "$out/exit_status.txt"' EXIT
source "$repo/setup.sh"
set -u
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2 MPLBACKEND=Agg
echo "$point on $(hostname) at $(date -u)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
python -B -u "$cfg/train.py" "$point" train
if [[ ! -s "$out/validation/validation_scores.root" ]]; then
    python -B -u "$cfg/train.py" "$point" validation
fi
python -B -u "$cfg/compare.py" "$point"
if [[ "$point" == double || "$point" == double_50epochs ]]; then
    if [[ -s "$root/double/validation/validation_scores.root" && -s "$root/double_50epochs/validation/validation_scores.root" ]]; then
        python -B -u "$cfg/compare.py" double_50epochs --reference double
    fi
fi
