#!/usr/bin/env bash
set -eo pipefail
run=$(realpath "${1:?supply run directory}")
reference=$(realpath "${2:?supply completed reference run}")
fold=${3:-1}
repo=$(cd "$run/../.." && pwd)
out="$run/output/fold_$fold"
mkdir -p "$out"
exec >> "$out/driver.log" 2>&1
trap 'rc=$?; echo "$rc" > "$out/pipeline.exit"; date -u' EXIT
source "$repo/setup.sh"
set -u
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLBACKEND=Agg
echo "Starting fold $fold on $(hostname) at $(date -u)"
while [[ ! -f "$run/dataset/audit.json" ]]; do
    if [[ -f "$run/publication.exit" ]]; then
        echo "Dataset publication ended without an audited dataset"
        exit 1
    fi
    sleep 10
done
python -B "$run/diagnostics/track.py" --run "$run" --reference "$reference" --fold "$fold" &
tracker=$!
if [[ ! -f "$out/exit_status.txt" ]] || [[ $(cat "$out/exit_status.txt") != 0 ]]; then
    bash "$run/worker.sh" "$fold"
fi
python -B -u "$run/diagnostics/analyze.py" --run "$run" --reference "$reference" --fold "$fold"
echo "Training, application, and comparison completed at $(date -u)"
