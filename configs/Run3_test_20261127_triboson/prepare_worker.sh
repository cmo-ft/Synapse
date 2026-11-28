#!/usr/bin/env bash
set -eo pipefail
run=$(realpath "${1:?supply run path}")
reference=$(realpath "${2:?supply source run}")
repo=$(realpath "$run/../..")
exec >> "$run/publication.log" 2>&1
trap 'rc=$?; echo "$rc" > "$run/publication.exit.tmp"; mv "$run/publication.exit.tmp" "$run/publication.exit"; date -u' EXIT
source "$repo/setup.sh"
set -u
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
echo "Preparing on $(hostname) at $(date -u)"
python -B -u "$run/frozen/recipe/prepare_dataset.py" --run "$run" --reference "$reference"
test -s "$run/dataset/audit.json"
python -B -u "$run/diagnostics/check_training.py" "$run"
echo "Dataset and training entry points verified at $(date -u)"
