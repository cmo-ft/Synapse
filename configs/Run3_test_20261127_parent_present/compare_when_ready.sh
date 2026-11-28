#!/usr/bin/env bash
set -eo pipefail
main=$(realpath "${1:?supply terminal run}")
extra=$(realpath "${2:?supply batch run}")
prior=$(realpath "${3:?supply completed reference}")
repo=$(realpath "$main/../..")
exec >> "$main/comparisons.log" 2>&1
trap 'rc=$?; echo "$rc" > "$main/comparisons.exit"; date -u' EXIT
source "$repo/setup.sh"
set -u
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1
plot="$main/diagnostics/compare_rejection.py"
compare="$main/diagnostics/analyze.py"
compare_once() {
    local run=$1 reference=$2 label=$3 prev=$4
    local dest="$run/output/fold_1/evaluation/comparison_$(basename "$reference")"
    if [[ ! -s "$dest/comparison.md" ]]; then
        nice -n 10 python -B -u "$compare" --run "$run" --reference "$reference" --fold 1
    fi
    if [[ ! -s "$dest/test/rejection_by_background.pdf" ]]; then
        nice -n 10 python -B -u "$plot" --run "$run" --reference "$reference" --fold 1 --labels "$label" "$prev"
    fi
}
check_run() {
    local run=$1
    local status="$run/output/fold_1/pipeline.exit"
    [[ -s "$status" ]] || return 1
    [[ $(cat "$status") == 0 ]] || { echo "Failed pipeline $run" >&2; return 2; }
}
main_done=0
extra_done=0
bad=0
until (( main_done && extra_done )); do
    for key in main extra; do
        if [[ $key == main ]]; then
            (( main_done == 0 )) || continue
            run="$main"
            label="Parent-conditioned"
        else
            (( extra_done == 0 )) || continue
            run="$extra"
            label="HH visible masses"
        fi
        if check_run "$run"; then
            compare_once "$run" "$prior" "$label" "Previous NN"
        else
            code=$?
            if (( code == 1 )); then continue; fi
            bad=1
        fi
        if [[ $key == main ]]; then main_done=1; else extra_done=1; fi
    done
    if (( !main_done || !extra_done )); then sleep 60; fi
done
if ((bad)); then exit 1; fi
compare_once "$extra" "$main" "HH visible masses" "Parent-conditioned"
echo "All comparisons ready at $(date -u)"
