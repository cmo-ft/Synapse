#!/usr/bin/env bash
set -eo pipefail
run=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$run/../.." && pwd)
fold=${1:?supply 0 or 1}
out="$run/output/fold_$fold"
mkdir -p "$out"
exec >> "$out/workflow.log" 2>&1
trap 'rc=$?; date -u; echo "exit $rc"; echo "$rc" > "$out/exit_status.txt"' EXIT
source "$repo/setup.sh"
set -u
export PYTHONPATH="$run/frozen:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLBACKEND=Agg
echo "Fold $fold on $(hostname) at $(date -u)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
launch="$run/frozen/recipe/launch.py"
python -B -u "$launch" train --fold "$fold" --run "$run"
shopt -s nullglob
best=("$out"/fit/checkpoints/BEST-*.ckpt)
(( ${#best[@]} == 1 ))
echo "Applying ${best[0]} at $(date -u)"
for split in validation test; do
    python -B -u "$launch" "$split" --fold "$fold" --run "$run" --checkpoint "${best[0]}"
done
val="$out/validation/validation_scores.root"
test="$out/test/test_scores.root"
(
    set +u
    source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
    python3 -B -u "$repo/configs/run3_common/run_hhml_benchmark.py" --fold "$fold" --validation "$val" --test "$test" --out "$out/evaluation/hhml_eval"
)
for split in validation test; do
    native=held_out
    if [[ $split == validation ]]; then native=validation; fi
    scores="$out/$split/${split}_scores.root"
    PYTHONPATH="$repo/hhml_eval:$PYTHONPATH" python -B -u "$run/frozen/plots/plot_background_rejection.py" --scores "$scores" --catalog "$run/dataset/summary.json" --benchmark "$out/evaluation/hhml_eval/$native" --out "$out/evaluation/$split/background_rejection"
    python -B -u "$run/frozen/recipe/evaluate_aux.py" "$scores" "$out/evaluation/$split/auxiliary.json"
    python -B -u "$run/frozen/recipe/working_point_composition.py" --run "$run" --fold "$fold" --split "$split"
done
