#!/usr/bin/env bash
set -eo pipefail
if [[ ${CONDA_DEFAULT_ENV:-} != ML ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate ML
fi
set -u
config_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$config_dir/../../src${PYTHONPATH:+:$PYTHONPATH}"
cd "$config_dir/../.."
exec python -m synapse.main \
    -d "$config_dir/data_config.yaml" -m "$config_dir/model_config.yaml" -r "$config_dir/run_fold_0.yaml" "$@"
