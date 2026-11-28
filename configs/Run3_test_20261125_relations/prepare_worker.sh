#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "$0")"
source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec python3 -B -u prepare_relations.py "$@"
