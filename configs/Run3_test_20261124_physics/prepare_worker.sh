#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "$0")"
source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
set -u
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
exec python3 -B -u prepare_physics.py --group "$1" --groups "$2"
