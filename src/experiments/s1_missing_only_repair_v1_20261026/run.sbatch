#!/bin/bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
unset PYTHONPATH
exec "$2" -B -s -u -m experiments.s1_missing_only_repair_v1_20261026.worker --binding "$1"
