#!/bin/bash
set -euo pipefail
unset PYTHONPATH PYTHONOPTIMIZE
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
exec /data/user_data/luyangz/native_campaign_20261023/envs/native-fp16-xformers-v2/bin/python -B -s -u -m experiments.root_dt_formal_selection_v1_20261026.worker --binding "$1"
