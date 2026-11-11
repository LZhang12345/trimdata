#!/bin/bash
set -euo pipefail
test "$#" -eq 1
unset PYTHONPATH PYTHONOPTIMIZE
export TZ=America/New_York PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
exec /data/user_data/luyangz/native_campaign_20261023/envs/native-fp16-xformers-v2/bin/python -B -s -u -m experiments.root_current_four_bank_eval_v1_20261027.worker --binding "$1"
