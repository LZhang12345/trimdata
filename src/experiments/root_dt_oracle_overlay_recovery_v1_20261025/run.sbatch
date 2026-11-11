#!/bin/bash
set -euo pipefail
unset PYTHONPATH PYTHONHOME LD_PRELOAD
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME=/data/hf_cache HF_HUB_CACHE=/data/hf_cache/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
exec /data/user_data/luyangz/native_campaign_20261023/envs/native-fp16-xformers-v2/bin/python -B -u -m experiments.root_dt_oracle_overlay_recovery_v1_20261025.worker --binding "$1"
