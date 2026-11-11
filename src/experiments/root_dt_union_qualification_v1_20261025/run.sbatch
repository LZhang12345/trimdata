#!/bin/bash
set -euo pipefail
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export HF_HOME=/data/hf_cache HF_HUB_CACHE=/data/hf_cache/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset PYTHONPATH PYTHONHOME LD_PRELOAD
cd /home/luyangz/data_selection_multimodal/root_dt_union_qualification_src_v1
exec /data/user_data/luyangz/native_campaign_20261023/envs/native-fp16-xformers-v2/bin/python -B -s -u -m experiments.root_dt_union_qualification_v1_20261025.worker --binding "$1"
