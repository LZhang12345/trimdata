#!/bin/bash
set -euo pipefail
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONHOME PYTHONPATH LD_PRELOAD
module load CUDA/12.1.1
test "$CUDA_HOME" = /sw/eb/sw/CUDA/12.1.1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 CUBLAS_WORKSPACE_CONFIG=:4096:8
exec "$2" -B -u -m experiments.session6_start_model_qualification_v4_eager_20261024.worker --binding "$1"
