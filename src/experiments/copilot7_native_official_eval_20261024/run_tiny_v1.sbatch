#!/bin/bash
set -euo pipefail
export PYTHONNOUSERSITE=1 PYTHONFAULTHANDLER=1
unset PYTHONHOME PYTHONPATH LD_PRELOAD
module load WebProxy CUDA/12.1.1
test "$CUDA_HOME" = /sw/eb/sw/CUDA/12.1.1
venv=/scratch/user/u.lz341613/dsmm_native_campaign_20261023/envs/tiny-py310-torch220-cu121-v1
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:$venv/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export TMPDIR="/tmp/job.${SLURM_JOB_ID}"
exec "$venv/bin/python" -B -u -m experiments.copilot7_native_official_eval_20261024.feature_v1 --binding "$1"
