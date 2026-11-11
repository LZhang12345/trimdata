#!/bin/bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TMPDIR="/tmp/job.${SLURM_JOB_ID}"
exec /scratch/user/u.lz341613/dsmm_native_campaign_20261023/envs/tiny-py310-torch220-cu121-v1/bin/python -B -u -m experiments.copilot7_native_official_eval_20261024.collect_tiny_v3 --binding "$1"
