#!/bin/bash
#SBATCH --account=cx
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --no-requeue
set -euo pipefail
export TZ=America/New_York
export HF_HOME=/data/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
# Same numerical/threading environment as the v7 evaluator that scored uniform-s17.
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
unset PYTHONHOME PYTHONPATH PYTHONOPTIMIZE LD_PRELOAD
cd /home/luyangz/data_selection_multimodal/claude_coincide_eval_src_v3_20261030
test "$#" -eq 1
exec /data/user_data/luyangz/native_campaign_20261023/envs/native-fp16-xformers-v2/bin/python -B -s -u -m claude_coincide_eval_v3_20261030.worker --request "$1"
