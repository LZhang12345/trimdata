#!/bin/bash
#SBATCH --account=cx
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --signal=B:USR1@600
#SBATCH --requeue
set -euo pipefail
export TZ=America/New_York
export HF_HOME=/data/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8
cd /home/luyangz/data_selection_multimodal/coincide_native_exec_src_v2_20261028
test "$#" -eq 2
exec /data/user_data/luyangz/native_campaign_20261023/envs/native-fp16-xformers-v2/bin/python -B -m experiments.copilot7_coincide_native_exec_v2_20261028.driver --binding "$1" --activation "$2"
