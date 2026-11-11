#!/bin/bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONFAULTHANDLER=1
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
unset PYTHONHOME PYTHONPATH LD_PRELOAD
module load CUDA/12.1.1
test "${CUDA_HOME:?}" = /sw/eb/sw/CUDA/12.1.1
root=/scratch/user/u.lz341613/dsmm_native_campaign_20261023
python="$root/envs/native-py310-cu121-v1/bin/python"
test "$2" = "$python"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$root/envs/native-py310-cu121-v1/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
child=
trap 'if [[ -n "$child" ]]; then kill -USR1 "$child" 2>/dev/null || true; fi' USR1 TERM
"$python" -B -u -m experiments.root_native_replay_dispersion_v1_20261024.worker --binding "$1" &
child=$!
set +e
while true; do
  wait "$child"
  code=$?
  if ! kill -0 "$child" 2>/dev/null; then break; fi
done
exit "$code"
