#!/bin/bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONFAULTHANDLER=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
unset PYTHONHOME PYTHONPATH LD_PRELOAD
module load CUDA/12.1.1
: "${CUDA_HOME:?CUDA module did not supply CUDA_HOME}"
test "$CUDA_HOME" = /sw/eb/sw/CUDA/12.1.1
test -f "$CUDA_HOME/lib64/libcusparse.so.12"
root=/scratch/user/u.lz341613/dsmm_native_campaign_20261023
python="$root/envs/native-py310-cu121-v1/bin/python"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$root/envs/native-py310-cu121-v1/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
test "$2" = "$python"
child=
trap 'if [[ -n "$child" ]]; then kill -USR1 "$child" 2>/dev/null || true; fi' USR1 TERM
"$python" -B -u -m experiments.root_h100_native_qualification_v10_20261024.supervisor --binding "$1" &
child=$!
set +e
while true; do
    wait "$child"
    code=$?
    if ! kill -0 "$child" 2>/dev/null; then break; fi
done
exit "$code"
