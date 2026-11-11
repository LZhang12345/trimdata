#!/bin/bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
unset PYTHONHOME PYTHONPATH LD_PRELOAD
module load CUDA/12.1.1
: "${CUDA_HOME:?CUDA module required}"
test "$CUDA_HOME" = /sw/eb/sw/CUDA/12.1.1
root=/scratch/user/u.lz341613/dsmm_native_campaign_20261023
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$root/envs/native-py310-cu121-v1/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
child=
trap 'if [[ -n "$child" ]]; then kill -USR1 "$child" 2>/dev/null || true; fi' USR1 TERM
"$2" -B -u -m experiments.root_production_epoch_qualification_v1_20261024.worker --binding "$1" &
child=$!
set +e
while true; do wait "$child"; code=$?; if ! kill -0 "$child" 2>/dev/null; then break; fi; done
exit "$code"
