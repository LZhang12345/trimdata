#!/bin/bash
# Explicit scheduler CPU+GPU resources are supplied by Root's admission.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
unset PYTHONHOME PYTHONPATH LD_PRELOAD
binding="$1"
bootstrap_python="$2"
# This route is registered in the immutable held admission. The worker still
# authenticates the actual site/account/QoS/resources before any training call.
if [[ "${DSMM_NATIVE_SITE:-}" == aces ]]; then
    module load CUDA/12.1.1
    test "${CUDA_HOME:-}" = /sw/eb/sw/CUDA/12.1.1
    test -f "$CUDA_HOME/lib64/libcusparse.so.12"
    native_prefix=$(dirname "$(dirname "$bootstrap_python")")
    torch_libraries=("$native_prefix"/lib/python*/site-packages/torch/lib)
    test "${#torch_libraries[@]}" -eq 1
    test -d "${torch_libraries[0]}"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${torch_libraries[0]}:${LD_LIBRARY_PATH:-}"
fi
child=""
trap 'if [[ -n "$child" ]]; then kill -USR1 "$child" 2>/dev/null || true; fi' USR1 TERM
"$bootstrap_python" -B -u -m experiments.root_native_campaign_v5_20261024.bootstrap --binding "$binding" &
child=$!
set +e
while true; do
    wait "$child"
    code=$?
    if kill -0 "$child" 2>/dev/null; then continue; fi
    break
done
exit "$code"
