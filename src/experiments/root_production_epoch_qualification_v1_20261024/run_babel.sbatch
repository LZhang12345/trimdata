#!/bin/bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
unset PYTHONHOME PYTHONPATH LD_PRELOAD
child=
trap 'if [[ -n "$child" ]]; then kill -USR1 "$child" 2>/dev/null || true; fi' USR1 TERM
"$2" -B -u -m experiments.root_production_epoch_qualification_v1_20261024.worker --binding "$1" &
child=$!
set +e
while true; do wait "$child"; code=$?; if ! kill -0 "$child" 2>/dev/null; then break; fi; done
exit "$code"
