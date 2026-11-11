#!/bin/bash
# Exact binding/CPU+GPU resources supplied by reviewed Root held admission.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
unset PYTHONHOME PYTHONPATH LD_PRELOAD
child=""
trap 'if [[ -n "$child" ]]; then kill -USR1 "$child" 2>/dev/null || true; fi' USR1 TERM
"$2" -B -u -m experiments.root_native_deterministic_replication_v3_20261024.worker --binding "$1" &
child=$!
set +e
while true; do
    wait "$child"; code=$?
    if kill -0 "$child" 2>/dev/null; then continue; fi
    break
done
exit "$code"
