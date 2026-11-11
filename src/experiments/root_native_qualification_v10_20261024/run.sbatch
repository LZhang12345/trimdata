#!/bin/bash
# Root supplies exact account/QoS/1GPU/CPU/RAM/time and durable binding.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
binding="$1"
python_executable="$2"
child=""
trap 'if [[ -n "$child" ]]; then kill -USR1 "$child" 2>/dev/null || true; fi' USR1 TERM
"$python_executable" -B -u -m experiments.root_native_qualification_v10_20261024.supervisor --binding "$binding" &
child=$!
set +e
while true; do
    wait "$child"
    code=$?
    if kill -0 "$child" 2>/dev/null; then continue; fi
    break
done
exit "$code"
