#!/bin/bash
# Explicit oneH100/eightCPU/64GiB/55min supplied by reviewed Root admission.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
binding="$1"
tiny_python="$2"
child=""
trap 'if [[ -n "$child" ]]; then kill -USR1 "$child" 2>/dev/null || true; fi' USR1 TERM
"$tiny_python" -B -u -m experiments.root_native_tiny_qualification_20261023.qualifier --binding "$binding" &
child=$!
set +e
while true; do
    wait "$child"
    code=$?
    if kill -0 "$child" 2>/dev/null; then continue; fi
    break
done
exit "$code"
