#!/bin/bash
# Explicit scheduler CPU+GPU resources are supplied by Root's admission.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
binding="$1"
bootstrap_python="$2"
child=""
trap 'if [[ -n "$child" ]]; then kill -USR1 "$child" 2>/dev/null || true; fi' USR1 TERM
"$bootstrap_python" -B -u -m experiments.root_native_campaign_20261023.bootstrap --binding "$binding" &
child=$!
set +e
while true; do
    wait "$child"
    code=$?
    if kill -0 "$child" 2>/dev/null; then continue; fi
    break
done
exit "$code"
