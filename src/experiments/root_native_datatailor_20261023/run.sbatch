#!/bin/bash
# Exact explicit CPU+GPU request, nonce and ownership are supplied by Root.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
binding=$1
native_python=$2
"$native_python" -B -u -m experiments.root_native_datatailor_20261023.provider --binding "$binding" &
child=$!
trap 'kill -USR1 "$child" 2>/dev/null || true' USR1 TERM
set +e
while true; do
  wait "$child"
  code=$?
  if ! kill -0 "$child" 2>/dev/null; then break; fi
done
exit "$code"
