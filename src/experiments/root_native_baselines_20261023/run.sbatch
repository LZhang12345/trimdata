#!/bin/bash
# Root supplies explicit CPU+GPU, account/QoS, finite wall time and ownership nonce.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
binding="$1"
provider_python="$2"
phase="${3:-features}"
case "$phase" in
    features) provider_module="provider" ;;
    postprocess) provider_module="postprocess" ;;
    *) exit 64 ;;
esac
child=""
trap 'if [[ -n "$child" ]]; then kill -USR1 "$child" 2>/dev/null || true; fi' USR1 TERM
"$provider_python" -B -u -m "experiments.root_native_baselines_20261023.$provider_module" --binding "$binding" &
child=$!
set +e
while true; do
    wait "$child"
    code=$?
    if kill -0 "$child" 2>/dev/null; then continue; fi
    break
done
exit "$code"
