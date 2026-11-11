#!/bin/bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONFAULTHANDLER=1
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
unset PYTHONHOME PYTHONPATH LD_PRELOAD
python=/data/user_data/luyangz/native_campaign_20261023/envs/native-fp16-xformers-v2/bin/python
test "$2" = "$python"
child=
trap 'if [[ -n "$child" ]]; then kill -USR1 "$child" 2>/dev/null || true; fi' USR1 TERM
"$python" -B -u -m experiments.root_babel_operator_trace_v2_20261024.launch worker --binding "$1" &
child=$!
set +e
while true; do
 wait "$child"
 code=$?
 if ! kill -0 "$child" 2>/dev/null; then break; fi
done
exit "$code"
