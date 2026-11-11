#!/bin/bash
set -euo pipefail
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TZ=America/New_York
unset PYTHONPATH PYTHONOPTIMIZE
cd /home/luyangz/data_selection_multimodal/native_campaign_v3_20261023/src
exec /data/user_data/luyangz/native_campaign_20261023/envs/native-fp16-xformers-v2/bin/python -B -s -m experiments.root_babel_s1_closure_phase3_v1_20261026.worker --configuration "$1" --review "$2" --registration "$3"
