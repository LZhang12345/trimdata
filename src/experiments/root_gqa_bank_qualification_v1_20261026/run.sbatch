#!/bin/bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH PYTHONOPTIMIZE
exec /data/user_data/luyangz/native_campaign_20261023/envs/native-fp16-xformers-v2/bin/python -B -s -u -m experiments.root_gqa_bank_qualification_v1_20261026.worker --binding "$1"
