#!/bin/bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
unset PYTHONPATH
exec /usr/bin/python3 -B -s -u -m experiments.s1_parallel_prerequisites_v2_20261026.worker --binding "$1"
