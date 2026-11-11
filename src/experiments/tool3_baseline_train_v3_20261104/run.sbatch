#!/bin/bash
set -euo pipefail
export PATH=/usr/bin:/bin
unset BASH_ENV ENV LD_PRELOAD LD_LIBRARY_PATH PYTHONHOME PYTHONPATH PYTHONOPTIMIZE
exec /usr/bin/python3 -I -B /home/luyangz/data_selection_multimodal/experiments/tool3_baseline_train_v3_20261104/chain.py "$@"
