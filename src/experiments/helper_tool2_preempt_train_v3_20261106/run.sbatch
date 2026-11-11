#!/bin/bash
set -euo pipefail
export PATH=/usr/bin:/bin
unset BASH_ENV ENV LD_PRELOAD LD_LIBRARY_PATH PYTHONHOME PYTHONPATH PYTHONOPTIMIZE
src=/home/luyangz/data_selection_multimodal/experiments/helper_tool2_preempt_train_v3_20261106/chain.py
echo "f02453ba58a4fe15b9facbd8caa33094e8d0f8d1e23c62128145f428dae02bf2  $src" | /usr/bin/sha256sum -c --quiet - || { echo "Pinned chain.py changed" >&2; exit 2; }
exec /usr/bin/python3 -I -B "$src" "$@"
