#!/bin/bash
set -euo pipefail
export PATH=/usr/bin:/bin
unset BASH_ENV ENV LD_PRELOAD LD_LIBRARY_PATH PYTHONHOME PYTHONPATH PYTHONOPTIMIZE
src=/home/luyangz/data_selection_multimodal/experiments/helper_tool2_preempt_train_v1d_20261105/chain.py
echo "918169241795bca88dc2c67978879c7205a5cd04445e25243f46d0d61a679a21  $src" | /usr/bin/sha256sum -c --quiet - || { echo "Pinned chain.py changed" >&2; exit 2; }
exec /usr/bin/python3 -I -B "$src" "$@"
