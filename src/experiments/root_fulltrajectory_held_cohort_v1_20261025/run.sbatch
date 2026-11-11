#!/bin/bash
set -euo pipefail
test "$#" -eq 2
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONHOME PYTHONPATH LD_PRELOAD
exec "$2" -B -m experiments.root_fulltrajectory_held_cohort_v1_20261025.worker --binding "$1"
