#!/bin/bash
set -euo pipefail
test "$#" -eq 2
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONHOME PYTHONPATH LD_PRELOAD
exec "$2" -B -m experiments.root_production_student_launcher_v4_20261025.worker --binding "$1"
