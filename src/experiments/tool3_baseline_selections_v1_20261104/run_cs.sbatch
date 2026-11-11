#!/bin/bash
set -euo pipefail
export PATH=/usr/bin:/bin
exec /data/user_data/luyangz/native_campaign_20261023/envs/native-fp16-xformers-v2/bin/python -I -B /home/luyangz/data_selection_multimodal/experiments/tool3_baseline_selections_v1_20261104/coincide_and_scores.py /home/luyangz/data_selection_multimodal/collaboration/helper-tool3-baselines-20261104/COINCIDE_AND_SCORES-$SLURM_JOB_ID.json
