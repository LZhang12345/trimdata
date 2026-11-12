#!/usr/bin/env bash
# READ-ONLY REFERENCE. Do NOT use this for new runs: use scripts/train_portable.py (see AGENT_CONTRACT.md section 3).
# It lists the executed training arguments for the official LLaVA entry point, but plain train.py / train_mem.py
# lack the three runtime changes our runs used (xformers CUTLASS attention with deterministic settings, structured
# v1 label masking, pre-scheduler removal of the empty optimizer group). Without the last one the projector weights
# train at the LoRA learning rate (2e-4) instead of 2e-5. train_portable.py applies all three.
# It is single-GPU only (per-device 4 x accumulation 32 = global batch 128); launching it on 8 GPUs gives global
# batch 1024 and a wrong step count.
# Usage (reference only): bash reference_train_command.sh <records.json> <image_root> <output_dir> <training_seed>
set -euo pipefail
DATA=$1; IMAGES=$2; OUT=$3; SEED=$4
deepspeed --num_gpus 1 llava/train/train_mem.py \
  --deepspeed ./scripts/zero3.json \
  --model_name_or_path lmsys/vicuna-7b-v1.5 --version v1 \
  --data_path "$DATA" --image_folder "$IMAGES" \
  --vision_tower openai/clip-vit-large-patch14-336 \
  --pretrain_mm_mlp_adapter ./checkpoints/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5/mm_projector.bin \
  --mm_projector_type mlp2x_gelu --mm_vision_select_layer -2 \
  --mm_use_im_start_end False --mm_use_im_patch_token False --image_aspect_ratio pad \
  --group_by_modality_length True --lora_enable True --lora_r 128 --lora_alpha 256 --lora_dropout 0.05 --lora_bias none \
  --mm_projector_lr 2e-5 --fp16 True --bf16 False --tf32 False --num_train_epochs 1 \
  --per_device_train_batch_size 4 --per_device_eval_batch_size 1 --gradient_accumulation_steps 32 \
  --evaluation_strategy no --save_strategy steps --save_steps 50000 --save_safetensors True \
  --learning_rate 2e-4 --weight_decay 0 --warmup_ratio 0.03 --lr_scheduler_type cosine \
  --model_max_length 2048 --gradient_checkpointing True --dataloader_num_workers 0 --lazy_preprocess True \
  --seed "$SEED" --data_seed "$SEED" --optim adamw_torch --report_to none --logging_steps 10 \
  --output_dir "$OUT"
# Note: train_mem.py enables FlashAttention-2, which our runs did NOT use. See scripts/train_portable.py.
