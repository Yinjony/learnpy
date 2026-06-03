#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIDEOX_FUN_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export WAN_MODEL_DIR="${WAN_MODEL_DIR:-$VIDEOX_FUN_ROOT/models/Wan2.1-T2V-1.3B}"
export DATASET_NAME="${DATASET_NAME:-$VIDEOX_FUN_ROOT/dataset/Wan21/Action2V/data}"
export OUTPUT_DIR="${OUTPUT_DIR:-$VIDEOX_FUN_ROOT/output_dir_world_model_stage0}"
export TRAINING_STRATEGY="${TRAINING_STRATEGY:-sft}"
export GRPO_GROUP_SIZE="${GRPO_GROUP_SIZE:-4}"
export GRPO_SFT_COEF="${GRPO_SFT_COEF:-0.1}"

cd "$SCRIPT_DIR"

python "$SCRIPT_DIR/check_world_model_env.py" \
  --model-dir="$WAN_MODEL_DIR" \
  --data-path="$DATASET_NAME"

accelerate launch --mixed_precision="bf16" "train_world_model_ours.py" \
  --config_path="$VIDEOX_FUN_ROOT/config/wan2.1/bidirectional_camera.yaml" \
  --pretrained_model_name_or_path="$WAN_MODEL_DIR" \
  --train_data_dir="$DATASET_NAME" \
  --training_strategy="$TRAINING_STRATEGY" \
  --grpo_group_size="$GRPO_GROUP_SIZE" \
  --grpo_sft_coef="$GRPO_SFT_COEF" \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=4 \
  --num_train_epochs=2 \
  --checkpointing_steps=5000 \
  --learning_rate=2e-6 \
  --seed=42 \
  --output_dir="$OUTPUT_DIR" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_beta1=0.0 \
  --adam_beta2=0.999 \
  --adam_weight_decay=1e-2 \
  --adam_epsilon=1e-8 \
  --max_grad_norm=10.0
