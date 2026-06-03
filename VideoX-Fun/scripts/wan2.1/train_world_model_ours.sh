#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIDEOX_FUN_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export WAN_MODEL_DIR="${WAN_MODEL_DIR:-$VIDEOX_FUN_ROOT/models/Wan2.1-T2V-1.3B}"
export DATASET_NAME="${DATASET_NAME:-$VIDEOX_FUN_ROOT/datasets/Wan21/Action2V/data}"

# Keep optimizer/model defaults in bidirectional_camera.yaml. The knobs below
# are intentionally small quick-run overrides for smoke-testing the pipeline.
export TRAINING_STRATEGY="${TRAINING_STRATEGY:-grpo}"
export OUTPUT_DIR="${OUTPUT_DIR:-$VIDEOX_FUN_ROOT/output_dir_world_model_quick_${TRAINING_STRATEGY}}"
export MIXED_PRECISION="${MIXED_PRECISION:-fp16}"
export TEXT_ENCODER_PRECISION="${TEXT_ENCODER_PRECISION:-fp32}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
export MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-8}"
export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-20}"
export CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-20}"
export GRPO_GROUP_SIZE="${GRPO_GROUP_SIZE:-2}"
export GRPO_REWARD_TYPE="${GRPO_REWARD_TYPE:-reconstruction}"
export GRPO_SFT_COEF="${GRPO_SFT_COEF:-0.1}"
export GRPO_ADVANTAGE_CLIP="${GRPO_ADVANTAGE_CLIP:-5.0}"
export MAX_GRAD_NORM="${MAX_GRAD_NORM:-10.0}"

cd "$SCRIPT_DIR"

python "$SCRIPT_DIR/check_world_model_env.py" \
  --model-dir="$WAN_MODEL_DIR" \
  --data-path="$DATASET_NAME"

accelerate launch --mixed_precision="$MIXED_PRECISION" "train_world_model_ours.py" \
  --config_path="$VIDEOX_FUN_ROOT/config/wan2.1/bidirectional_camera.yaml" \
  --pretrained_model_name_or_path="$WAN_MODEL_DIR" \
  --train_data_dir="$DATASET_NAME" \
  --training_strategy="$TRAINING_STRATEGY" \
  --grpo_group_size="$GRPO_GROUP_SIZE" \
  --grpo_reward_type="$GRPO_REWARD_TYPE" \
  --grpo_sft_coef="$GRPO_SFT_COEF" \
  --grpo_advantage_clip="$GRPO_ADVANTAGE_CLIP" \
  --train_batch_size="$TRAIN_BATCH_SIZE" \
  --gradient_accumulation_steps="$GRADIENT_ACCUMULATION_STEPS" \
  --dataloader_num_workers="$DATALOADER_NUM_WORKERS" \
  --max_train_samples="$MAX_TRAIN_SAMPLES" \
  --max_train_steps="$MAX_TRAIN_STEPS" \
  --checkpointing_steps="$CHECKPOINTING_STEPS" \
  --output_dir="$OUTPUT_DIR" \
  --mixed_precision="$MIXED_PRECISION" \
  --text_encoder_precision="$TEXT_ENCODER_PRECISION" \
  --max_grad_norm="$MAX_GRAD_NORM"
