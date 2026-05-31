export MODEL_NAME="/ltl/VideoX-Fun/models/Wan2.1-T2V-1.3B"
export DATASET_NAME="/ltl/dataset/OpenVidHD/generated_pairs"
export DATASET_META_NAME="/ltl/dataset/OpenVidHD/reward_scores_with_motion.csv"

accelerate launch --mixed_precision="bf16" train_dpo_ours.py \
  --config_path="/ltl/VideoX-Fun/config/wan2.1/wan_civitai_ours.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --video_sample_size=480 \
  --video_sample_stride=2 \
  --video_sample_n_frames=81 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=4 \
  --num_train_epochs=2 \
  --checkpointing_steps=5000 \
  --learning_rate=1e-05 \
  --seed=42 \
  --output_dir="output_dir_dpo_1.3b" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --uniform_sampling \
  --rank=64 \
  --network_alpha=32 \
  --target_name="q,k,v,ffn.0,ffn.2" \
  --use_peft_lora \
  --cpo_beta=500.0 \
  # --validation_prompts "a cat walking on the grass" "a beautiful sunset over the ocean" \
  # --validation_steps 5000
