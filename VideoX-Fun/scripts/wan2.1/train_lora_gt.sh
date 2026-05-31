export MODEL_NAME="/root/data1/yizeli/VideoX-Fun/models/Wan2.1-T2V-1.3B"
export DATASET_NAME="/root/data1/yizeli/OpenVidHD/videos"
export DATASET_META_NAME="/root/data1/yizeli/OpenVidHD/reward_scores_with_motion_GT_compat.csv"

accelerate launch --mixed_precision="bf16" train_lora_gt.py \
  --config_path="/root/data1/yizeli/VideoX-Fun/config/wan2.1/wan_civitai_ours.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=480 \
  --video_sample_size=480 \
  --token_sample_size=480 \
  --video_sample_stride=2 \
  --video_sample_n_frames=81 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=4 \
  --num_train_epochs=1 \
  --checkpointing_steps=5000 \
  --learning_rate=1e-04 \
  --seed=42 \
  --output_dir="output_dir_lora_gt" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --random_hw_adapt \
  --training_with_video_token_length \
  --enable_bucket \
  --uniform_sampling \
  --rank=64 \
  --network_alpha=32 \
  --target_name="q,k,v,ffn.0,ffn.2" \
  --use_peft_lora \
  --validation_prompts "a cat walking on the grass" "a beautiful sunset over the ocean" \
  --validation_steps 5000 \
  #--low_vram
 # --max_train_steps 50