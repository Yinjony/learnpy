# Wan2.1 CPO World-Model Training

This extension changes the two CPO stages from whole-video denoising into
camera-conditioned future prediction. It follows minWM's PRoPE idea while
keeping VideoX-Fun's existing CPO score expert and preference loss.

## Training behavior

- Each sample is split in latent time into a clean history block and a noisy
  future block.
- Only the future block contributes to the Stage1 flow-matching loss and the
  Stage2 CPO distances.
- Every latent frame carries a first-frame-relative world-to-camera matrix and
  an intrinsic matrix.
- Wan self-attention keeps its original RoPE path and adds a parallel PRoPE
  path through a zero-initialized `prope_o` residual projection.
- Stage1 saves `quality_*`, `motion_*`, and `prope_o` parameters together in
  `score_modules-<step>.safetensors`.
- Stage2 loads the Stage1 PRoPE parameters into the score expert, target model,
  and reference model before optimizing the new preference LoRA.

The training path predicts a fixed future chunk jointly. It does not yet
include minWM's inference-time KV cache or multi-chunk rollout pipeline.

## Metadata

World-model training requires a video-only CSV or JSON metadata file. Each row
must include:

```text
type,file_path,text,camera_pose_path,VQ_norm,MQ_norm,generated_motion_score_norm
video,clips/example.mp4,"prompt",poses/example.txt,0.8,0.7,0.6
```

`camera_pose_path` uses the existing CameraCtrl text format already consumed by
VideoX-Fun camera-control code: one header row followed by per-frame rows. The
loader reads normalized `[fx, fy, cx, cy]` from columns `1:5` and a `3x4`
world-to-camera matrix from columns `7:`.

## Stage1

Add these arguments to the existing Stage1 launch command:

```bash
--enable_world_model \
--camera_pose_column camera_pose_path \
--world_model_history_frames 1 \
--world_model_future_frames 4
```

`--world_model_history_frames` and `--world_model_future_frames` count VAE
latent frames. Pass `--world_model_modules_path` to warm-start PRoPE from an
earlier camera-world checkpoint. If omitted, Stage1 starts `prope_o` from zero.

## Stage2

Use the Stage1 LoRA and the Stage1 module file:

```bash
--enable_world_model \
--camera_pose_column camera_pose_path \
--world_model_history_frames 1 \
--world_model_future_frames 4 \
--stage1_lora_path /path/to/checkpoint-stage1.safetensors \
--stage1_score_path /path/to/score_modules-stage1.safetensors
```

By default, Stage2 also reads PRoPE weights from `--stage1_score_path`. Pass
`--world_model_modules_path` only when the PRoPE weights are stored separately.
