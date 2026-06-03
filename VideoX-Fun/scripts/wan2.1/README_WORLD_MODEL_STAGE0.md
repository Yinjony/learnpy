# Wan2.1 Camera World-Model Stage0

`train_world_model_ours.py` keeps the VideoX-Fun Accelerate training shell, but its
training step is camera-aware bidirectional SFT:

1. `CameraLatentLMDBDataset` loads pre-encoded Wan latents, prompts, camera
   `viewmats`, and intrinsics `Ks`.
2. Vendored `CameraBidirectionalDiffusion` creates the PRoPE-enabled
   `WanDiffusionWrapper(use_camera=True)`, frozen UMT5 encoder, and frozen VAE.
3. `generator_loss()` samples flow-matching timesteps and trains the full Wan
   generator, including the camera PRoPE path.
4. Accelerate owns gradient accumulation, distributed generator wrapping,
   optimizer state, resume, and checkpoint directories.

The copied Stage0 implementation lives under `videox_fun/world_model`. Training
does not import source files from a sibling `minWM` checkout.

## Install

Use a Linux CUDA environment. From `VideoX-Fun`:

```bash
# Install a CUDA-enabled PyTorch 2.1.2+ wheel appropriate for the host first.
pip install -r scripts/wan2.1/requirements_world_model.txt
pip install flash-attn --no-build-isolation
```

## Model Files

The trainer defaults to `VideoX-Fun/models/Wan2.1-T2V-1.3B`. Override it with
`WAN_MODEL_DIR=/path/to/Wan2.1-T2V-1.3B` when needed. From `VideoX-Fun`:

```bash
hf download Wan-AI/Wan2.1-T2V-1.3B \
    --local-dir ./models/Wan2.1-T2V-1.3B \
    --include "Wan2.1_VAE.pth" "models_t5_umt5-xxl-enc-bf16.pth" \
              "google/umt5-xxl/*" "diffusion_pytorch_model.safetensors" "config.json"
```

## Camera LMDB

From `VideoX-Fun`, build the camera-aware latent LMDB:

```bash
bash scripts/wan2.1/world_model_data_preprocessing/run_build_worldplaygen_lmdb.sh
```

The default training dataset path is `VideoX-Fun/dataset/Wan21/Action2V/data`.

## Preflight And Launch

Run the preflight before allocating a long training job:

```bash
python scripts/wan2.1/check_world_model_env.py
```

Then launch Stage0:

```bash
bash scripts/wan2.1/train_world_model_ours.sh
```

Override paths without editing the script:

```bash
WAN_MODEL_DIR=/path/to/Wan2.1-T2V-1.3B \
DATASET_NAME=/path/to/camera_lmdb \
OUTPUT_DIR=/path/to/output \
    bash scripts/wan2.1/train_world_model_ours.sh
```

Enable the GRPO strategy:

```bash
TRAINING_STRATEGY=grpo \
GRPO_GROUP_SIZE=4 \
GRPO_SFT_COEF=0.1 \
    bash scripts/wan2.1/train_world_model_ours.sh
```

The current GRPO reward is `reconstruction`: each LMDB sample is repeated as a
group with different diffusion noise/timesteps, reward is the negative
per-sample flow-matching loss, and the group-normalized advantage reweights the
diffusion loss with a small SFT anchor.

This entry supports Accelerate DDP/FSDP-style execution. It does not initialize
sequence parallelism. `--low_vram` is only valid when the text encoder is
not sharded, such as a single-process run.
