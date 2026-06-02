# Wan2.1 Camera World-Model Stage0

`train_dpo_ours copy.py` keeps the VideoX-Fun Accelerate training shell, but its
training step is minWM Phase 1 bidirectional camera SFT:

1. `CameraLatentLMDBDataset` loads pre-encoded Wan latents, prompts, camera
   `viewmats`, and intrinsics `Ks`.
2. minWM `CameraBidirectionalDiffusion` creates the PRoPE-enabled
   `WanDiffusionWrapper(use_camera=True)`, frozen UMT5 encoder, and frozen VAE.
3. `generator_loss()` samples flow-matching timesteps and trains the full Wan
   generator, including the camera PRoPE path.
4. Accelerate owns gradient accumulation, distributed generator wrapping,
   optimizer state, resume, and checkpoint directories.

## Install

Use a Linux CUDA environment. From `VideoX-Fun`:

```bash
# Install a CUDA-enabled PyTorch 2.5+ wheel appropriate for the host first.
pip install -r scripts/wan2.1/requirements_world_model.txt
pip install flash-attn --no-build-isolation
```

## Model Files

The trainer defaults to a sibling `minWM` checkout. Override it with
`MINWM_ROOT=/path/to/minWM` when needed. From `minWM`:

```bash
hf download Wan-AI/Wan2.1-T2V-1.3B \
    --local-dir ./ckpts/Wan2.1-T2V-1.3B \
    --include "Wan2.1_VAE.pth" "models_t5_umt5-xxl-enc-bf16.pth" \
              "google/umt5-xxl/*" "diffusion_pytorch_model.safetensors" "config.json"
mkdir -p Wan21/wan_models
ln -s "$(realpath ./ckpts/Wan2.1-T2V-1.3B)" \
    Wan21/wan_models/Wan2.1-T2V-1.3B
```

## Camera LMDB

From `minWM`, build the camera-aware latent LMDB:

```bash
bash Wan21/scripts/data_preprocessing/run_build_worldplaygen_lmdb.sh
```

The default training dataset path is `minWM/dataset/Wan21/Action2V/data`.

## Preflight And Launch

Run the preflight before allocating a long training job:

```bash
python scripts/wan2.1/check_world_model_env.py
```

Then launch Stage0:

```bash
accelerate launch "scripts/wan2.1/train_dpo_ours copy.py" \
    --mixed_precision bf16 \
    --gradient_checkpointing \
    --train_batch_size 1 \
    --learning_rate 2e-6 \
    --output_dir logs/world_model_stage0
```

This entry supports Accelerate DDP/FSDP-style execution. It does not initialize
minWM sequence parallelism. `--low_vram` is only valid when the text encoder is
not sharded, such as a single-process run.
