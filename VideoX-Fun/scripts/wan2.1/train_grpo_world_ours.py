"""Modified from https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py
"""
# !/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import argparse
import csv
import gc
import logging
import math
import pickle
import shutil
import sys

import decord

import accelerate
import diffusers
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision.transforms.functional as TF
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import DDIMScheduler, FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (EMAModel,
                                      compute_density_for_timestep_sampling,
                                      compute_loss_weighting_for_sd3)
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.torch_utils import is_compiled_module
from einops import rearrange
from omegaconf import OmegaConf
from packaging import version
from PIL import Image
from torch.utils.data import RandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from transformers.utils import ContextManagers

import datasets

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)),
                 os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None
from videox_fun.data.bucket_sampler import (ASPECT_RATIO_512,
                                            ASPECT_RATIO_RANDOM_CROP_512,
                                            ASPECT_RATIO_RANDOM_CROP_PROB,
                                            AspectRatioBatchImageVideoSampler,
                                            RandomSampler, get_closest_ratio)
from videox_fun.data.grpo_world_dataset import OfflineGRPOGroupDataset
from videox_fun.data.world_model_utils import (future_only_per_sample_mean,
                                               prepare_world_model_training_inputs)
#from videox_fun.models import (AutoencoderKLWan, CLIPModel, WanT5EncoderModel,
#                               WanTransformer3DModel)

from videox_fun.models import (AutoencoderKLWan, CLIPModel, WanT5EncoderModel)
from videox_fun.models import WanTransformer3DModel
from videox_fun.models.wan_prope import add_prope_parameters

from videox_fun.pipeline import WanI2VPipeline, WanPipeline
from videox_fun.utils.discrete_sampler import DiscreteSampling
from videox_fun.utils.lora_utils import (convert_peft_lora_to_kohya_lora,
                                         create_network, merge_lora,
                                         unmerge_lora)
from videox_fun.utils.grpo_utils import (diffusion_grpo_loss,
                                         normalize_group_advantages)
from videox_fun.utils.utils import (calculate_dimensions, get_image_latent,
                                    get_image_to_video_latent,
                                    save_videos_grid)

if is_wandb_available():
    import wandb

def filter_kwargs(cls, kwargs):
    import inspect
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {'self', 'cls'}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    return filtered_kwargs


def get_random_downsample_ratio(sample_size, image_ratio=[],
                                all_choices=False, rng=None):
    def _create_special_list(length):
        if length == 1:
            return [1.0]
        if length >= 2:
            first_element = 0.75
            remaining_sum = 1.0 - first_element
            other_elements_value = remaining_sum / (length - 1)
            special_list = [first_element] + [other_elements_value] * (length - 1)
            return special_list

    if sample_size >= 1536:
        number_list = [1, 1.25, 1.5, 2, 2.5, 3] + image_ratio
    elif sample_size >= 1024:
        number_list = [1, 1.25, 1.5, 2] + image_ratio
    elif sample_size >= 768:
        number_list = [1, 1.25, 1.5] + image_ratio
    elif sample_size >= 512:
        number_list = [1] + image_ratio
    else:
        number_list = [1]

    if all_choices:
        return number_list

    number_list_prob = np.array(_create_special_list(len(number_list)))
    if rng is None:
        return np.random.choice(number_list, p=number_list_prob)
    else:
        return rng.choice(number_list, p=number_list_prob)


def resize_mask(mask, latent, process_first_frame_only=True):
    latent_size = latent.size()
    batch_size, channels, num_frames, height, width = mask.shape

    if process_first_frame_only:
        target_size = list(latent_size[2:])
        target_size[0] = 1
        first_frame_resized = F.interpolate(
            mask[:, :, 0:1, :, :],
            size=target_size,
            mode='trilinear',
            align_corners=False
        )

        target_size = list(latent_size[2:])
        target_size[0] = target_size[0] - 1
        if target_size[0] != 0:
            remaining_frames_resized = F.interpolate(
                mask[:, :, 1:, :, :],
                size=target_size,
                mode='trilinear',
                align_corners=False
            )
            resized_mask = torch.cat([first_frame_resized, remaining_frames_resized], dim=2)
        else:
            resized_mask = first_frame_resized
    else:
        target_size = list(latent_size[2:])
        resized_mask = F.interpolate(
            mask,
            size=target_size,
            mode='trilinear',
            align_corners=False
        )
    return resized_mask


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.18.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def log_validation(vae, text_encoder, tokenizer, clip_image_encoder, transformer3d, network, args, config, accelerator,
                   weight_dtype, global_step):
    try:
        is_deepspeed = type(transformer3d).__name__ == 'DeepSpeedEngine'
        if is_deepspeed:
            origin_config = transformer3d.config
            transformer3d.config = accelerator.unwrap_model(transformer3d).config
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=weight_dtype), torch.cuda.device(device=accelerator.device):
            logger.info("Running validation... ")
            scheduler = FlowMatchEulerDiscreteScheduler(
                **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config['scheduler_kwargs']))
            )

            if args.train_mode != "normal":
                pipeline = WanI2VPipeline(
                    vae=vae,
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    transformer=accelerator.unwrap_model(transformer3d) if type(
                        transformer3d).__name__ == 'DistributedDataParallel' else transformer3d,
                    scheduler=scheduler,
                    clip_image_encoder=clip_image_encoder,
                )
            else:
                pipeline = WanPipeline(
                    vae=vae,
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    transformer=accelerator.unwrap_model(transformer3d) if type(
                        transformer3d).__name__ == 'DistributedDataParallel' else transformer3d,
                    scheduler=scheduler,
                )
            pipeline = pipeline.to(accelerator.device)

            if args.seed is None:
                generator = None
            else:
                rank_seed = args.seed + accelerator.process_index
                generator = torch.Generator(device=accelerator.device).manual_seed(rank_seed)
                logger.info(f"Rank {accelerator.process_index} using seed: {rank_seed}")

            for i in range(len(args.validation_prompts)):
                if args.train_mode != "normal":
                    start_image = Image.open(args.validation_paths[i])
                    width, height = start_image.width, start_image.height
                    width, height = calculate_dimensions(args.image_sample_size * args.image_sample_size,
                                                         width / height)

                    video_length = int((
                                               args.video_sample_n_frames - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1 if args.video_sample_n_frames != 1 else 1
                    input_video, input_video_mask, _ = get_image_to_video_latent(args.validation_paths[i], None,
                                                                                 video_length=video_length,
                                                                                 sample_size=[height, width])
                    sample = pipeline(
                        args.validation_prompts[i],
                        num_frames=video_length,
                        negative_prompt="bad detailed",
                        height=height,
                        width=width,
                        generator=generator,

                        video=input_video,
                        mask_video=input_video_mask,
                        num_inference_steps=25,
                        guidance_scale=4.5,
                    ).videos

                    os.makedirs(os.path.join(args.output_dir, "sample"), exist_ok=True)
                    save_videos_grid(
                        sample,
                        os.path.join(
                            args.output_dir,
                            f"sample/sample-{global_step}-rank{accelerator.process_index}-image-{i}.gif"
                        )
                    )
                else:
                    sample = pipeline(
                        args.validation_prompts[i],
                        num_frames=args.video_sample_n_frames,
                        negative_prompt="bad detailed",
                        height=args.video_sample_size,
                        width=args.video_sample_size,
                        generator=generator,
                        num_inference_steps=25,
                        guidance_scale=4.5,
                    ).videos
                    os.makedirs(os.path.join(args.output_dir, "sample"), exist_ok=True)
                    save_videos_grid(
                        sample,
                        os.path.join(
                            args.output_dir,
                            f"sample/sample-{global_step}-rank{accelerator.process_index}-image-{i}.gif"
                        )
                    )

            del pipeline
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            vae.to(accelerator.device if not args.low_vram else "cpu", dtype=weight_dtype)
            transformer3d.to(accelerator.device, dtype=weight_dtype)
            if not args.enable_text_encoder_in_dataloader:
                text_encoder.to(accelerator.device if not args.low_vram else "cpu", dtype=weight_dtype)
        if is_deepspeed:
            transformer3d.config = origin_config
    except Exception as e:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print(f"Eval error on rank {accelerator.process_index} with info {e}")
        vae.to(accelerator.device if not args.low_vram else "cpu", dtype=weight_dtype)
        transformer3d.to(accelerator.device, dtype=weight_dtype)
        if not args.enable_text_encoder_in_dataloader:
            text_encoder.to(accelerator.device if not args.low_vram else "cpu", dtype=weight_dtype)


def linear_decay(initial_value, final_value, total_steps, current_step):
    if current_step >= total_steps:
        return final_value
    current_step = max(0, current_step)
    step_size = (final_value - initial_value) / total_steps
    current_value = initial_value + step_size * current_step
    return current_value


def generate_timestep_with_lognorm(low, high, shape, device="cpu", generator=None):
    u = torch.normal(mean=0.0, std=1.0, size=shape, device=device, generator=generator)
    t = 1 / (1 + torch.exp(-u)) * (high - low) + low
    return torch.clip(t.to(torch.int32), low, high - 1)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--input_perturbation", type=float, default=0, help="The scale of input perturbation. Recommended 0.1."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. "
        ),
    )
    parser.add_argument(
        "--train_data_meta",
        type=str,
        default=None,
        help=(
            "A csv containing the training data. "
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--validation_prompts",
        type=str,
        default=None,
        nargs="+",
        help=("A set of prompts evaluated every `--validation_epochs` and logged to `--report_to`."),
    )
    parser.add_argument(
        "--validation_paths",
        type=str,
        default=None,
        nargs="+",
        help=("A set of control videos evaluated every `--validation_epochs` and logged to `--report_to`."),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--use_came",
        action="store_true",
        help="whether to use came",
    )
    parser.add_argument(
        "--multi_stream",
        action="store_true",
        help="whether to use cuda multi-stream",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--vae_mini_batch", type=int, default=32, help="mini batch size for vae."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=5,
        help="Run validation every X epochs.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=2000,
        help="Run validation every X steps.",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="text2image-fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    parser.add_argument(
        "--rank",
        type=int,
        default=128,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--network_alpha",
        type=int,
        default=64,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--use_peft_lora", action="store_true", help="Whether or not to use peft lora."
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )
    parser.add_argument(
        "--snr_loss", action="store_true", help="Whether or not to use snr_loss."
    )
    parser.add_argument(
        "--uniform_sampling", action="store_true", help="Whether or not to use uniform_sampling."
    )
    parser.add_argument(
        "--enable_text_encoder_in_dataloader", action="store_true",
        help="Whether or not to use text encoder in dataloader."
    )
    parser.add_argument(
        "--enable_bucket", action="store_true", help="Whether enable bucket sample in datasets."
    )
    parser.add_argument(
        "--random_ratio_crop", action="store_true", help="Whether enable random ratio crop sample in datasets."
    )
    parser.add_argument(
        "--random_frame_crop", action="store_true", help="Whether enable random frame crop sample in datasets."
    )
    parser.add_argument(
        "--random_hw_adapt", action="store_true", help="Whether enable random adapt height and width in datasets."
    )
    parser.add_argument(
        "--training_with_video_token_length", action="store_true", help="The training stage of the model in training.",
    )
    parser.add_argument(
        "--auto_tile_batch_size", action="store_true", help="Whether to auto tile batch size.",
    )
    parser.add_argument(
        "--noise_share_in_frames", action="store_true", help="Whether enable noise share in frames."
    )
    parser.add_argument(
        "--noise_share_in_frames_ratio", type=float, default=0.5, help="Noise share ratio.",
    )
    parser.add_argument(
        "--motion_sub_loss", action="store_true", help="Whether enable motion sub loss."
    )
    parser.add_argument(
        "--motion_sub_loss_ratio", type=float, default=0.25, help="The ratio of motion sub loss."
    )
    parser.add_argument(
        "--keep_all_node_same_token_length",
        action="store_true",
        help="Reference of the length token.",
    )
    parser.add_argument(
        "--train_sampling_steps",
        type=int,
        default=1000,
        help="Run train_sampling_steps.",
    )
    parser.add_argument(
        "--token_sample_size",
        type=int,
        default=512,
        help="Sample size of the token.",
    )
    parser.add_argument(
        "--video_sample_size",
        type=int,
        default=512,
        help="Sample size of the video.",
    )
    parser.add_argument(
        "--image_sample_size",
        type=int,
        default=512,
        help="Sample size of the image.",
    )
    parser.add_argument(
        "--fix_sample_size",
        nargs=2, type=int, default=None,
        help="Fix Sample size [height, width] when using bucket and collate_fn."
    )
    parser.add_argument(
        "--video_sample_stride",
        type=int,
        default=4,
        help="Sample stride of the video.",
    )
    parser.add_argument(
        "--video_sample_n_frames",
        type=int,
        default=17,
        help="Num frame of video.",
    )
    parser.add_argument(
        "--video_repeat",
        type=int,
        default=0,
        help="Num of repeat video.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help=(
            "The config of the model in training."
        ),
    )
    parser.add_argument(
        "--transformer_path",
        type=str,
        default=None,
        help=("If you want to load the weight from other transformers, input its path."),
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default=None,
        help=("If you want to load the weight from other vaes, input its path."),
    )
    parser.add_argument("--save_state", action="store_true", help="Whether or not to save state.")

    parser.add_argument(
        '--tokenizer_max_length',
        type=int,
        default=512,
        help='Max length of tokenizer'
    )
    parser.add_argument(
        "--use_deepspeed", action="store_true", help="Whether or not to use deepspeed."
    )
    parser.add_argument(
        "--use_fsdp", action="store_true", help="Whether or not to use fsdp."
    )
    parser.add_argument(
        "--low_vram", action="store_true", help="Whether enable low_vram mode."
    )
    parser.add_argument(
        "--train_mode",
        type=str,
        default="normal",
        help=(
            'The format of training data. Support `"normal"`'
            ' (default), `"i2v"`.'
        ),
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help=('We default to the "none" weighting scheme for uniform sampling and uniform loss'),
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument(
        "--lora_skip_name",
        type=str,
        default=None,
        help=("The module is not trained in loras. "),
    )
    parser.add_argument(
        "--target_name",
        type=str,
        default=None,
        help=("The module is trained in loras. "),
    )

    parser.add_argument("--grpo_group_size", type=int, default=4)
    parser.add_argument("--grpo_beta", type=float, default=1.0)
    parser.add_argument("--grpo_clip_range", type=float, default=0.2)
    parser.add_argument("--grpo_kl_coeff", type=float, default=0.01)
    parser.add_argument("--reward_vq_column", type=str, default="VQ")
    parser.add_argument("--reward_motion_column", type=str, default="generated_motion_score")
    parser.add_argument("--reward_vq_weight", type=float, default=0.5)
    parser.add_argument("--reward_motion_weight", type=float, default=0.5)
    parser.add_argument("--enable_world_model", action="store_true")
    parser.add_argument("--history_video_column", type=str, default="original_video")
    parser.add_argument("--camera_pose_column", type=str, default="camera_pose_path")
    parser.add_argument("--world_model_history_frames", type=int, default=1)
    parser.add_argument("--world_model_future_frames", type=int, default=4)
    parser.add_argument("--world_model_modules_path", type=str, default=None)

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


def main():
    args = parse_args()
    if args.train_mode != "normal":
        raise ValueError("GRPO world-model training currently supports --train_mode normal only.")
    if not args.enable_world_model:
        raise ValueError("This entry point requires --enable_world_model.")
    if args.world_model_history_frames < 1:
        raise ValueError("--world_model_history_frames must be at least 1.")
    if args.world_model_modules_path is None:
        raise ValueError("--world_model_modules_path is required for GRPO world-model training.")

    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    config = OmegaConf.load(args.config_path)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    deepspeed_plugin = accelerator.state.deepspeed_plugin if hasattr(accelerator.state, "deepspeed_plugin") else None
    fsdp_plugin = accelerator.state.fsdp_plugin if hasattr(accelerator.state, "fsdp_plugin") else None
    if deepspeed_plugin is not None:
        zero_stage = int(deepspeed_plugin.zero_stage)
        fsdp_stage = 0
        print(f"Using DeepSpeed Zero stage: {zero_stage}")

        args.use_deepspeed = True
        if zero_stage == 3:
            print(f"Auto set save_state to True because zero_stage == 3")
            args.save_state = True
    elif fsdp_plugin is not None:
        from torch.distributed.fsdp import ShardingStrategy
        zero_stage = 0
        if fsdp_plugin.sharding_strategy is ShardingStrategy.FULL_SHARD:
            fsdp_stage = 3
        elif fsdp_plugin.sharding_strategy is None:  # The fsdp_plugin.sharding_strategy is None in FSDP 2.
            fsdp_stage = 3
        elif fsdp_plugin.sharding_strategy is ShardingStrategy.SHARD_GRAD_OP:
            fsdp_stage = 2
        else:
            fsdp_stage = 0
        print(f"Using FSDP stage: {fsdp_stage}")

        args.use_fsdp = True
        if fsdp_stage == 3:
            print(f"Auto set save_state to True because fsdp_stage == 3")
            args.save_state = True
    else:
        zero_stage = 0
        fsdp_stage = 0
        print("DeepSpeed is not enabled.")

    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir=logging_dir)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
        rng = np.random.default_rng(np.random.PCG64(args.seed + accelerator.process_index))
        torch_rng = torch.Generator(accelerator.device).manual_seed(args.seed + accelerator.process_index)
    else:
        rng = None
        torch_rng = None
    index_rng = np.random.default_rng(np.random.PCG64(43))
    process_seed = None if args.seed is None else args.seed + accelerator.process_index
    print(f"Init rng with seed {process_seed}. Process_index is {accelerator.process_index}")

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)



    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora transformer3d) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # Load scheduler, tokenizer and models.
    noise_scheduler = FlowMatchEulerDiscreteScheduler(
        **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config['scheduler_kwargs']))
    )

    # Get Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path,
                     config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
    )

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
    # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
    # will try to assign the same optimizer with the same weights to all models during
    # `deepspeed.initialize`, which of course doesn't work.
    #
    # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
    # frozen models from being partitioned during `zero.Init` which gets called during
    # `from_pretrained` So CLIPTextModel and AutoencoderKL will not enjoy the parameter sharding
    # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        # Get Text encoder
        text_encoder = WanT5EncoderModel.from_pretrained(
            os.path.join(args.pretrained_model_name_or_path,
                         config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
            additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
            low_cpu_mem_usage=True,
            torch_dtype=weight_dtype,
        )
        text_encoder = text_encoder.eval()
        # Get Vae
        vae = AutoencoderKLWan.from_pretrained(
            os.path.join(args.pretrained_model_name_or_path, config['vae_kwargs'].get('vae_subpath', 'vae')),
            additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
        )
        vae.eval()
        # Get Clip Image Encoder
        if args.train_mode != "normal":
            clip_image_encoder = CLIPModel.from_pretrained(
                os.path.join(args.pretrained_model_name_or_path,
                             config['image_encoder_kwargs'].get('image_encoder_subpath', 'image_encoder')),
            )
            clip_image_encoder = clip_image_encoder.eval()
        else:
            clip_image_encoder = None

    # Target world model: frozen PRoPE base with a trainable LoRA adapter.
    transformer3d = WanTransformer3DModel.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path,
                     config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
    ).to(weight_dtype)

    # Reference world model: frozen PRoPE base without the target LoRA adapter.
    transformer_ref = WanTransformer3DModel.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path,
                     config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
    ).to(weight_dtype)
    add_prope_parameters(transformer3d)
    add_prope_parameters(transformer_ref)
    from safetensors.torch import load_file
    world_model_state_dict = load_file(args.world_model_modules_path)
    if not any(".prope_o." in key for key in world_model_state_dict):
        raise ValueError("--world_model_modules_path does not contain learned PRoPE parameters.")
    m, u = transformer3d.load_state_dict(world_model_state_dict, strict=False)
    print(f"theta world-model modules loaded - missing: {len(m)}, unexpected: {len(u)}")
    m, u = transformer_ref.load_state_dict(world_model_state_dict, strict=False)
    print(f"theta_ref world-model modules loaded - missing: {len(m)}, unexpected: {len(u)}")


    # Freeze vae and text_encoder and set transformer3d to trainable
    vae.requires_grad_(False)

    text_encoder.requires_grad_(False)

    transformer3d.requires_grad_(False)

    transformer_ref.requires_grad_(False)
    transformer_ref.eval()


    if args.train_mode != "normal":
        clip_image_encoder.requires_grad_(False)

    # Lora will work with this...
    if args.use_peft_lora:
        from peft import (LoraConfig, get_peft_model_state_dict,
                          inject_adapter_in_model)
        lora_config = LoraConfig(r=args.rank, lora_alpha=args.network_alpha, target_modules=args.target_name.split(","))
        transformer3d = inject_adapter_in_model(lora_config, transformer3d)

        network = None
    else:
        network = create_network(
            1.0,
            args.rank,
            args.network_alpha,
            text_encoder,
            transformer3d,
            neuron_dropout=None,
            target_name=args.target_name,
            skip_name=args.lora_skip_name,
        )
        network = network.to(weight_dtype)
        network.apply_to(text_encoder, transformer3d,
                         args.train_text_encoder and not args.training_with_video_token_length, True)


    if args.transformer_path is not None:
        print(f"From checkpoint: {args.transformer_path}")
        if args.transformer_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(args.transformer_path)
        else:
            state_dict = torch.load(args.transformer_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        m, u = transformer3d.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
        assert len(u) == 0

    if args.vae_path is not None:
        print(f"From checkpoint: {args.vae_path}")
        if args.vae_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(args.vae_path)
        else:
            state_dict = torch.load(args.vae_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        m, u = vae.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
        assert len(u) == 0

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        if fsdp_stage != 0 or zero_stage == 3:
            def save_model_hook(models, weights, output_dir):
                accelerate_state_dict = accelerator.get_state_dict(models[-1], unwrap=True)
                if accelerator.is_main_process:
                    from safetensors.torch import save_file
                    safetensor_save_path = os.path.join(output_dir, f"lora_diffusion_pytorch_model.safetensors")
                    if args.use_peft_lora:
                        network_state_dict = get_peft_model_state_dict(accelerator.unwrap_model(models[-1]),
                                                                       accelerate_state_dict)
                        network_state_dict_kohya = convert_peft_lora_to_kohya_lora(network_state_dict)
                        safetensor_kohya_format_save_path = os.path.join(output_dir,
                                                                         f"lora_diffusion_pytorch_model_compatible_with_comfyui.safetensors")
                        save_model(safetensor_kohya_format_save_path, network_state_dict_kohya)
                    else:
                        network_state_dict = {}
                        for key in accelerate_state_dict:
                            if "network" in key:
                                network_state_dict[key.replace("network.", "")] = accelerate_state_dict[key].to(
                                    weight_dtype)
                    save_file(network_state_dict, safetensor_save_path, metadata={"format": "pt"})

                    with open(os.path.join(output_dir, "sampler_pos_start.pkl"), 'wb') as file:
                        pickle.dump([0, first_epoch], file)

            def load_model_hook(models, input_dir):
                pkl_path = os.path.join(input_dir, "sampler_pos_start.pkl")
                if os.path.exists(pkl_path):
                    with open(pkl_path, 'rb') as file:
                        loaded_number, _ = pickle.load(file)
                    print(f"Load pkl from {pkl_path}. Get loaded_number = {loaded_number}.")

        else:
            # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
            def save_model_hook(models, weights, output_dir):
                accelerate_state_dict = accelerator.get_state_dict(models[-1], unwrap=True)
                if accelerator.is_main_process:
                    from safetensors.torch import save_file
                    safetensor_save_path = os.path.join(output_dir, f"lora_diffusion_pytorch_model.safetensors")
                    if args.use_peft_lora:
                        network_state_dict = get_peft_model_state_dict(accelerator.unwrap_model(models[-1]),
                                                                       accelerate_state_dict)
                        network_state_dict_kohya = convert_peft_lora_to_kohya_lora(network_state_dict)
                        safetensor_kohya_format_save_path = os.path.join(output_dir,
                                                                         f"lora_diffusion_pytorch_model_compatible_with_comfyui.safetensors")
                        save_model(safetensor_kohya_format_save_path, network_state_dict_kohya)
                    else:
                        network_state_dict = {}
                        for key in accelerate_state_dict:
                            if "network" in key:
                                network_state_dict[key.replace("network.", "")] = accelerate_state_dict[key].to(
                                    weight_dtype)
                    save_file(network_state_dict, safetensor_save_path, metadata={"format": "pt"})

                    if not args.use_deepspeed:
                        for _ in range(len(weights)):
                            weights.pop()

                    with open(os.path.join(output_dir, "sampler_pos_start.pkl"), 'wb') as file:
                        pickle.dump([0, first_epoch], file)

            def load_model_hook(models, input_dir):
                pkl_path = os.path.join(input_dir, "sampler_pos_start.pkl")
                if os.path.exists(pkl_path):
                    with open(pkl_path, 'rb') as file:
                        loaded_number, _ = pickle.load(file)
                    print(f"Load pkl from {pkl_path}. Get loaded_number = {loaded_number}.")

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        transformer3d.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
                args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    elif args.use_came:
        try:
            from came_pytorch import CAME
        except Exception:
            raise ImportError(
                "Please install came_pytorch to use CAME. You can do so by running `pip install came_pytorch`"
            )

        optimizer_cls = CAME
    else:
        optimizer_cls = torch.optim.AdamW

    if args.use_peft_lora:
        logging.info("Add peft parameters")
        trainable_params = list(filter(lambda p: p.requires_grad, transformer3d.parameters()))
        trainable_params_optim = list(filter(lambda p: p.requires_grad, transformer3d.parameters()))
    else:
        logging.info("Add network parameters")
        trainable_params = list(filter(lambda p: p.requires_grad, network.parameters()))
        trainable_params_optim = network.prepare_optimizer_params(args.learning_rate / 2, args.learning_rate,
                                                                  args.learning_rate)

    if args.use_came:
        optimizer = optimizer_cls(
            trainable_params_optim,
            lr=args.learning_rate,
            # weight_decay=args.adam_weight_decay,
            betas=(0.9, 0.999, 0.9999),
            eps=(1e-30, 1e-16)
        )
    else:
        optimizer = optimizer_cls(
            trainable_params_optim,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    # Each dataset item is one world state with a group of offline rollout candidates.
    train_dataset = OfflineGRPOGroupDataset(
        csv_path=args.train_data_meta,
        video_dir=args.train_data_dir,
        group_size=args.grpo_group_size,
        video_sample_n_frames=args.video_sample_n_frames,
        video_sample_stride=args.video_sample_stride,
        reward_vq_column=args.reward_vq_column,
        reward_motion_column=args.reward_motion_column,
        reward_vq_weight=args.reward_vq_weight,
        reward_motion_weight=args.reward_motion_weight,
        enable_world_model=args.enable_world_model,
        history_video_column=args.history_video_column,
        camera_pose_column=args.camera_pose_column,
    )

    sample_n_frames_bucket_interval = vae.config.temporal_compression_ratio

    def collate_fn_grpo(examples):
        new_examples = {}
        new_examples["text"] = [ex["text"] for ex in examples]
        new_examples["rewards"] = torch.stack([ex["rewards"] for ex in examples])
        new_examples["viewmats"] = torch.stack([ex["viewmats"] for ex in examples])
        new_examples["Ks"] = torch.stack([ex["Ks"] for ex in examples])

        def transform_video(frames):
            pixel_values = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous().float() / 255.0
            h, w = pixel_values.shape[2], pixel_values.shape[3]
            size = args.video_sample_size
            if h > w:
                resize_h, resize_w = int(round(h * size / w)), size
            else:
                resize_h, resize_w = size, int(round(w * size / h))
            transform = transforms.Compose([
                transforms.Resize([resize_h, resize_w],
                                  interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop([size, size]),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ])
            return transform(pixel_values)[:args.video_sample_n_frames]

        group_pixel_values = [
            [transform_video(frames) for frames in example["group_pixel_values"]]
            for example in examples
        ]
        history_pixel_values = [transform_video(example["history_pixel_values"]) for example in examples]
        min_f = min(
            [video.shape[0] for group in group_pixel_values for video in group]
            + [video.shape[0] for video in history_pixel_values]
        )
        min_f = max((min_f - 1) // sample_n_frames_bucket_interval * sample_n_frames_bucket_interval + 1, 1)
        new_examples["group_pixel_values"] = torch.stack([
            torch.stack([video[:min_f] for video in group]) for group in group_pixel_values
        ])
        new_examples["history_pixel_values"] = torch.stack([video[:min_f] for video in history_pixel_values])
        new_examples["viewmats"] = new_examples["viewmats"][:, :min_f]
        new_examples["Ks"] = new_examples["Ks"][:, :min_f]

        if args.enable_text_encoder_in_dataloader:
            prompt_ids = tokenizer(
                new_examples['text'], max_length=args.tokenizer_max_length,
                padding="max_length", add_special_tokens=True,
                truncation=True, return_tensors="pt"
            )
            new_examples['encoder_hidden_states'] = text_encoder(prompt_ids.input_ids)[0]
            new_examples['encoder_attention_mask'] = prompt_ids.attention_mask

        return new_examples

    batch_sampler_generator = torch.Generator()
    if args.seed is not None:
        batch_sampler_generator.manual_seed(args.seed)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn_grpo,
        num_workers=args.dataloader_num_workers,
        persistent_workers=True if args.dataloader_num_workers != 0 else False,
        drop_last=True,
        generator=batch_sampler_generator,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    if args.use_peft_lora:
        transformer3d, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            transformer3d, optimizer, train_dataloader, lr_scheduler
        )
    else:
        transformer3d.network = network
        transformer3d = transformer3d.to(dtype=weight_dtype)

        transformer3d, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            transformer3d, optimizer, train_dataloader, lr_scheduler
        )

    if fsdp_stage != 0 or zero_stage != 0:
        from functools import partial

        from videox_fun.dist import set_multi_gpus_devices, shard_model
        shard_fn = partial(shard_model, device_id=accelerator.device, param_dtype=weight_dtype)
        text_encoder = shard_fn(text_encoder)

    # Move text_encode and vae to gpu and cast to weight_dtype
    vae.to(accelerator.device if not args.low_vram else "cpu", dtype=weight_dtype)
    transformer3d.to(accelerator.device, dtype=weight_dtype)
    transformer_ref.to(accelerator.device, dtype=weight_dtype)

    if not args.enable_text_encoder_in_dataloader:
        text_encoder.to(accelerator.device if not args.low_vram else "cpu", dtype=weight_dtype)
    if args.train_mode != "normal":
        clip_image_encoder.to(accelerator.device if not args.low_vram else "cpu", dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        keys_to_pop = [k for k, v in tracker_config.items() if isinstance(v, list)]
        for k in keys_to_pop:
            tracker_config.pop(k)
            print(f"Removed tracker_config['{k}']")
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # Function for unwrapping if model was compiled with `torch.compile`.
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            global_step = int(path.split("-")[1])

            initial_global_step = global_step

            checkpoint_folder_path = os.path.join(args.output_dir, path)
            pkl_path = os.path.join(checkpoint_folder_path, "sampler_pos_start.pkl")
            if os.path.exists(pkl_path):
                with open(pkl_path, 'rb') as file:
                    _, first_epoch = pickle.load(file)
            else:
                first_epoch = global_step // num_update_steps_per_epoch
            print(f"Load pkl from {pkl_path}. Get first_epoch = {first_epoch}.")

            if zero_stage != 3 and not args.use_fsdp:
                from safetensors.torch import load_file
                state_dict = load_file(os.path.join(checkpoint_folder_path, "lora_diffusion_pytorch_model.safetensors"),
                                       device=str(accelerator.device))
                m, u = accelerator.unwrap_model(network).load_state_dict(state_dict, strict=False)
                print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

                optimizer_file_pt = os.path.join(checkpoint_folder_path, "optimizer.pt")
                optimizer_file_bin = os.path.join(checkpoint_folder_path, "optimizer.bin")
                optimizer_file_to_load = None

                if os.path.exists(optimizer_file_pt):
                    optimizer_file_to_load = optimizer_file_pt
                elif os.path.exists(optimizer_file_bin):
                    optimizer_file_to_load = optimizer_file_bin

                if optimizer_file_to_load:
                    try:
                        accelerator.print(f"Loading optimizer state from {optimizer_file_to_load}")
                        optimizer_state = torch.load(optimizer_file_to_load, map_location=accelerator.device)
                        optimizer.load_state_dict(optimizer_state)
                        accelerator.print("Optimizer state loaded successfully.")
                    except Exception as e:
                        accelerator.print(f"Failed to load optimizer state from {optimizer_file_to_load}: {e}")

                scheduler_file_pt = os.path.join(checkpoint_folder_path, "scheduler.pt")
                scheduler_file_bin = os.path.join(checkpoint_folder_path, "scheduler.bin")
                scheduler_file_to_load = None

                if os.path.exists(scheduler_file_pt):
                    scheduler_file_to_load = scheduler_file_pt
                elif os.path.exists(scheduler_file_bin):
                    scheduler_file_to_load = scheduler_file_bin

                if scheduler_file_to_load:
                    try:
                        accelerator.print(f"Loading scheduler state from {scheduler_file_to_load}")
                        scheduler_state = torch.load(scheduler_file_to_load, map_location=accelerator.device)
                        lr_scheduler.load_state_dict(scheduler_state)
                        accelerator.print("Scheduler state loaded successfully.")
                    except Exception as e:
                        accelerator.print(f"Failed to load scheduler state from {scheduler_file_to_load}: {e}")

                if hasattr(accelerator, 'scaler') and accelerator.scaler is not None:
                    scaler_file = os.path.join(checkpoint_folder_path, "scaler.pt")
                    if os.path.exists(scaler_file):
                        try:
                            accelerator.print(f"Loading GradScaler state from {scaler_file}")
                            scaler_state = torch.load(scaler_file, map_location=accelerator.device)
                            accelerator.scaler.load_state_dict(scaler_state)
                            accelerator.print("GradScaler state loaded successfully.")
                        except Exception as e:
                            accelerator.print(f"Failed to load GradScaler state: {e}")

            else:
                accelerator.load_state(checkpoint_folder_path)
                accelerator.print("accelerator.load_state() completed for zero_stage 3.")

    else:
        initial_global_step = 0

    # function for saving/removing
    def save_model(ckpt_file, unwrapped_nw):
        os.makedirs(args.output_dir, exist_ok=True)
        accelerator.print(f"\nsaving checkpoint: {ckpt_file}")
        if isinstance(unwrapped_nw, dict):
            from safetensors.torch import save_file
            save_file(unwrapped_nw, ckpt_file, metadata={"format": "pt"})
            return ckpt_file
        unwrapped_nw.save_weights(ckpt_file, weight_dtype, None)

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    if args.multi_stream and args.train_mode != "normal":
        # create extra cuda streams to speedup inpaint vae computation
        vae_stream_1 = torch.cuda.Stream()
        vae_stream_2 = torch.cuda.Stream()
    else:
        vae_stream_1 = None
        vae_stream_2 = None

    idx_sampling = DiscreteSampling(args.train_sampling_steps, uniform_sampling=args.uniform_sampling)

    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            if epoch == first_epoch and step == 0:
                # Sanity check: save the first candidate from each rollout group.
                candidate_pv = batch["group_pixel_values"][:, 0].cpu()
                candidate_pv = rearrange(candidate_pv, "b f c h w -> b c f h w")
                os.makedirs(os.path.join(args.output_dir, "sanity_check"), exist_ok=True)
                texts = batch["text"]
                for idx, (pv, text) in enumerate(zip(candidate_pv, texts)):
                    pv = pv[None, ...]
                    gif_name = '-'.join(text.replace('/', '').split()[:10]) if text else f'{global_step}-{idx}'
                    save_videos_grid(pv, f"{args.output_dir}/sanity_check/candidate_{gif_name[:10]}.gif", rescale=True)

            with accelerator.accumulate(transformer3d):
                group_pixel_values = batch["group_pixel_values"].to(weight_dtype)
                history_pixel_values = batch["history_pixel_values"].to(weight_dtype)
                batch_size, group_size = group_pixel_values.shape[:2]
                flat_pixel_values = rearrange(group_pixel_values, "b g f c h w -> (b g) f c h w")

                if args.low_vram:
                    torch.cuda.empty_cache()
                    vae.to(accelerator.device)
                    if not args.enable_text_encoder_in_dataloader:
                        text_encoder.to("cpu")

                with torch.no_grad():
                    def _batch_encode_vae(pixel_values):
                        pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
                        bs = args.vae_mini_batch
                        new_pixel_values = []
                        for i in range(0, pixel_values.shape[0], bs):
                            pixel_values_bs = pixel_values[i: i + bs]
                            pixel_values_bs = vae.encode(pixel_values_bs)[0]
                            pixel_values_bs = pixel_values_bs.sample()
                            new_pixel_values.append(pixel_values_bs)
                        return torch.cat(new_pixel_values, dim=0)

                    candidate_latents = _batch_encode_vae(flat_pixel_values)
                    history_latents = _batch_encode_vae(history_pixel_values)
                    history_latents = history_latents.repeat_interleave(group_size, dim=0)

                if args.low_vram:
                    vae.to('cpu')
                    torch.cuda.empty_cache()
                    if not args.enable_text_encoder_in_dataloader:
                        text_encoder.to(accelerator.device)

                if args.enable_text_encoder_in_dataloader:
                    prompt_embeds = batch["encoder_hidden_states"].to(device=candidate_latents.device)
                    prompt_embeds = prompt_embeds.repeat_interleave(group_size, dim=0)
                else:
                    with torch.no_grad():
                        prompt_ids = tokenizer(
                            batch["text"],
                            padding="max_length",
                            max_length=args.tokenizer_max_length,
                            truncation=True,
                            add_special_tokens=True,
                            return_tensors="pt"
                        )
                        text_input_ids = prompt_ids.input_ids
                        prompt_attention_mask = prompt_ids.attention_mask

                        seq_lens = prompt_attention_mask.gt(0).sum(dim=1).long()
                        prompt_embeds = text_encoder(text_input_ids.to(candidate_latents.device),
                                                     attention_mask=prompt_attention_mask.to(candidate_latents.device))[0]
                        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
                        prompt_embeds = [u for u in prompt_embeds for _ in range(group_size)]

                if args.low_vram and not args.enable_text_encoder_in_dataloader:
                    text_encoder.to('cpu')
                    torch.cuda.empty_cache()

                flat_batch_size = candidate_latents.shape[0]

                # Use common random numbers inside each group so the relative
                # advantage reflects candidate quality rather than extra noise.
                noise = torch.randn(
                    (batch_size, 1, *candidate_latents.shape[1:]),
                    device=candidate_latents.device,
                    generator=torch_rng,
                    dtype=weight_dtype,
                )
                noise = noise.expand(-1, group_size, *([-1] * (noise.ndim - 2)))
                noise = noise.reshape(flat_batch_size, *candidate_latents.shape[1:])

                if not args.uniform_sampling:
                    u = compute_density_for_timestep_sampling(
                        weighting_scheme=args.weighting_scheme,
                        batch_size=batch_size,
                        logit_mean=args.logit_mean,
                        logit_std=args.logit_std,
                        mode_scale=args.mode_scale,
                    )
                    indices = (u * noise_scheduler.config.num_train_timesteps).long()
                else:
                    indices = idx_sampling(batch_size, generator=torch_rng, device=candidate_latents.device)
                    indices = indices.long().cpu()
                timesteps = noise_scheduler.timesteps[indices].to(device=candidate_latents.device)
                timesteps = timesteps.repeat_interleave(group_size, dim=0)

                def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
                    sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
                    schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
                    timesteps = timesteps.to(accelerator.device)
                    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

                    sigma = sigmas[step_indices].flatten()
                    while len(sigma.shape) < n_dim:
                        sigma = sigma.unsqueeze(-1)
                    return sigma

                sigmas = get_sigmas(timesteps, n_dim=candidate_latents.ndim, dtype=candidate_latents.dtype)
                world_batch = prepare_world_model_training_inputs(
                    latents=candidate_latents,
                    history_latents=history_latents,
                    noise=noise,
                    sigmas=sigmas,
                    timesteps=timesteps,
                    viewmats=batch["viewmats"].repeat_interleave(group_size, dim=0),
                    Ks=batch["Ks"].repeat_interleave(group_size, dim=0),
                    patch_size=accelerator.unwrap_model(transformer3d).config.patch_size,
                    history_frames=args.world_model_history_frames,
                    future_frames=args.world_model_future_frames,
                )
                noisy_latents = world_batch["noisy_latents"]
                target = world_batch["target"]
                model_timesteps = world_batch["timesteps"]
                future_mask = world_batch["future_mask"]
                viewmats = world_batch["viewmats"]
                Ks = world_batch["Ks"]
                _, _, num_frames, height, width = noisy_latents.size()

                target_shape = (vae.latent_channels, num_frames, width, height)
                seq_len = math.ceil(
                    (target_shape[2] * target_shape[3]) /
                    (accelerator.unwrap_model(transformer3d).config.patch_size[1] *
                     accelerator.unwrap_model(transformer3d).config.patch_size[2]) *
                    target_shape[1]
                )

                # Target world-model predictions with trainable LoRA parameters.
                with torch.cuda.amp.autocast(dtype=weight_dtype), torch.cuda.device(device=accelerator.device):
                    pred_theta = transformer3d(
                        x=noisy_latents, context=prompt_embeds, t=model_timesteps,
                        seq_len=seq_len, y=None, clip_fea=None, viewmats=viewmats, Ks=Ks)

                # Frozen reference world-model predictions.
                with torch.no_grad(), torch.cuda.amp.autocast(dtype=weight_dtype):
                    pred_ref = transformer_ref(
                        x=noisy_latents, context=prompt_embeds, t=model_timesteps,
                        seq_len=seq_len, y=None, clip_fea=None, viewmats=viewmats, Ks=Ks)

                # Offline diffusion-GRPO surrogate, evaluated on future frames only.
                theta_mse = future_only_per_sample_mean(
                    (pred_theta.float() - target.float()).pow(2), future_mask
                ).view(batch_size, group_size)
                ref_mse = future_only_per_sample_mean(
                    (pred_ref.float() - target.float()).pow(2), future_mask
                ).view(batch_size, group_size)
                rewards = batch["rewards"].to(device=theta_mse.device, dtype=theta_mse.dtype)
                advantages = normalize_group_advantages(rewards)
                loss, grpo_stats = diffusion_grpo_loss(
                    theta_mse,
                    ref_mse,
                    advantages,
                    beta=args.grpo_beta,
                    clip_range=args.grpo_clip_range,
                    kl_coeff=args.grpo_kl_coeff,
                )
                step_metrics = {
                    "reward_mean": rewards.detach().mean(),
                    "reward_std": rewards.detach().std(unbiased=False),
                    "advantage_std": advantages.detach().std(unbiased=False),
                    **grpo_stats,
                }

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                train_metrics = {"train_loss": train_loss}
                train_metrics.update({
                    key: accelerator.gather(value.reshape(1)).mean().item()
                    for key, value in step_metrics.items()
                })
                accelerator.log(train_metrics, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if args.use_deepspeed or args.use_fsdp or accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)
                        gc.collect()
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                        if not args.save_state:
                            if args.use_peft_lora:
                                safetensor_save_path = os.path.join(args.output_dir,
                                                                    f"checkpoint-{global_step}.safetensors")
                                network_state_dict = get_peft_model_state_dict(accelerator.unwrap_model(transformer3d))
                                save_model(safetensor_save_path, network_state_dict)

                                safetensor_kohya_format_save_path = os.path.join(args.output_dir,
                                                                                 f"checkpoint-{global_step}-compatible_with_comfyui.safetensors")
                                network_state_dict_kohya = convert_peft_lora_to_kohya_lora(network_state_dict)
                                save_model(safetensor_kohya_format_save_path, network_state_dict_kohya)
                                logger.info(f"Saved safetensor to {safetensor_save_path}")
                            else:
                                safetensor_save_path = os.path.join(args.output_dir,
                                                                    f"checkpoint-{global_step}.safetensors")
                                save_model(safetensor_save_path, accelerator.unwrap_model(network))
                                logger.info(f"Saved safetensor to {safetensor_save_path}")
                        else:
                            accelerator_save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                            accelerator.save_state(accelerator_save_path)
                            logger.info(f"Saved state to {accelerator_save_path}")

                    ## lyz:save checkpoints
                    #score_state_dict = {}
                    #for name, param in accelerator.unwrap_model(transformer3d).named_parameters():
                    #    if any(k in name for k in ['quality_embedding', 'quality_projection',
                    #                            'motion_embedding', 'motion_projection']):
                    #        score_state_dict[name] = param.data.cpu()
                    #if len(score_state_dict) > 0:
                    #    from safetensors.torch import save_file
                    #    score_save_path = os.path.join(args.output_dir, f"score_modules-{global_step}.safetensors")
                    #    save_file(score_state_dict, score_save_path)


                if args.validation_prompts is not None and global_step % args.validation_steps == 0:
                    log_validation(
                        vae,
                        text_encoder,
                        tokenizer,
                        clip_image_encoder,
                        transformer3d,
                        network,
                        args,
                        config,
                        accelerator,
                        weight_dtype,
                        global_step,
                    )

            logs = {
                "step_loss": loss.detach().item(),
                "reward": step_metrics["reward_mean"].item(),
                "ratio": step_metrics["ratio"].item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        if args.validation_prompts is not None and epoch % args.validation_epochs == 0:
            log_validation(
                vae,
                text_encoder,
                tokenizer,
                clip_image_encoder,
                transformer3d,
                network,
                args,
                config,
                accelerator,
                weight_dtype,
                global_step,
            )

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if args.use_deepspeed or args.use_fsdp or accelerator.is_main_process:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        if not args.save_state:
            if args.use_peft_lora:
                safetensor_save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}.safetensors")
                network_state_dict = get_peft_model_state_dict(accelerator.unwrap_model(transformer3d))
                save_model(safetensor_save_path, network_state_dict)

                safetensor_kohya_format_save_path = os.path.join(args.output_dir,
                                                                 f"checkpoint-{global_step}-compatible_with_comfyui.safetensors")
                network_state_dict_kohya = convert_peft_lora_to_kohya_lora(network_state_dict)
                save_model(safetensor_kohya_format_save_path, network_state_dict_kohya)
                logger.info(f"Saved safetensor to {safetensor_save_path}")
            else:
                safetensor_save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}.safetensors")
                save_model(safetensor_save_path, accelerator.unwrap_model(network))
                logger.info(f"Saved safetensor to {safetensor_save_path}")
        else:
            accelerator_save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
            accelerator.save_state(accelerator_save_path)
            logger.info(f"Saved state to {accelerator_save_path}")

        ## lyz:save checkpoints
        #score_state_dict = {}
        #for name, param in accelerator.unwrap_model(transformer3d).named_parameters():
        #    if any(k in name for k in ['quality_embedding', 'quality_projection',
        #                               'motion_embedding', 'motion_projection']):
        #        score_state_dict[name] = param.data.cpu()
        #if len(score_state_dict) > 0:
        #    from safetensors.torch import save_file
        #    score_save_path = os.path.join(args.output_dir, f"score_modules-{global_step}.safetensors")
        #    save_file(score_state_dict, score_save_path)

    accelerator.end_training()


if __name__ == "__main__":
    main()
