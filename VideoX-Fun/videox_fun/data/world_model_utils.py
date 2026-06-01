import os
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from .utils import Camera, get_relative_pose


def load_world_model_camera_matrices(
    pose_file_path: str,
    frame_indices: Optional[Sequence[int]] = None,
    source_video_length: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load CameraCtrl poses as first-frame-relative w2c matrices and normalized intrinsics."""
    with open(pose_file_path, "r") as pose_file:
        rows = [line.strip().split() for line in pose_file.readlines()[1:] if line.strip()]
    if not rows:
        raise ValueError(f"No camera poses found in {pose_file_path}.")

    cameras = [Camera([float(value) for value in row]) for row in rows]
    relative_c2ws = get_relative_pose(cameras)
    viewmats = np.linalg.inv(relative_c2ws).astype(np.float32)
    Ks = np.asarray(
        [[[camera.fx, 0.0, camera.cx], [0.0, camera.fy, camera.cy], [0.0, 0.0, 1.0]] for camera in cameras],
        dtype=np.float32,
    )

    if frame_indices is not None:
        frame_indices = np.asarray(frame_indices)
        if source_video_length is not None and source_video_length != len(cameras):
            frame_indices = np.rint(frame_indices * (len(cameras) - 1) / max(source_video_length - 1, 1))
        frame_indices = np.clip(frame_indices, 0, len(cameras) - 1).astype(np.int64)
        viewmats = viewmats[frame_indices]
        Ks = Ks[frame_indices]

    return torch.from_numpy(viewmats), torch.from_numpy(Ks)


def resolve_pose_path(data_info: Dict[str, str], data_root: Optional[str], pose_column: str) -> str:
    pose_path = data_info.get(pose_column)
    if not pose_path:
        raise ValueError(f"World-model sample is missing camera pose column '{pose_column}'.")
    if data_root is not None and not os.path.isabs(pose_path):
        pose_path = os.path.join(data_root, pose_path)
    return pose_path


def prepare_world_model_training_inputs(
    *,
    latents: torch.Tensor,
    noise: torch.Tensor,
    sigmas: torch.Tensor,
    timesteps: torch.Tensor,
    viewmats: torch.Tensor,
    Ks: torch.Tensor,
    patch_size: Sequence[int],
    history_frames: int,
    future_frames: int,
) -> Dict[str, torch.Tensor]:
    """Build a teacher-forcing window with clean history and noisy future latent frames."""
    viewmats = repeat_batch_to_size(viewmats, latents.shape[0])
    Ks = repeat_batch_to_size(Ks, latents.shape[0])
    latent_frames = latents.shape[2]
    if latent_frames < 2:
        raise ValueError("World-model training requires at least two latent frames.")

    history_frames = min(history_frames, latent_frames - 1)
    available_future = latent_frames - history_frames
    future_frames = available_future if future_frames <= 0 else min(future_frames, available_future)
    window_frames = history_frames + future_frames
    max_start = latent_frames - window_frames
    start = 0 if max_start == 0 else int(torch.randint(0, max_start + 1, (1,), device=latents.device).item())
    end = start + window_frames

    latents = latents[:, :, start:end]
    noise = noise[:, :, start:end]
    latent_viewmats, latent_Ks = _sample_camera_matrices_to_latents(viewmats, Ks, latent_frames)
    latent_viewmats = latent_viewmats[:, start:end].to(device=latents.device, dtype=torch.float32)
    latent_Ks = latent_Ks[:, start:end].to(device=latents.device, dtype=torch.float32)

    noisy_latents = (1.0 - sigmas) * latents + sigmas * noise
    noisy_latents[:, :, :history_frames] = latents[:, :, :history_frames]
    target = noise - latents

    future_mask = torch.zeros_like(latents, dtype=torch.float32)
    future_mask[:, :, history_frames:] = 1.0

    tokens_per_frame = (
        int(np.ceil(latents.shape[3] / patch_size[1]))
        * int(np.ceil(latents.shape[4] / patch_size[2]))
    )
    model_timesteps = timesteps[:, None].expand(-1, window_frames * tokens_per_frame).clone()
    model_timesteps[:, : history_frames * tokens_per_frame] = 0
    return {
        "noisy_latents": noisy_latents,
        "target": target,
        "viewmats": latent_viewmats,
        "Ks": latent_Ks,
        "future_mask": future_mask,
        "timesteps": model_timesteps,
        "history_frames": history_frames,
    }


def future_only_mean(values: torch.Tensor, future_mask: torch.Tensor) -> torch.Tensor:
    mask = future_mask.expand_as(values).to(values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def future_only_per_sample_mean(values: torch.Tensor, future_mask: torch.Tensor) -> torch.Tensor:
    mask = future_mask.expand_as(values).to(values.dtype)
    return (values * mask).flatten(1).sum(dim=1) / mask.flatten(1).sum(dim=1).clamp_min(1.0)


def repeat_batch_to_size(values: torch.Tensor, batch_size: int) -> torch.Tensor:
    if values.shape[0] == batch_size:
        return values
    if batch_size % values.shape[0] != 0:
        raise ValueError(f"Cannot repeat batch of size {values.shape[0]} to {batch_size}.")
    return values.repeat((batch_size // values.shape[0],) + (1,) * (values.ndim - 1))


def _sample_camera_matrices_to_latents(
    viewmats: torch.Tensor,
    Ks: torch.Tensor,
    latent_frames: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    indices = torch.linspace(0, viewmats.shape[1] - 1, latent_frames, device=viewmats.device).round().long()
    return viewmats.index_select(1, indices), Ks.index_select(1, indices)
