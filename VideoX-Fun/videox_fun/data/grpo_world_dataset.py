import csv
import os
import random
from typing import Dict, List

import decord
import numpy as np
import torch

from .world_model_utils import load_world_model_camera_matrices, resolve_pose_path


class OfflineGRPOGroupDataset(torch.utils.data.Dataset):
    """Load offline rollout groups with rewards and an optional shared world state."""

    def __init__(
        self,
        csv_path: str,
        video_dir: str,
        *,
        group_size: int,
        video_sample_n_frames: int,
        video_sample_stride: int,
        reward_vq_column: str,
        reward_motion_column: str,
        reward_vq_weight: float,
        reward_motion_weight: float,
        enable_world_model: bool,
        history_video_column: str,
        camera_pose_column: str,
    ):
        if group_size < 2:
            raise ValueError("GRPO requires --grpo_group_size >= 2.")
        self.video_dir = video_dir
        self.group_size = group_size
        self.video_sample_n_frames = video_sample_n_frames
        self.video_sample_stride = video_sample_stride
        self.reward_vq_column = reward_vq_column
        self.reward_motion_column = reward_motion_column
        self.reward_vq_weight = reward_vq_weight
        self.reward_motion_weight = reward_motion_weight
        self.enable_world_model = enable_world_model
        self.history_video_column = history_video_column
        self.camera_pose_column = camera_pose_column
        if reward_vq_weight + reward_motion_weight <= 0:
            raise ValueError("At least one GRPO reward weight must be positive.")

        grouped_rows: Dict[str, List[dict]] = {}
        with open(csv_path, "r", newline="") as csv_file:
            for row in csv.DictReader(csv_file):
                grouped_rows.setdefault(row["original_video"], []).append(row)
        if not grouped_rows:
            raise ValueError("GRPO metadata CSV contains no rollout candidates.")

        all_vq = [float(row[reward_vq_column]) for rows in grouped_rows.values() for row in rows]
        all_motion = [float(row[reward_motion_column]) for rows in grouped_rows.values() for row in rows]
        self.vq_min, self.vq_max = min(all_vq), max(all_vq)
        self.motion_min, self.motion_max = min(all_motion), max(all_motion)

        self.groups = []
        for original_video, rows in grouped_rows.items():
            if len(rows) < group_size:
                continue
            prompts = {row.get("prompt") for row in rows}
            if None in prompts or len(prompts) != 1:
                raise ValueError(f"Group '{original_video}' must share one prompt.")
            if enable_world_model:
                pose_paths = {row.get(camera_pose_column) for row in rows}
                history_paths = {row.get(history_video_column) or original_video for row in rows}
                if None in pose_paths or "" in pose_paths or len(pose_paths) != 1:
                    raise ValueError(f"Group '{original_video}' must share one '{camera_pose_column}'.")
                if len(history_paths) != 1:
                    raise ValueError(f"Group '{original_video}' must share one '{history_video_column}'.")
                pose_path = resolve_pose_path(rows[0], video_dir, camera_pose_column)
                if not os.path.isfile(pose_path):
                    raise FileNotFoundError(f"Camera pose file not found: {pose_path}")
            self.groups.append({"original_video": original_video, "rows": rows})

        if not self.groups:
            raise ValueError("No GRPO groups contain enough rollout candidates.")
        print(f"OfflineGRPOGroupDataset: {len(self.groups)} groups, group_size={group_size}.")

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        group = self.groups[idx]
        rows = random.sample(group["rows"], self.group_size)
        candidates = [self._load_video(row["video_filename"]) for row in rows]
        rewards = torch.tensor([self._score(row) for row in rows], dtype=torch.float32)
        sample = {
            "group_pixel_values": candidates,
            "rewards": rewards,
            "text": rows[0]["prompt"],
        }
        if self.enable_world_model:
            history_filename = rows[0].get(self.history_video_column) or group["original_video"]
            history_frames, frame_indices, source_video_length = self._load_video(history_filename, return_indices=True)
            pose_path = resolve_pose_path(rows[0], self.video_dir, self.camera_pose_column)
            viewmats, Ks = load_world_model_camera_matrices(
                pose_path,
                frame_indices=frame_indices,
                source_video_length=source_video_length,
            )
            sample.update({
                "history_pixel_values": history_frames,
                "viewmats": viewmats,
                "Ks": Ks,
            })
        return sample

    def _score(self, row: dict) -> float:
        vq = (float(row[self.reward_vq_column]) - self.vq_min) / (self.vq_max - self.vq_min + 1e-8)
        motion = (float(row[self.reward_motion_column]) - self.motion_min) / (
            self.motion_max - self.motion_min + 1e-8
        )
        total_weight = self.reward_vq_weight + self.reward_motion_weight
        return (self.reward_vq_weight * vq + self.reward_motion_weight * motion) / total_weight

    def _load_video(self, filename: str, return_indices: bool = False):
        path = filename if os.path.isabs(filename) else os.path.join(self.video_dir, filename)
        video_reader = decord.VideoReader(path)
        total = len(video_reader)
        indices = list(
            range(0, min(total, self.video_sample_n_frames * self.video_sample_stride), self.video_sample_stride)
        )[: self.video_sample_n_frames]
        if len(indices) < min(self.video_sample_n_frames, total):
            indices = list(range(min(self.video_sample_n_frames, total)))
        frames = video_reader.get_batch(indices).asnumpy()
        if return_indices:
            return frames, indices, total
        return frames
