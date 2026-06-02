from wan_utils.lmdb_ import get_array_shape_from_lmdb, retrieve_row_from_lmdb
from torch.utils.data import Dataset
import numpy as np
import torch
import lmdb
import json
from pathlib import Path
from PIL import Image
import os
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Rotation

class CameraLatentLMDBDataset(LatentLMDBDataset):
    """LatentLMDBDataset extended with per-frame camera data for PRoPE.

    Expects the LMDB to contain raw camera parameters:
      - ``intrinsics``: float32 array of shape ``(N, 4)`` — [fx, fy, cx, cy] normalized
      - ``poses``:      float32 array of shape ``(N, F, 7)`` — [tx,ty,tz, qx,qy,qz,qw] w2c

    viewmats (F, 4, 4) and Ks (F, 3, 3) are built on-the-fly via
    ``build_viewmats_and_Ks()``, which normalizes poses to the first frame.

    ``data_path`` can be either:
      - a single LMDB directory (has ``data.mdb`` inside), or
      - a parent directory containing multiple LMDB subdirectories (sharding).
    """

    def __init__(self, data_path: str, max_pair: int = int(1e8)):
        # Detect sharding: if data_path contains data.mdb, it's a single LMDB;
        # otherwise treat each subdirectory as a shard.
        if os.path.isfile(os.path.join(data_path, "data.mdb")):
            self._sharded = False
            super().__init__(data_path, max_pair)
            self.intrinsics_shape = get_array_shape_from_lmdb(
                self.env, 'intrinsics')
            self.poses_shape = get_array_shape_from_lmdb(self.env, 'poses')
        else:
            self._sharded = True
            self.envs = []
            self.index = []  # list of (shard_id, local_idx)
            self._latents_shapes = []
            self._intrinsics_shapes = []
            self._poses_shapes = []
            for fname in sorted(os.listdir(data_path)):
                sub = os.path.join(data_path, fname)
                if not os.path.isdir(sub):
                    continue
                if not os.path.isfile(os.path.join(sub, "data.mdb")):
                    continue
                env = lmdb.open(sub, readonly=True, lock=False,
                                readahead=False, meminit=False)
                sid = len(self.envs)
                self.envs.append(env)
                ls = get_array_shape_from_lmdb(env, 'latents')
                self._latents_shapes.append(ls)
                self._intrinsics_shapes.append(
                    get_array_shape_from_lmdb(env, 'intrinsics'))
                self._poses_shapes.append(
                    get_array_shape_from_lmdb(env, 'poses'))
                for j in range(ls[0]):
                    self.index.append((sid, j))
            self.max_pair = max_pair

    def __len__(self):
        if self._sharded:
            return min(len(self.index), self.max_pair)
        return super().__len__()

    def __getitem__(self, idx):
        if self._sharded:
            sid, local_idx = self.index[idx]
            env = self.envs[sid]
            ls = self._latents_shapes[sid]
            latents = retrieve_row_from_lmdb(
                env, "latents", np.float16, local_idx, shape=ls[1:])
            if len(latents.shape) == 4:
                latents = latents[None, ...]
            prompts = retrieve_row_from_lmdb(env, "prompts", str, local_idx)
            intrinsics = retrieve_row_from_lmdb(
                env, "intrinsics", np.float32, local_idx,
                shape=self._intrinsics_shapes[sid][1:])
            poses = retrieve_row_from_lmdb(
                env, "poses", np.float32, local_idx,
                shape=self._poses_shapes[sid][1:])
        else:
            # Single LMDB path — original behavior
            latents = retrieve_row_from_lmdb(
                self.env, "latents", np.float16, idx,
                shape=self.latents_shape[1:])
            if len(latents.shape) == 4:
                latents = latents[None, ...]
            prompts = retrieve_row_from_lmdb(self.env, "prompts", str, idx)
            intrinsics = retrieve_row_from_lmdb(
                self.env, "intrinsics", np.float32, idx,
                shape=self.intrinsics_shape[1:])
            poses = retrieve_row_from_lmdb(
                self.env, "poses", np.float32, idx,
                shape=self.poses_shape[1:])

        viewmats, Ks = build_viewmats_and_Ks(intrinsics, poses)
        return {
            "prompts": prompts,
            "clean_latent": torch.tensor(latents, dtype=torch.float32)[-1],
            "viewmats": torch.tensor(viewmats, dtype=torch.float32),
            "Ks": torch.tensor(Ks, dtype=torch.float32),
        }


def build_viewmats_and_Ks(intrinsics, poses):
    """Build 4x4 w2c view matrices and 3x3 intrinsics from raw poses.

    Called at dataset load time (in ``CameraLatentLMDBDataset.__getitem__``).

    Args:
        intrinsics: (4,) ndarray [fx, fy, cx, cy] (normalized)
        poses:      (T, 7) ndarray [tx, ty, tz, qx, qy, qz, qw] w2c OpenCV

    Returns:
        viewmats: (T, 4, 4) float32 — w2c SE3, normalized to first frame
        Ks:       (T, 3, 3) float32 — intrinsics
    """
    T = len(poses)
    fx, fy, cx, cy = intrinsics

    viewmats = np.zeros((T, 4, 4), dtype=np.float32)
    for i in range(T):
        tx, ty, tz, qx, qy, qz, qw = poses[i]
        R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        viewmats[i, :3, :3] = R
        viewmats[i, :3, 3] = [tx, ty, tz]
        viewmats[i, 3, 3] = 1.0

    # Normalize: align all poses to first frame
    c2w = np.linalg.inv(viewmats)
    C0_inv = np.linalg.inv(c2w[0])
    c2w_aligned = np.array([C0_inv @ C for C in c2w])
    viewmats = np.linalg.inv(c2w_aligned).astype(np.float32)

    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]], dtype=np.float32)
    Ks = np.tile(K, (T, 1, 1))

    return viewmats, Ks


def cycle(dl):
    while True:
        for data in dl:
            yield data
