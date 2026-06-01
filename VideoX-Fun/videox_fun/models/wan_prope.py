from functools import partial
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn


def prope_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    viewmats: torch.Tensor,
    Ks: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]:
    """Apply projective positional encoding to self-attention Q/K/V tensors."""
    batch, _, seqlen, head_dim = q.shape
    cameras = viewmats.shape[1]
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("PRoPE requires self-attention Q/K/V tensors with matching shapes.")
    if viewmats.shape != (batch, cameras, 4, 4):
        raise ValueError(f"Expected viewmats [B, L, 4, 4], got {tuple(viewmats.shape)}.")
    if Ks is not None and Ks.shape != (batch, cameras, 3, 3):
        raise ValueError(f"Expected Ks [B, L, 3, 3], got {tuple(Ks.shape)}.")
    if cameras != seqlen:
        raise ValueError(f"Expected one camera matrix per token, got {cameras} matrices for {seqlen} tokens.")

    apply_fn_q, apply_fn_kv, apply_fn_o = _prepare_apply_fns_all_dim(
        head_dim=head_dim,
        viewmats=viewmats,
        Ks=Ks,
    )
    return apply_fn_q(q), apply_fn_kv(k), apply_fn_kv(v), apply_fn_o


def _prepare_apply_fns_all_dim(
    *,
    head_dim: int,
    viewmats: torch.Tensor,
    Ks: Optional[torch.Tensor],
) -> Tuple[
    Callable[[torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor], torch.Tensor],
]:
    if Ks is not None:
        Ks_norm = torch.zeros_like(Ks)
        Ks_norm[..., 0, 0] = Ks[..., 0, 0]
        Ks_norm[..., 1, 1] = Ks[..., 1, 1]
        Ks_norm[..., 2, 2] = 1.0

        P = torch.einsum("...ij,...jk->...ik", _lift_K(Ks_norm), viewmats)
        P_T = P.transpose(-1, -2)
        P_inv = torch.einsum(
            "...ij,...jk->...ik",
            _invert_SE3(viewmats),
            _lift_K(_invert_K(Ks_norm)),
        )
    else:
        P = viewmats
        P_T = P.transpose(-1, -2)
        P_inv = _invert_SE3(viewmats)

    if head_dim % 4 != 0:
        raise ValueError(f"PRoPE requires head_dim divisible by 4, got {head_dim}.")

    transforms_q = [(partial(_apply_tiled_projmat, matrix=P_T), head_dim)]
    transforms_kv = [(partial(_apply_tiled_projmat, matrix=P_inv), head_dim)]
    transforms_o = [(partial(_apply_tiled_projmat, matrix=P), head_dim)]
    return (
        partial(_apply_block_diagonal, func_size_pairs=transforms_q),
        partial(_apply_block_diagonal, func_size_pairs=transforms_kv),
        partial(_apply_block_diagonal, func_size_pairs=transforms_o),
    )


def _apply_tiled_projmat(feats: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    batch, num_heads, seqlen, feat_dim = feats.shape
    cameras = matrix.shape[1]
    dim = matrix.shape[-1]
    if seqlen % cameras != 0 or feat_dim % dim != 0:
        raise ValueError("PRoPE matrix and feature shapes are incompatible.")

    out = torch.einsum(
        "bcij,bncpkj->bncpki",
        matrix.float(),
        feats.float().reshape(batch, num_heads, cameras, -1, feat_dim // dim, dim),
    )
    return out.reshape(feats.shape).to(feats.dtype)


def _apply_block_diagonal(
    feats: torch.Tensor,
    func_size_pairs: List[Tuple[Callable[[torch.Tensor], torch.Tensor], int]],
) -> torch.Tensor:
    funcs, block_sizes = zip(*func_size_pairs)
    if feats.shape[-1] != sum(block_sizes):
        raise ValueError("PRoPE block sizes do not match the feature dimension.")
    return torch.cat([func(x) for func, x in zip(funcs, torch.split(feats, block_sizes, dim=-1))], dim=-1)


def _invert_SE3(transforms: torch.Tensor) -> torch.Tensor:
    Rinv = transforms[..., :3, :3].transpose(-1, -2)
    out = torch.zeros_like(transforms)
    out[..., :3, :3] = Rinv
    out[..., :3, 3] = -torch.einsum("...ij,...j->...i", Rinv, transforms[..., :3, 3])
    out[..., 3, 3] = 1.0
    return out


def _lift_K(Ks: torch.Tensor) -> torch.Tensor:
    out = torch.zeros(Ks.shape[:-2] + (4, 4), device=Ks.device, dtype=Ks.dtype)
    out[..., :3, :3] = Ks
    out[..., 3, 3] = 1.0
    return out


def _invert_K(Ks: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(Ks)
    out[..., 0, 0] = 1.0 / Ks[..., 0, 0]
    out[..., 1, 1] = 1.0 / Ks[..., 1, 1]
    out[..., 0, 2] = -Ks[..., 0, 2] / Ks[..., 0, 0]
    out[..., 1, 2] = -Ks[..., 1, 2] / Ks[..., 1, 1]
    out[..., 2, 2] = 1.0
    return out


def add_prope_parameters(model: nn.Module, zero_init: bool = True) -> None:
    """Attach a zero-initialized PRoPE residual projection to every self-attention block."""
    for block in model.blocks:
        attn = block.self_attn
        if hasattr(attn, "prope_o"):
            continue
        prope_o = nn.Linear(attn.o.out_features, attn.o.out_features)
        if zero_init:
            nn.init.zeros_(prope_o.weight)
            nn.init.zeros_(prope_o.bias)
        attn.prope_o = prope_o.to(device=attn.o.weight.device, dtype=attn.o.weight.dtype)


def expand_camera_matrices(
    viewmats: torch.Tensor,
    Ks: torch.Tensor,
    grid_sizes: torch.Tensor,
    seq_len: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Expand per-latent-frame matrices to the patch-token sequence."""
    expanded_viewmats = []
    expanded_Ks = []
    for sample_viewmats, sample_Ks, grid_size in zip(viewmats, Ks, grid_sizes.tolist()):
        frames, height, width = grid_size
        if sample_viewmats.shape[0] != frames:
            indices = torch.linspace(
                0, sample_viewmats.shape[0] - 1, frames, device=sample_viewmats.device
            ).round().long()
            sample_viewmats = sample_viewmats.index_select(0, indices)
            sample_Ks = sample_Ks.index_select(0, indices)
        sample_viewmats = sample_viewmats[:, None, None].expand(frames, height, width, 4, 4).reshape(-1, 4, 4)
        sample_Ks = sample_Ks[:, None, None].expand(frames, height, width, 3, 3).reshape(-1, 3, 3)
        if sample_viewmats.shape[0] < seq_len:
            pad = seq_len - sample_viewmats.shape[0]
            sample_viewmats = torch.cat([sample_viewmats, sample_viewmats[-1:].expand(pad, -1, -1)])
            sample_Ks = torch.cat([sample_Ks, sample_Ks[-1:].expand(pad, -1, -1)])
        expanded_viewmats.append(sample_viewmats)
        expanded_Ks.append(sample_Ks)
    return torch.stack(expanded_viewmats), torch.stack(expanded_Ks)
