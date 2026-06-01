from typing import Tuple

import torch


def normalize_group_advantages(rewards: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalize rewards independently inside each rollout group."""
    if rewards.ndim != 2:
        raise ValueError(f"Expected rewards [batch, group], got {tuple(rewards.shape)}.")
    centered = rewards - rewards.mean(dim=1, keepdim=True)
    scale = rewards.std(dim=1, keepdim=True, unbiased=False)
    return centered / scale.clamp_min(eps)


def diffusion_grpo_loss(
    theta_mse: torch.Tensor,
    ref_mse: torch.Tensor,
    advantages: torch.Tensor,
    *,
    beta: float,
    clip_range: float,
    kl_coeff: float,
) -> Tuple[torch.Tensor, dict]:
    """Compute a GRPO surrogate using diffusion denoising error as a log-probability proxy.

    The offline candidates are assumed to come from the reference policy. Therefore
    `-beta * (theta_mse - ref_mse)` acts as an approximate log policy ratio.
    """
    if theta_mse.shape != ref_mse.shape or theta_mse.shape != advantages.shape:
        raise ValueError("theta_mse, ref_mse, and advantages must have matching [batch, group] shapes.")

    log_ratio = -beta * (theta_mse - ref_mse)
    safe_log_ratio = log_ratio.clamp(min=-20.0, max=20.0)
    ratio = torch.exp(safe_log_ratio)
    clipped_ratio = ratio.clamp(min=1.0 - clip_range, max=1.0 + clip_range)
    surrogate = torch.minimum(ratio * advantages, clipped_ratio * advantages)

    # This non-negative approximation is zero when theta and reference coincide.
    approx_kl = ratio - safe_log_ratio - 1.0
    loss = -surrogate.mean() + kl_coeff * approx_kl.mean()
    stats = {
        "ratio": ratio.detach().mean(),
        "approx_kl": approx_kl.detach().mean(),
        "clip_fraction": (ratio.detach() != clipped_ratio.detach()).float().mean(),
    }
    return loss, stats
