"""Official-source checkerboard masks and complementary composition.

Faithful to MaJinWakeUp/DisPatch LDM/scripts/d3_inference.py
commit 80c59b9b4a6c4060107a8b4485d3885128c2ae81.
"""
from __future__ import annotations

import numpy as np
import torch


def make_checkerboard_masks(size: int = 512, num_grids: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    """Return mask0, mask1 as float tensors (1,1,H,W). 1 = inpaint, 0 = preserve.

    Official uses integer tiles: grid_size = size // num_grids.
    Remainder rows/cols (if any) stay 0 in mask0 and therefore 1 in mask1.
    At 512/32 this covers every pixel with no remainder.
    """
    mask = torch.zeros((size, size), dtype=torch.float32)
    grid_size = size // num_grids
    for i in range(num_grids):
        for j in range(num_grids):
            if (i + j) % 2 == 0:
                mask[i * grid_size : (i + 1) * grid_size, j * grid_size : (j + 1) * grid_size] = 1
    mask0 = mask.unsqueeze(0).unsqueeze(0)
    mask1 = 1.0 - mask0
    return mask0, mask1


def validate_complementary(mask0: torch.Tensor, mask1: torch.Tensor) -> dict:
    m0 = mask0.squeeze().cpu().numpy()
    m1 = mask1.squeeze().cpu().numpy()
    s = m0 + m1
    both_one = int(np.sum((m0 >= 0.5) & (m1 >= 0.5)))
    both_zero = int(np.sum((m0 < 0.5) & (m1 < 0.5)))
    return {
        "shape": list(m0.shape),
        "m0_mean": float(m0.mean()),
        "m1_mean": float(m1.mean()),
        "sum_min": float(s.min()),
        "sum_max": float(s.max()),
        "sum_allclose_1": bool(np.allclose(s, 1.0)),
        "overlap_both_one_pixels": both_one,
        "hole_both_zero_pixels": both_zero,
        "binary_m0": bool(set(np.unique(m0)).issubset({0.0, 1.0})),
        "binary_m1": bool(set(np.unique(m1)).issubset({0.0, 1.0})),
    }


def compose_regenerated(mask0: torch.Tensor, mask1: torch.Tensor, predicted0: torch.Tensor, predicted1: torch.Tensor) -> torch.Tensor:
    """I_tilde = pred0 * m0 + pred1 * m1 (official generated_image line).

    mask* and predicted* are in [0,1], CHW or BCHW squeezed to CHW.
    """
    if mask0.ndim == 4:
        mask0 = mask0[0]
        mask1 = mask1[0]
    if predicted0.ndim == 4:
        predicted0 = predicted0[0]
        predicted1 = predicted1[0]
    return mask0 * predicted0 + mask1 * predicted1
