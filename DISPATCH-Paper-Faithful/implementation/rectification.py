"""Official rectification: I_hat = A * I_tilde + (1-A) * I at 512, then resize."""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToPILImage


def rectify(adv_mask: torch.Tensor, generated_01: torch.Tensor, input_01: torch.Tensor) -> torch.Tensor:
    """A, I_tilde, I are at processing resolution. Values in [0,1]."""
    if adv_mask.ndim == 2:
        a = adv_mask
    else:
        a = adv_mask.squeeze()
    return a * generated_01 + (1.0 - a) * input_01


def to_pil_rgb(t_chw_01: torch.Tensor) -> Image.Image:
    return ToPILImage()(t_chw_01.detach().cpu().clamp(0, 1))


def resize_to_original(img: Image.Image, ori_size: tuple[int, int]) -> Image.Image:
    """Official uses Image.resize(ori_size) — PIL default BICUBIC."""
    return img.resize(ori_size)


def mask_to_uint8_orig(adv_mask: torch.Tensor, ori_size: tuple[int, int]) -> np.ndarray:
    """Nearest upsample for evaluation IoU (current project protocol)."""
    m = (adv_mask.detach().cpu().numpy() > 0.5).astype(np.uint8) * 255
    return np.array(Image.fromarray(m, mode="L").resize(ori_size, Image.Resampling.NEAREST))
