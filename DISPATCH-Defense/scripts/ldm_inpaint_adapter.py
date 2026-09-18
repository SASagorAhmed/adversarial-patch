#!/usr/bin/env python
"""Adapter around CompVis LDM inpainting (no text prompts; DDIM)."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.dispatch_config import (
    DIFFUSION_STEPS,
    DISPATCH_SEED,
    RESOLUTION,
    seed_everything,
)
from config.paths_config import LDM_INPAINT_CKPT, LDM_INPAINT_CONFIG, LDM_ROOT, assert_under_project_root

# Ensure CompVis package imports (main.instantiate_from_config lives in LDM repo root)
# Also add taming-transformers for VQ modules used by inpainting_big.
_TAMING = ROOT / "third_party" / "taming-transformers"
for _p in (LDM_ROOT, _TAMING):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


class LDMInpaintAdapter:
    """Load official CompVis inpainting_big once; inpaint RGB+mask arrays."""

    def __init__(
        self,
        config_path: Path | str | None = None,
        ckpt_path: Path | str | None = None,
        device: Optional[str] = None,
        steps: int = DIFFUSION_STEPS,
        seed: int = DISPATCH_SEED,
    ) -> None:
        self.config_path = Path(config_path or LDM_INPAINT_CONFIG)
        self.ckpt_path = Path(ckpt_path or LDM_INPAINT_CKPT)
        self.steps = int(steps)
        self.seed = int(seed)
        if not self.config_path.is_file():
            raise FileNotFoundError(f"Missing LDM config: {self.config_path}")
        if not self.ckpt_path.is_file():
            raise FileNotFoundError(f"Missing LDM checkpoint: {self.ckpt_path}")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        seed_everything(self.seed)

        from main import instantiate_from_config
        from ldm.models.diffusion.ddim import DDIMSampler

        config = OmegaConf.load(str(self.config_path))
        self.model = instantiate_from_config(config.model)
        sd = torch.load(str(self.ckpt_path), map_location="cpu")
        if "state_dict" not in sd:
            raise RuntimeError("Checkpoint missing 'state_dict' key; refusing silent fallback")
        self.model.load_state_dict(sd["state_dict"], strict=False)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.sampler = DDIMSampler(self.model)
        self._DDIMSampler = DDIMSampler

    @staticmethod
    def _prepare_batch(
        image_rgb_uint8: np.ndarray,
        mask_hw: np.ndarray,
        device: torch.device,
    ) -> dict:
        if image_rgb_uint8.ndim != 3 or image_rgb_uint8.shape[2] != 3:
            raise ValueError(f"image must be HxWx3 uint8, got {image_rgb_uint8.shape}")
        if mask_hw.shape[:2] != image_rgb_uint8.shape[:2]:
            raise ValueError("mask/image spatial mismatch")

        image = image_rgb_uint8.astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image_t = torch.from_numpy(image)

        mask = mask_hw.astype(np.float32)
        if mask.max() > 1.0:
            mask = mask / 255.0
        mask = mask[None, None]
        mask = np.where(mask < 0.5, 0.0, 1.0).astype(np.float32)
        mask_t = torch.from_numpy(mask)

        masked_image = (1.0 - mask_t) * image_t
        batch = {"image": image_t, "mask": mask_t, "masked_image": masked_image}
        for k in batch:
            batch[k] = batch[k].to(device=device)
            batch[k] = batch[k] * 2.0 - 1.0
        return batch

    @torch.no_grad()
    def inpaint(
        self,
        image_rgb_uint8: np.ndarray,
        mask_hw: np.ndarray,
        steps: Optional[int] = None,
    ) -> np.ndarray:
        """Inpaint masked regions. mask==1 means regenerate (CompVis convention).

        Returns uint8 HxWx3 RGB at the same spatial size as the input image.
        """
        steps = self.steps if steps is None else int(steps)
        seed_everything(self.seed)
        batch = self._prepare_batch(image_rgb_uint8, mask_hw, self.device)

        with self.model.ema_scope():
            c = self.model.cond_stage_model.encode(batch["masked_image"])
            cc = torch.nn.functional.interpolate(batch["mask"], size=c.shape[-2:])
            c = torch.cat((c, cc), dim=1)
            shape = (c.shape[1] - 1,) + c.shape[2:]
            samples_ddim, _ = self.sampler.sample(
                S=steps,
                conditioning=c,
                batch_size=c.shape[0],
                shape=shape,
                verbose=False,
            )
            x_samples_ddim = self.model.decode_first_stage(samples_ddim)

            image = torch.clamp((batch["image"] + 1.0) / 2.0, min=0.0, max=1.0)
            mask = torch.clamp((batch["mask"] + 1.0) / 2.0, min=0.0, max=1.0)
            predicted_image = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
            inpainted = (1 - mask) * image + mask * predicted_image
            out = inpainted.cpu().numpy().transpose(0, 2, 3, 1)[0] * 255.0
            if not np.isfinite(out).all():
                raise RuntimeError("LDM inpainting produced NaN/Inf — STOP (no silent fallback)")
            return out.astype(np.uint8)

    def inpaint_pil(
        self,
        image: Image.Image,
        mask: Image.Image,
        steps: Optional[int] = None,
    ) -> Image.Image:
        rgb = np.array(image.convert("RGB"))
        m = np.array(mask.convert("L"))
        out = self.inpaint(rgb, m, steps=steps)
        return Image.fromarray(out, mode="RGB")


def resize_rgb(image: Image.Image, size: int = RESOLUTION) -> Image.Image:
    return image.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)


def save_rgb(arr_or_img, path: Path) -> None:
    path = assert_under_project_root(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(arr_or_img, np.ndarray):
        Image.fromarray(arr_or_img.astype(np.uint8), mode="RGB").save(path)
    else:
        arr_or_img.save(path)
