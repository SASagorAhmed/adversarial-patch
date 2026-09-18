"""Official-source batched two-mask CompVis inpainting (read-only checkpoint)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torchvision.transforms import ToTensor

ROOT = Path(r"D:\project CS\DISPATCH-Paper-Faithful").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.reference_config import (
    LDM_CKPT,
    LDM_CONFIG,
    OFFICIAL_LDM_ROOT,
    RUN_SEED,
    TAMING_ROOT,
)


def _ensure_ldm_imports() -> None:
    for p in (OFFICIAL_LDM_ROOT, TAMING_ROOT):
        s = str(p)
        if p.is_dir() and s not in sys.path:
            sys.path.insert(0, s)


class OfficialLDMAdapter:
    """One DDIM sample() with batch_size=2 for m0 and m1 (official d3_inference.py)."""

    def __init__(self, steps: int = 5, device: str | None = None) -> None:
        _ensure_ldm_imports()
        if not LDM_CONFIG.is_file():
            raise FileNotFoundError(LDM_CONFIG)
        if not LDM_CKPT.is_file():
            raise FileNotFoundError(LDM_CKPT)

        from main import instantiate_from_config
        from ldm.models.diffusion.ddim import DDIMSampler

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.steps = int(steps)

        config = OmegaConf.load(str(LDM_CONFIG))
        self.model = instantiate_from_config(config.model)
        sd = torch.load(str(LDM_CKPT), map_location="cpu")
        if "state_dict" not in sd:
            raise RuntimeError("Checkpoint missing state_dict")
        self.model.load_state_dict(sd["state_dict"], strict=False)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.sampler = DDIMSampler(self.model)

    def make_generation_batch(
        self,
        image_path: Path,
        mask0: torch.Tensor,
        mask1: torch.Tensor,
        size: int = 512,
    ) -> tuple[dict, tuple[int, int], np.ndarray]:
        """Official make_generation_batch: PIL RGB, default resize, ToTensor, batch=2."""
        image = Image.open(image_path).convert("RGB")
        ori_size = image.size
        image = image.resize((size, size))  # PIL default BICUBIC
        rgb_uint8 = np.array(image)
        image_t = ToTensor()(image).unsqueeze(0)
        image_t = image_t.repeat(2, 1, 1, 1)
        masks = torch.cat([mask0.detach().cpu(), mask1.detach().cpu()], dim=0)
        masked_image = (1.0 - masks) * image_t
        batch = {"image": image_t, "mask": masks, "masked_image": masked_image}
        for k in batch:
            batch[k] = batch[k].to(self.device)
            batch[k] = batch[k] * 2.0 - 1.0
        return batch, ori_size, rgb_uint8

    @torch.no_grad()
    def regenerate_pair(
        self,
        image_path: Path,
        mask0: torch.Tensor,
        mask1: torch.Tensor,
        size: int = 512,
        steps: int | None = None,
    ) -> dict:
        steps = self.steps if steps is None else int(steps)
        torch.manual_seed(RUN_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(RUN_SEED)

        batch, ori_size, rgb_uint8 = self.make_generation_batch(image_path, mask0, mask1, size=size)
        with self.model.ema_scope():
            try:
                predicted_image, input_image, mask = self._sample_batch(batch, steps)
                self.last_sample_mode = "official_batch_size_2"
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # Hardware fallback: two sequential samples WITHOUT reseeding so noise differs
                # (official uses one batch_size=2 call). Documented in audit, not a mask-oracle change.
                predicted_image, input_image, mask = self._sample_sequential(batch, steps)
                self.last_sample_mode = "sequential_no_reseed_oom_fallback"

        generated = mask[0] * predicted_image[0] + mask[1] * predicted_image[1]
        return {
            "input_01": input_image[0],
            "generated_01": generated,
            "predicted_0": predicted_image[0],
            "predicted_1": predicted_image[1],
            "mask_01": mask,
            "ori_size": ori_size,
            "proc_rgb_uint8": rgb_uint8,
            "sample_mode": getattr(self, "last_sample_mode", "unknown"),
        }

    def _decode_from_samples(self, batch, samples_ddim):
        x_samples_ddim = self.model.decode_first_stage(samples_ddim)
        input_image = torch.clamp((batch["image"] + 1.0) / 2.0, min=0.0, max=1.0)
        mask = torch.clamp((batch["mask"] + 1.0) / 2.0, min=0.0, max=1.0)
        predicted_image = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
        return predicted_image, input_image, mask

    def _sample_batch(self, batch, steps):
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
        return self._decode_from_samples(batch, samples_ddim)

    def _sample_sequential(self, batch, steps):
        preds = []
        for i in range(2):
            sub = {k: v[i : i + 1] for k, v in batch.items()}
            c = self.model.cond_stage_model.encode(sub["masked_image"])
            cc = torch.nn.functional.interpolate(sub["mask"], size=c.shape[-2:])
            c = torch.cat((c, cc), dim=1)
            shape = (c.shape[1] - 1,) + c.shape[2:]
            samples_ddim, _ = self.sampler.sample(
                S=steps,
                conditioning=c,
                batch_size=1,
                shape=shape,
                verbose=False,
            )
            x = self.model.decode_first_stage(samples_ddim)
            preds.append(torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        predicted_image = torch.cat(preds, dim=0)
        input_image = torch.clamp((batch["image"] + 1.0) / 2.0, min=0.0, max=1.0)
        mask = torch.clamp((batch["mask"] + 1.0) / 2.0, min=0.0, max=1.0)
        return predicted_image, input_image, mask
