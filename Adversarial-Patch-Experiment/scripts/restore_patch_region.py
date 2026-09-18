#!/usr/bin/env python3
"""Stable Diffusion Img2Img worker for known-location patch restoration.

Run with the Stable-Diffusion-Patch virtual environment only.
Does not write into Hyper-YOLO or Stable-Diffusion-Patch project trees.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Restore a known patch region using StableDiffusionImg2ImgPipeline."
    )
    parser.add_argument("--attacked-image", required=True)
    parser.add_argument("--output-restored", required=True)
    parser.add_argument("--output-mask", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--x1", type=int, required=True)
    parser.add_argument("--y1", type=int, required=True)
    parser.add_argument("--x2", type=int, required=True)
    parser.add_argument("--y2", type=int, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--strength", type=float, default=0.55)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--context-factor", type=float, default=3.0)
    parser.add_argument("--min-context-side", type=int, default=128)
    parser.add_argument("--feather-pixels", type=int, default=4)
    parser.add_argument(
        "--neutralize",
        action="store_true",
        help="Neutralize exact patch region in the attacked crop before Img2Img.",
    )
    parser.add_argument(
        "--output-neutralized-512",
        default=None,
        help="Optional path to save the 512x512 Img2Img input.",
    )
    parser.add_argument(
        "--output-raw-img2img",
        default=None,
        help="Optional path to save the raw 512x512 Img2Img output.",
    )
    parser.add_argument("--metadata-json", default=None, help="Optional path to write crop metadata.")
    return parser.parse_args()


def compute_context_crop(
    image_width: int,
    image_height: int,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    context_factor: float,
    min_context_side: int,
) -> tuple[int, int, int, int]:
    patch_w = max(1, x2 - x1)
    patch_h = max(1, y2 - y1)
    patch_cx = (x1 + x2) / 2.0
    patch_cy = (y1 + y2) / 2.0
    side = max(min_context_side, int(round(context_factor * max(patch_w, patch_h))))
    side = min(side, image_width, image_height)

    left = int(round(patch_cx - side / 2.0))
    top = int(round(patch_cy - side / 2.0))
    left = max(0, min(left, image_width - side))
    top = max(0, min(top, image_height - side))
    right = left + side
    bottom = top + side
    return left, top, right, bottom


def feathered_alpha(size: int, feather_pixels: int) -> np.ndarray:
    alpha = np.ones((size, size), dtype=np.float32)
    feather = max(0, int(feather_pixels))
    if feather <= 0 or size < 2:
        return alpha
    for i in range(feather):
        weight = (i + 1) / (feather + 1)
        alpha[i, :] = np.minimum(alpha[i, :], weight)
        alpha[-(i + 1), :] = np.minimum(alpha[-(i + 1), :], weight)
        alpha[:, i] = np.minimum(alpha[:, i], weight)
        alpha[:, -(i + 1)] = np.minimum(alpha[:, -(i + 1)], weight)
    return alpha


def neutralize_patch_in_crop(
    crop_rgb: Image.Image,
    rel_x1: int,
    rel_y1: int,
    rel_x2: int,
    rel_y2: int,
    feather: int = 4,
) -> Image.Image:
    """Replace exact patch square using only surrounding attacked-crop pixels."""
    from PIL import ImageFilter

    crop = np.array(crop_rgb).astype(np.float32)
    blurred = np.array(crop_rgb.filter(ImageFilter.GaussianBlur(radius=18))).astype(np.float32)

    pad = max(4, (rel_x2 - rel_x1) // 6)
    h, w = crop.shape[:2]
    y0 = max(0, rel_y1 - pad)
    y1b = min(h, rel_y2 + pad)
    x0 = max(0, rel_x1 - pad)
    x1b = min(w, rel_x2 + pad)
    ring = crop[y0:y1b, x0:x1b].copy()
    local_mask = np.ones(ring.shape[:2], dtype=bool)
    local_mask[rel_y1 - y0 : rel_y2 - y0, rel_x1 - x0 : rel_x2 - x0] = False
    if local_mask.any():
        mean_color = ring[local_mask].mean(axis=0)
    else:
        mean_color = crop.reshape(-1, 3).mean(axis=0)

    filled = blurred.copy()
    filled[rel_y1:rel_y2, rel_x1:rel_x2] = (
        0.65 * blurred[rel_y1:rel_y2, rel_x1:rel_x2] + 0.35 * mean_color
    )

    patch_h = rel_y2 - rel_y1
    patch_w = rel_x2 - rel_x1
    alpha = feathered_alpha(max(patch_h, patch_w), feather)[:patch_h, :patch_w][:, :, None]
    out = crop.copy()
    region = out[rel_y1:rel_y2, rel_x1:rel_x2]
    out[rel_y1:rel_y2, rel_x1:rel_x2] = (
        alpha * filled[rel_y1:rel_y2, rel_x1:rel_x2] + (1.0 - alpha) * region
    )
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def main() -> None:
    args = parse_args()
    attacked_path = Path(args.attacked_image)
    output_restored = Path(args.output_restored)
    output_mask = Path(args.output_mask)
    model_path = Path(args.model_path)

    if not attacked_path.is_file():
        raise FileNotFoundError(f"Attacked image not found: {attacked_path}")
    if not model_path.is_dir():
        raise FileNotFoundError(f"SD model not found: {model_path}")
    if args.x2 <= args.x1 or args.y2 <= args.y1:
        raise ValueError("Invalid patch coordinates: x2/y2 must be greater than x1/y1")

    image = Image.open(attacked_path).convert("RGB")
    image_width, image_height = image.size
    x1, y1, x2, y2 = args.x1, args.y1, args.x2, args.y2
    if x1 < 0 or y1 < 0 or x2 > image_width or y2 > image_height:
        raise ValueError(
            f"Patch coordinates out of bounds for {attacked_path.name}: "
            f"({x1},{y1})-({x2},{y2}) vs {image_width}x{image_height}"
        )

    # Documentation mask over exact patch square
    mask = Image.new("L", (image_width, image_height), 0)
    mask_arr = np.array(mask)
    mask_arr[y1:y2, x1:x2] = 255
    mask = Image.fromarray(mask_arr)
    output_mask.parent.mkdir(parents=True, exist_ok=True)
    mask.save(output_mask)

    crop_box = compute_context_crop(
        image_width,
        image_height,
        x1,
        y1,
        x2,
        y2,
        args.context_factor,
        args.min_context_side,
    )
    crop_left, crop_top, crop_right, crop_bottom = crop_box
    crop = image.crop(crop_box)
    crop_w = crop_right - crop_left
    crop_h = crop_bottom - crop_top

    # Relative patch location inside crop
    rel_x1 = x1 - crop_left
    rel_y1 = y1 - crop_top
    rel_x2 = x2 - crop_left
    rel_y2 = y2 - crop_top

    if args.neutralize:
        crop = neutralize_patch_in_crop(
            crop, rel_x1, rel_y1, rel_x2, rel_y2, args.feather_pixels
        )

    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    dtype = torch.float16 if use_cuda else torch.float32

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        str(model_path),
        local_files_only=True,
        torch_dtype=dtype,
        safety_checker=None,
    )
    pipe = pipe.to(device)
    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None

    generator = torch.Generator(device=device).manual_seed(args.seed)
    init_512 = crop.resize((args.resolution, args.resolution), Image.Resampling.LANCZOS)
    if args.output_neutralized_512:
        out_n = Path(args.output_neutralized_512)
        out_n.parent.mkdir(parents=True, exist_ok=True)
        init_512.save(out_n)

    result = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        image=init_512,
        strength=args.strength,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.steps,
        generator=generator,
    )
    restored_512 = result.images[0]
    if args.output_raw_img2img:
        out_raw = Path(args.output_raw_img2img)
        out_raw.parent.mkdir(parents=True, exist_ok=True)
        restored_512.save(out_raw)

    restored_crop = restored_512.resize((crop_w, crop_h), Image.Resampling.LANCZOS)

    # Composite only the patch region (with feather) into a full-image copy
    base = np.array(image).astype(np.float32)
    restored_crop_np = np.array(restored_crop).astype(np.float32)
    patch_restored = restored_crop_np[rel_y1:rel_y2, rel_x1:rel_x2]
    patch_h = y2 - y1
    patch_w = x2 - x1
    if patch_restored.shape[0] != patch_h or patch_restored.shape[1] != patch_w:
        raise RuntimeError(
            f"Restored patch shape mismatch: got {patch_restored.shape[:2]}, "
            f"expected {(patch_h, patch_w)}"
        )

    alpha = feathered_alpha(patch_h if patch_h == patch_w else max(patch_h, patch_w), args.feather_pixels)
    # If non-square (should not happen for frozen attack patches), crop alpha to patch size
    alpha = alpha[:patch_h, :patch_w]
    alpha_3 = alpha[:, :, None]
    region = base[y1:y2, x1:x2]
    blended = alpha_3 * patch_restored + (1.0 - alpha_3) * region
    base[y1:y2, x1:x2] = blended
    out = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))

    output_restored.parent.mkdir(parents=True, exist_ok=True)
    out.save(output_restored)

    meta = {
        "attacked_image": str(attacked_path),
        "output_restored": str(output_restored),
        "output_mask": str(output_mask),
        "patch": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "crop_box": {
            "left": crop_left,
            "top": crop_top,
            "right": crop_right,
            "bottom": crop_bottom,
        },
        "device": device,
        "seed": args.seed,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "strength": args.strength,
        "resolution": args.resolution,
        "feather_pixels": args.feather_pixels,
        "neutralize": bool(args.neutralize),
    }
    if args.metadata_json:
        meta_path = Path(args.metadata_json)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(json.dumps({"status": "ok", "restored": str(output_restored), "mask": str(output_mask)}))


if __name__ == "__main__":
    main()
