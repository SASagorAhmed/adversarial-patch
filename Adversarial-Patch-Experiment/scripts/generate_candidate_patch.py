#!/usr/bin/env python3
"""Generate a candidate diffusion patch using local Stable Diffusion 2.1."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = EXPERIMENT_ROOT.parent / "Stable-Diffusion-Patch" / "model"
WIDTH = 512
HEIGHT = 512


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a 512x512 candidate diffusion patch image."
    )
    parser.add_argument("--output", required=True, help="Output PNG path.")
    parser.add_argument("--seed", type=int, required=True, help="Random seed.")
    parser.add_argument("--prompt", required=True, help="Diffusion prompt.")
    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL_PATH),
        help="Local Stable Diffusion model directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    model_path = Path(args.model_path)

    if not model_path.is_dir():
        raise FileNotFoundError(f"Stable Diffusion model not found: {model_path}")

    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    dtype = torch.float16 if use_cuda else torch.float32

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Model path: {model_path}")
    print(f"Prompt: {args.prompt}")
    print(f"Seed: {args.seed}")
    print(f"Device: {device}")
    print(f"Output path: {output_path}")

    pipe = StableDiffusionPipeline.from_pretrained(
        str(model_path),
        local_files_only=True,
        torch_dtype=dtype,
    )
    pipe = pipe.to(device)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    result = pipe(
        args.prompt,
        height=HEIGHT,
        width=WIDTH,
        generator=generator,
    )
    image = result.images[0]
    if not isinstance(image, Image.Image):
        raise RuntimeError("Stable Diffusion did not return a PIL image.")

    image.save(output_path)
    print("Generation completed successfully.")


if __name__ == "__main__":
    main()
