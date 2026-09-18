import argparse
import random
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline

MODEL_PATH = Path("./model")
WIDTH = 512
HEIGHT = 512


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an image with local Stable Diffusion 2.1"
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="Text prompt for image generation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (auto-generated if omitted)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    dtype = torch.float16 if use_cuda else torch.float32
    model_path = MODEL_PATH.resolve()

    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    output_path = Path(f"results/diffusion_{seed}.png")

    pipe = StableDiffusionPipeline.from_pretrained(
        str(model_path),
        local_files_only=True,
        torch_dtype=dtype,
    )
    pipe = pipe.to(device)

    generator = torch.Generator(device=device).manual_seed(seed)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Prompt: {args.prompt}")
    print(f"Seed: {seed}")
    print(f"Device: {device}")
    print(f"Output path: {output_path.resolve()}")

    result = pipe(
        args.prompt,
        height=HEIGHT,
        width=WIDTH,
        generator=generator,
    )
    image = result.images[0]
    image.save(output_path)

    print("Generation completed successfully.")


if __name__ == "__main__":
    main()
