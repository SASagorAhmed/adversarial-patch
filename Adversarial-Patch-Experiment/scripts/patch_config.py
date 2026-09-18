#!/usr/bin/env python3
"""Candidate patch definitions for automatic diffusion patch generation."""

from __future__ import annotations

from dataclasses import dataclass
import re

PATCH_FILENAME_PATTERN = re.compile(r"^diffusion_patch_(\d{2})\.png$")

PATCH_DEFINITIONS: dict[int, dict[str, str | int]] = {
    1: {
        "filename": "diffusion_patch_01.png",
        "seed": 101,
        "prompt": (
            "A colorful abstract square sticker with complex geometric patterns, "
            "high contrast, realistic printed texture, centered composition"
        ),
    },
    2: {
        "filename": "diffusion_patch_02.png",
        "seed": 102,
        "prompt": (
            "A vivid abstract square sticker with sharp zigzag lines, bold colors, "
            "high contrast, realistic printed texture, centered composition"
        ),
    },
    3: {
        "filename": "diffusion_patch_03.png",
        "seed": 103,
        "prompt": (
            "A bright abstract square sticker with concentric circles, saturated hues, "
            "high contrast, realistic printed texture, centered composition"
        ),
    },
    4: {
        "filename": "diffusion_patch_04.png",
        "seed": 104,
        "prompt": (
            "A colorful abstract square sticker with diagonal stripes, strong contrast, "
            "realistic printed texture, centered composition"
        ),
    },
    5: {
        "filename": "diffusion_patch_05.png",
        "seed": 105,
        "prompt": (
            "A bold abstract square sticker with mosaic tiles, vivid colors, "
            "high contrast, realistic printed texture, centered composition"
        ),
    },
    6: {
        "filename": "diffusion_patch_06.png",
        "seed": 106,
        "prompt": (
            "A colorful abstract square sticker with fractal-like shapes, intense contrast, "
            "realistic printed texture, centered composition"
        ),
    },
    7: {
        "filename": "diffusion_patch_07.png",
        "seed": 107,
        "prompt": (
            "A vivid abstract square sticker with checkerboard motifs, saturated palette, "
            "high contrast, realistic printed texture, centered composition"
        ),
    },
    8: {
        "filename": "diffusion_patch_08.png",
        "seed": 108,
        "prompt": (
            "A bright abstract square sticker with overlapping triangles, bold contrast, "
            "realistic printed texture, centered composition"
        ),
    },
    9: {
        "filename": "diffusion_patch_09.png",
        "seed": 109,
        "prompt": (
            "A colorful abstract square sticker with radial bursts, high contrast colors, "
            "realistic printed texture, centered composition"
        ),
    },
    10: {
        "filename": "diffusion_patch_10.png",
        "seed": 110,
        "prompt": (
            "A vivid abstract square sticker with labyrinth patterns, strong contrast, "
            "realistic printed texture, centered composition"
        ),
    },
}


@dataclass(frozen=True)
class PatchSpec:
    number: int
    filename: str
    seed: int
    prompt: str


def patch_spec_for_number(number: int) -> PatchSpec:
    if number not in PATCH_DEFINITIONS:
        raise ValueError(
            f"No automatic patch definition for attack number {number}. "
            "Use --patch for attacks above 10."
        )
    entry = PATCH_DEFINITIONS[number]
    return PatchSpec(
        number=number,
        filename=str(entry["filename"]),
        seed=int(entry["seed"]),
        prompt=str(entry["prompt"]),
    )


def patch_number_from_filename(filename: str) -> int | None:
    match = PATCH_FILENAME_PATTERN.match(filename)
    if not match:
        return None
    return int(match.group(1))


def patch_spec_for_filename(filename: str) -> PatchSpec:
    number = patch_number_from_filename(filename)
    if number is None:
        raise ValueError(
            f"Cannot infer automatic patch settings for filename: {filename}. "
            "Expected format diffusion_patch_NN.png with NN from 01 to 10."
        )
    return patch_spec_for_number(number)  # noqa: RET504 — explicit return for clarity
