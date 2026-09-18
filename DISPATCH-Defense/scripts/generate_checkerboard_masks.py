#!/usr/bin/env python
"""Generate complementary N×N checkerboard masks for DISPATCH regeneration."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.dispatch_config import CHECKERBOARD_GRID_N, RESOLUTION, seed_everything
from config.paths_config import assert_under_project_root


def generate_checkerboard_masks(
    height: int,
    width: int,
    n: int = CHECKERBOARD_GRID_N,
) -> tuple[np.ndarray, np.ndarray]:
    """Return binary float masks m0, m1 in {0,1}, shape (H,W), complementary.

    Convention (matches CompVis inpaint.py): 1 = region to inpaint (white).
    Cells are an N×N grid over the image; cell (i,j) is white in m0 when (i+j)%2==0.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if height <= 0 or width <= 0:
        raise ValueError(f"invalid size {(height, width)}")

    ys = np.arange(height)
    xs = np.arange(width)
    cell_y = (ys * n) // height
    cell_x = (xs * n) // width
    grid = (cell_y[:, None] + cell_x[None, :]) % 2
    m0 = (grid == 0).astype(np.float32)
    m1 = 1.0 - m0
    return m0, m1


def validate_masks(m0: np.ndarray, m1: np.ndarray) -> None:
    if m0.shape != m1.shape:
        raise ValueError("m0/m1 shape mismatch")
    if not set(np.unique(m0)).issubset({0.0, 1.0}):
        raise ValueError("m0 not binary")
    if not set(np.unique(m1)).issubset({0.0, 1.0}):
        raise ValueError("m1 not binary")
    if not np.allclose(m0 + m1, 1.0):
        raise ValueError("m0 and m1 are not complementary")
    if not np.all((m0 + m1) >= 1.0 - 1e-6):
        raise ValueError("combined coverage incomplete")


def save_mask_png(mask: np.ndarray, path: Path) -> None:
    path = assert_under_project_root(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = (np.clip(mask, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate DISPATCH checkerboard masks")
    parser.add_argument("--height", type=int, default=RESOLUTION)
    parser.add_argument("--width", type=int, default=RESOLUTION)
    parser.add_argument("--n", type=int, default=CHECKERBOARD_GRID_N)
    parser.add_argument("--out-m0", type=str, required=True)
    parser.add_argument("--out-m1", type=str, required=True)
    args = parser.parse_args()
    seed_everything()
    m0, m1 = generate_checkerboard_masks(args.height, args.width, args.n)
    validate_masks(m0, m1)
    save_mask_png(m0, Path(args.out_m0))
    save_mask_png(m1, Path(args.out_m1))
    print(f"Wrote m0={args.out_m0} m1={args.out_m1} shape={m0.shape} n={args.n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
