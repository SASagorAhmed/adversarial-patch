#!/usr/bin/env python
"""L2 difference map between attacked image I and regenerated I_tilde."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.dispatch_config import DIFFERENCE_BLUR_KERNEL, DIFFERENCE_BLUR_SIGMA
from config.paths_config import assert_under_project_root


def compute_l2_difference(i_rgb: np.ndarray, i_tilde_rgb: np.ndarray) -> np.ndarray:
    """Per-pixel RGB L2 distance; inputs uint8 or float; normalized to [0,1] first."""
    a = i_rgb.astype(np.float32)
    b = i_tilde_rgb.astype(np.float32)
    if a.max() > 1.0:
        a = a / 255.0
    if b.max() > 1.0:
        b = b / 255.0
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    diff = a - b
    return np.sqrt(np.sum(diff * diff, axis=2)).astype(np.float32)


def smooth_difference(
    d: np.ndarray,
    ksize: int = DIFFERENCE_BLUR_KERNEL,
    sigma: float = DIFFERENCE_BLUR_SIGMA,
) -> np.ndarray:
    k = int(ksize)
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(d, (k, k), sigmaX=float(sigma))


def to_vis_png(d: np.ndarray) -> np.ndarray:
    dmin, dmax = float(d.min()), float(d.max())
    if dmax <= dmin:
        norm = np.zeros_like(d, dtype=np.uint8)
    else:
        norm = ((d - dmin) / (dmax - dmin) * 255.0).astype(np.uint8)
    return norm


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--original", required=True)
    p.add_argument("--regenerated", required=True)
    p.add_argument("--out-npy", required=True)
    p.add_argument("--out-raw-png", required=True)
    p.add_argument("--out-smoothed-png", required=True)
    p.add_argument("--out-smoothed-npy", required=False, default=None)
    args = p.parse_args()

    i = np.array(Image.open(args.original).convert("RGB"))
    it = np.array(Image.open(args.regenerated).convert("RGB"))
    raw = compute_l2_difference(i, it)
    sm = smooth_difference(raw)

    npy_path = assert_under_project_root(args.out_npy)
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy_path, raw)

    raw_png = assert_under_project_root(args.out_raw_png)
    Image.fromarray(to_vis_png(raw), mode="L").save(raw_png)

    sm_png = assert_under_project_root(args.out_smoothed_png)
    Image.fromarray(to_vis_png(sm), mode="L").save(sm_png)

    if args.out_smoothed_npy:
        sn = assert_under_project_root(args.out_smoothed_npy)
        np.save(sn, sm)

    print(f"raw_minmax=({raw.min():.6f},{raw.max():.6f}) smoothed_saved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
