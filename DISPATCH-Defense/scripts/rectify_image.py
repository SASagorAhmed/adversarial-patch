#!/usr/bin/env python
"""Selective rectification: I_hat = A * I_tilde + (1-A) * I."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.paths_config import assert_under_project_root


def rectify_image(
    original_rgb: np.ndarray,
    regenerated_rgb: np.ndarray,
    adversarial_mask: np.ndarray,
) -> np.ndarray:
    if original_rgb.shape != regenerated_rgb.shape:
        raise ValueError("I / I_tilde shape mismatch")
    a = adversarial_mask.astype(np.float32)
    if a.max() > 1.0:
        a = a / 255.0
    a = (a >= 0.5).astype(np.float32)
    if a.ndim == 2:
        a = a[..., None]
    out = a * regenerated_rgb.astype(np.float32) + (1.0 - a) * original_rgb.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--original", required=True)
    p.add_argument("--regenerated", required=True)
    p.add_argument("--mask", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    i = np.array(Image.open(args.original).convert("RGB"))
    it = np.array(Image.open(args.regenerated).convert("RGB"))
    a = np.array(Image.open(args.mask).convert("L"))
    out = rectify_image(i, it, a)
    path = assert_under_project_root(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out).save(path)
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
