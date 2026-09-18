#!/usr/bin/env python
"""Combine two complementary inpainting passes into full regenerated image."""
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


def combine_regeneration(
    i_tilde_0: np.ndarray,
    i_tilde_1: np.ndarray,
    m0: np.ndarray,
    m1: np.ndarray,
) -> np.ndarray:
    """I_tilde = I_tilde_0 * m0 + I_tilde_1 * m1 (masks broadcast over RGB)."""
    if i_tilde_0.shape != i_tilde_1.shape:
        raise ValueError("pass0/pass1 shape mismatch")
    m0f = m0.astype(np.float32)
    m1f = m1.astype(np.float32)
    if m0f.max() > 1.0:
        m0f = m0f / 255.0
        m1f = m1f / 255.0
    if m0f.ndim == 2:
        m0f = m0f[..., None]
        m1f = m1f[..., None]
    out = i_tilde_0.astype(np.float32) * m0f + i_tilde_1.astype(np.float32) * m1f
    return np.clip(out, 0, 255).astype(np.uint8)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pass0", required=True)
    p.add_argument("--pass1", required=True)
    p.add_argument("--m0", required=True)
    p.add_argument("--m1", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    i0 = np.array(Image.open(args.pass0).convert("RGB"))
    i1 = np.array(Image.open(args.pass1).convert("RGB"))
    m0 = np.array(Image.open(args.m0).convert("L"))
    m1 = np.array(Image.open(args.m1).convert("L"))
    out = combine_regeneration(i0, i1, m0, m1)
    out_path = assert_under_project_root(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out).save(out_path)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
