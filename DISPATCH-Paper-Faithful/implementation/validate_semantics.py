#!/usr/bin/env python
"""Numerical mask-semantics + checkerboard validation (no dataset, no writes to DISPATCH-Defense)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(r"D:\project CS\DISPATCH-Paper-Faithful").resolve()
sys.path.insert(0, str(ROOT))

from implementation.regeneration import make_checkerboard_masks, validate_complementary


def main() -> int:
    m0, m1 = make_checkerboard_masks(512, 32)
    val = validate_complementary(m0, m1)
    # CompVis / official: masked = (1-mask)*image; mask=1 must zero the pixel.
    image = torch.ones((1, 3, 512, 512))
    masked0 = (1.0 - m0) * image
    masked1 = (1.0 - m1) * image
    hole0 = float(masked0[0, 0][m0[0, 0] > 0.5].abs().max())
    keep0 = float(masked0[0, 0][m0[0, 0] < 0.5].min())
    semantics = {
        "mask_1_means_inpaint_hole": hole0 == 0.0,
        "mask_0_means_preserve": keep0 == 1.0,
        "white_fraction_m0": float(m0.mean()),
        "checkerboard": val,
    }
    out = ROOT / "audit" / "mask_semantics_validation.json"
    out.write_text(json.dumps(semantics, indent=2), encoding="utf-8")
    if not semantics["mask_1_means_inpaint_hole"] or not val["sum_allclose_1"]:
        raise SystemExit("SEMANTICS FAIL")
    print(json.dumps(semantics, indent=2))
    print("MASK_SEMANTICS: PASS (1=inpaint, complementary)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
