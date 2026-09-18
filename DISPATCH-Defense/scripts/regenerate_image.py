#!/usr/bin/env python
"""Two-pass LDM regeneration at processing resolution (no true patch coords)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.dispatch_config import (
    CHECKERBOARD_GRID_N,
    DIFFUSION_STEPS,
    RESOLUTION,
    seed_everything,
)
from config.paths_config import assert_under_project_root
from scripts.combine_regeneration import combine_regeneration
from scripts.generate_checkerboard_masks import generate_checkerboard_masks, save_mask_png, validate_masks
from scripts.ldm_inpaint_adapter import LDMInpaintAdapter


def regenerate_image(
    image_path: Path,
    out_dir: Path,
    image_id: str,
    adapter: LDMInpaintAdapter | None = None,
    resolution: int = RESOLUTION,
    grid_n: int = CHECKERBOARD_GRID_N,
    steps: int = DIFFUSION_STEPS,
) -> dict:
    """Run full regenerate stage; write intermediates under out_dir (DISPATCH root only)."""
    out_dir = assert_under_project_root(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything()

    src = Image.open(image_path).convert("RGB")
    orig_w, orig_h = src.size
    proc = src.resize((resolution, resolution), Image.Resampling.LANCZOS)
    proc_arr = np.array(proc)

    m0, m1 = generate_checkerboard_masks(resolution, resolution, grid_n)
    validate_masks(m0, m1)
    m0_path = out_dir / f"{image_id}_mask_m0.png"
    m1_path = out_dir / f"{image_id}_mask_m1.png"
    save_mask_png(m0, m0_path)
    save_mask_png(m1, m1_path)

    if adapter is None:
        adapter = LDMInpaintAdapter(steps=steps)

    t0 = time.perf_counter()
    pass0 = adapter.inpaint(proc_arr, m0, steps=steps)
    t_pass0 = time.perf_counter() - t0
    pass0_path = out_dir / f"{image_id}_regenerated_pass0.png"
    Image.fromarray(pass0).save(assert_under_project_root(pass0_path))

    t1 = time.perf_counter()
    pass1 = adapter.inpaint(proc_arr, m1, steps=steps)
    t_pass1 = time.perf_counter() - t1
    pass1_path = out_dir / f"{image_id}_regenerated_pass1.png"
    Image.fromarray(pass1).save(assert_under_project_root(pass1_path))

    full = combine_regeneration(pass0, pass1, m0, m1)
    full_path = out_dir / f"{image_id}_regenerated_full.png"
    Image.fromarray(full).save(assert_under_project_root(full_path))

    # Keep a processing-resolution copy of the attacked input for difference stage
    proc_in_path = out_dir / f"{image_id}_attacked_proc{resolution}.png"
    proc.save(assert_under_project_root(proc_in_path))

    meta = {
        "image_id": image_id,
        "source_path": str(image_path),
        "original_width": orig_w,
        "original_height": orig_h,
        "processing_resolution": resolution,
        "checkerboard_grid_n": grid_n,
        "diffusion_steps": steps,
        "sampler": "DDIM",
        "pass0_time_sec": t_pass0,
        "pass1_time_sec": t_pass1,
        "paths": {
            "m0": str(m0_path),
            "m1": str(m1_path),
            "pass0": str(pass0_path),
            "pass1": str(pass1_path),
            "full": str(full_path),
            "attacked_proc": str(proc_in_path),
        },
    }
    meta_path = assert_under_project_root(out_dir / f"{image_id}_regen_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main() -> int:
    p = argparse.ArgumentParser(description="DISPATCH two-pass regeneration")
    p.add_argument("--image", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--image-id", required=True)
    p.add_argument("--steps", type=int, default=DIFFUSION_STEPS)
    p.add_argument("--n", type=int, default=CHECKERBOARD_GRID_N)
    args = p.parse_args()
    meta = regenerate_image(
        Path(args.image),
        Path(args.out_dir),
        args.image_id,
        steps=args.steps,
        grid_n=args.n,
    )
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
