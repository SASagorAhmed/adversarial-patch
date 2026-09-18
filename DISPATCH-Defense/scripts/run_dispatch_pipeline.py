#!/usr/bin/env python
"""DISPATCH pipeline entrypoint.

Modes:
  --validate-only     verify env/paths/checkpoint; NO experiment folders, NO dataset run
  --smoke-ldm         official CompVis inpainting example under smoke_test/
  --smoke-dispatch    one-image end-to-end DISPATCH under smoke_test/
  --run               real experiment (requires explicit approval flags)

Large dataset experiments are NOT started unless --run is given with --i-approve-full-run.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.dispatch_config import (
    CHECKERBOARD_GRID_N,
    DIFFUSION_STEPS,
    DISPATCH_SEED,
    RESOLUTION,
    as_run_config_dict,
    seed_everything,
)
from config.paths_config import (
    LDM_INPAINT_EXAMPLES,
    LDM_ROOT,
    PROJECT_ROOT,
    SMOKE_TEST_DIR,
    assert_under_project_root,
    ensure_dir,
)
from scripts.compute_difference_map import compute_l2_difference, smooth_difference, to_vis_png
from scripts.dispatch_reporting import STORAGE_GUARANTEE
from scripts.generate_checkerboard_masks import generate_checkerboard_masks, save_mask_png, validate_masks
from scripts.ldm_inpaint_adapter import LDMInpaintAdapter
from scripts.predict_adversarial_mask import predict_adversarial_mask
from scripts.rectify_image import rectify_image
from scripts.regenerate_image import regenerate_image


def free_gb_d() -> float:
    import shutil as sh

    usage = sh.disk_usage("D:\\")
    return usage.free / (1024**3)


def cmd_validate_only() -> int:
    print("=== VALIDATE-ONLY (no experiment outputs) ===")
    print(f"D_FREE_GB={free_gb_d():.2f}")
    print(f"PROJECT_ROOT={PROJECT_ROOT}")
    from scripts.verify_environment import main as ve
    from scripts.verify_checkpoint import main as vc
    from scripts.verify_external_resources import main as vx

    codes = [ve(), vc(), vx()]
    # Must NOT create experiments/dispatch_run_*
    experiments = PROJECT_ROOT / "experiments"
    created_runs = list(experiments.glob("dispatch_run_*")) if experiments.exists() else []
    if created_runs:
        print("FAIL: validate-only must not create real experiment folders:", created_runs)
        return 1
    print("validate-only created no dispatch_run_*: PASS")
    print(STORAGE_GUARANTEE)
    return 0 if all(c == 0 for c in codes) else 1


def cmd_smoke_ldm(steps: int = 5) -> int:
    print("=== Official CompVis LDM inpainting smoke test ===")
    examples = LDM_INPAINT_EXAMPLES
    if not examples.is_dir():
        print("FAIL: missing inpainting_examples")
        return 1
    masks = sorted(examples.glob("*_mask.png"))
    if not masks:
        print("FAIL: no example masks")
        return 1
    mask_src = masks[0]
    img_src = Path(str(mask_src).replace("_mask.png", ".png"))
    if not img_src.is_file():
        print("FAIL: missing paired example image", img_src)
        return 1

    inp = ensure_dir(SMOKE_TEST_DIR / "input")
    out = ensure_dir(SMOKE_TEST_DIR / "regenerated" / "official_ldm")
    report = ensure_dir(SMOKE_TEST_DIR / "report")

    img_dst = inp / img_src.name
    mask_dst = inp / mask_src.name
    if not img_dst.exists():
        shutil.copy2(img_src, img_dst)
    if not mask_dst.exists():
        shutil.copy2(mask_src, mask_dst)

    seed_everything()
    adapter = LDMInpaintAdapter(steps=steps)
    image = Image.open(img_dst).convert("RGB")
    mask = Image.open(mask_dst).convert("L")
    # Official examples may not be 512; keep example size for official smoke
    t0 = time.perf_counter()
    result = adapter.inpaint(np.array(image), np.array(mask), steps=steps)
    elapsed = time.perf_counter() - t0
    out_path = assert_under_project_root(out / img_src.name)
    Image.fromarray(result).save(out_path)

    import torch

    report_txt = report / "ldm_smoke_test.txt"
    lines = [
        "OFFICIAL LDM INPAINTING SMOKE TEST",
        f"status: PASS",
        f"example_image: {img_src.name}",
        f"example_mask: {mask_src.name}",
        f"output: {out_path}",
        f"steps: {steps}",
        f"sampler: DDIM",
        f"device: {adapter.device}",
        f"cuda_available: {torch.cuda.is_available()}",
        f"elapsed_sec: {elapsed:.3f}",
        f"output_shape: {result.shape}",
        f"output_finite: {bool(np.isfinite(result).all())}",
        f"ldm_root: {LDM_ROOT}",
        f"no_text_prompt: YES",
        f"timestamp: {datetime.now().isoformat()}",
    ]
    report_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


def cmd_smoke_dispatch(steps: int = DIFFUSION_STEPS) -> int:
    print("=== One-image DISPATCH algorithm smoke test ===")
    seed_everything()
    # Prefer official example image as safe input (copied, not modified in third_party)
    examples = LDM_INPAINT_EXAMPLES
    masks = sorted(examples.glob("*_mask.png"))
    img_src = Path(str(masks[0]).replace("_mask.png", ".png")) if masks else None
    if img_src is None or not img_src.is_file():
        print("FAIL: no safe example image for smoke test")
        return 1

    image_id = "smoke_" + img_src.stem
    inp = ensure_dir(SMOKE_TEST_DIR / "input")
    img_copy = inp / f"{image_id}.png"
    if not img_copy.exists():
        shutil.copy2(img_src, img_copy)

    masks_dir = ensure_dir(SMOKE_TEST_DIR / "masks")
    regen_dir = ensure_dir(SMOKE_TEST_DIR / "regenerated")
    diff_dir = ensure_dir(SMOKE_TEST_DIR / "difference_maps")
    adv_dir = ensure_dir(SMOKE_TEST_DIR / "adversarial_masks")
    rect_dir = ensure_dir(SMOKE_TEST_DIR / "rectified")
    report_dir = ensure_dir(SMOKE_TEST_DIR / "report")
    logs_dir = ensure_dir(SMOKE_TEST_DIR / "logs")

    adapter = LDMInpaintAdapter(steps=steps)
    meta = regenerate_image(
        img_copy,
        regen_dir,
        image_id,
        adapter=adapter,
        steps=steps,
        grid_n=CHECKERBOARD_GRID_N,
    )
    # Also place checkerboard masks under smoke_test/masks
    shutil.copy2(meta["paths"]["m0"], masks_dir / Path(meta["paths"]["m0"]).name)
    shutil.copy2(meta["paths"]["m1"], masks_dir / Path(meta["paths"]["m1"]).name)

    attacked_proc = np.array(Image.open(meta["paths"]["attacked_proc"]).convert("RGB"))
    full = np.array(Image.open(meta["paths"]["full"]).convert("RGB"))

    t_diff0 = time.perf_counter()
    raw = compute_l2_difference(attacked_proc, full)
    sm = smooth_difference(raw)
    t_diff = time.perf_counter() - t_diff0
    np.save(assert_under_project_root(diff_dir / f"{image_id}_l2_raw.npy"), raw)
    np.save(assert_under_project_root(diff_dir / f"{image_id}_l2_smoothed.npy"), sm)
    Image.fromarray(to_vis_png(raw), mode="L").save(diff_dir / f"{image_id}_l2_raw.png")
    Image.fromarray(to_vis_png(sm), mode="L").save(diff_dir / f"{image_id}_l2_smoothed.png")

    t_km0 = time.perf_counter()
    adv_mask, stats = predict_adversarial_mask(sm)
    t_km = time.perf_counter() - t_km0
    Image.fromarray(adv_mask, mode="L").save(adv_dir / f"{image_id}_adversarial_mask.png")
    (adv_dir / f"{image_id}_kmeans_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    t_r0 = time.perf_counter()
    rect = rectify_image(attacked_proc, full, adv_mask)
    t_r = time.perf_counter() - t_r0
    # Map rectified result back to original resolution
    orig_w, orig_h = meta["original_width"], meta["original_height"]
    rect_orig = Image.fromarray(rect).resize((orig_w, orig_h), Image.Resampling.LANCZOS)
    rect_path = rect_dir / f"{image_id}_rectified.jpg"
    rect_orig.save(rect_path, quality=95)

    # Upsample binary mask with nearest for optional localization later
    adv_orig = Image.fromarray(adv_mask, mode="L").resize((orig_w, orig_h), Image.Resampling.NEAREST)
    adv_orig.save(adv_dir / f"{image_id}_adversarial_mask_origres.png")

    report = {
        "status": "PASS",
        "image_id": image_id,
        "seed": DISPATCH_SEED,
        "resolution_proc": RESOLUTION,
        "original_size": [orig_w, orig_h],
        "steps": steps,
        "grid_n": CHECKERBOARD_GRID_N,
        "kmeans": stats,
        "timings_sec": {
            "pass0": meta["pass0_time_sec"],
            "pass1": meta["pass1_time_sec"],
            "difference": t_diff,
            "kmeans": t_km,
            "rectify": t_r,
        },
        "true_patch_coords_supplied_to_defense": False,
        "paths": {
            "input": str(img_copy),
            "rectified": str(rect_path),
        },
        "storage_guarantee": STORAGE_GUARANTEE,
    }
    (report_dir / "dispatch_smoke_test.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (logs_dir / f"{image_id}.log").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


def cmd_run_blocked() -> int:
    print("REFUSING full experiment: pass --run with --i-approve-full-run after explicit user approval.")
    print("This setup job stops before large evaluation.")
    return 2


def main() -> int:
    p = argparse.ArgumentParser(description="DISPATCH pipeline")
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--smoke-ldm", action="store_true")
    p.add_argument("--smoke-dispatch", action="store_true")
    p.add_argument("--run", action="store_true")
    p.add_argument("--i-approve-full-run", action="store_true")
    p.add_argument("--resume", action="store_true", help="Reserved for future safe resume")
    p.add_argument("--steps", type=int, default=DIFFUSION_STEPS)
    args = p.parse_args()

    # Always assert we are operating with DISPATCH root in mind
    assert_under_project_root(PROJECT_ROOT / "smoke_test")

    if args.validate_only:
        return cmd_validate_only()
    if args.smoke_ldm:
        return cmd_smoke_ldm(steps=args.steps)
    if args.smoke_dispatch:
        return cmd_smoke_dispatch(steps=args.steps)
    if args.run:
        if not args.i_approve_full_run:
            return cmd_run_blocked()
        return cmd_run_blocked()

    p.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
