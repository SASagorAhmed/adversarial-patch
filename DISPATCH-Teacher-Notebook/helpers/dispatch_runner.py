#!/usr/bin/env python
"""Paper-faithful Automatic + Known Location runners. Writes only under Teacher-Notebook."""
from __future__ import annotations

import os

# Jupyter sets MPLBACKEND=matplotlib_inline; the DISPATCH venv cannot use that backend.
os.environ["MPLBACKEND"] = "Agg"

import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(r"D:\project CS\DISPATCH-Teacher-Notebook").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from helpers.evaluation import localization_metrics
from helpers.safety import assert_writable, ensure_dir, load_config
from helpers.visualization import overlay_mask, save_jpg, save_png, to_vis_png


def seed_all(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_rows() -> list[dict]:
    path = ROOT / "source_data" / "source_attack_data" / "selected_images.csv"
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def rect_mask(h, w, x1, y1, x2, y2) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.uint8)
    m[max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)] = 255
    return m


def expand_xyxy(x1, y1, x2, y2, exp, w, h):
    return max(0, x1 - exp), max(0, y1 - exp), min(w, x2 + exp), min(h, y2 + exp)


def outside_mae(attacked, restored, mask) -> float:
    outside = mask <= 127
    if not np.any(outside):
        return 0.0
    return float(np.abs(restored.astype(np.float32) - attacked.astype(np.float32))[outside].mean())


def _prepare_paths() -> dict:
    cfg = load_config()
    pf = Path(cfg["paper_faithful_root"])
    dd = Path(cfg["dispatch_root"])
    for p in (str(pf), str(pf / "implementation"), str(dd), str(cfg["official_ldm_root"]), str(cfg["taming_root"])):
        if p not in sys.path:
            sys.path.insert(0, p)
    return cfg


def run_automatic() -> None:
    cfg = _prepare_paths()
    from implementation.ldm_adapter import OfficialLDMAdapter
    from implementation.localization import detect_adversarial_mask, gaussian_kernel_size
    from implementation.rectification import mask_to_uint8_orig, rectify, resize_to_original, to_pil_rgb
    from implementation.regeneration import make_checkerboard_masks, validate_complementary

    seed = int(cfg["seed"])
    res = int(cfg["resolution"])
    n = int(cfg["checkerboard_n"])
    steps = int(cfg["ddim_steps"])
    auto = ROOT / "automatic"
    for d in [
        auto / "masks" / "checkerboard_mask_0",
        auto / "masks" / "checkerboard_mask_1",
        auto / "masks" / "predicted_masks",
        auto / "masks" / "true_patch_masks_evaluation_only",
        auto / "regenerated_images",
        auto / "restored_images",
        auto / "results" / "difference_maps" / "raw",
        auto / "results" / "difference_maps" / "smoothed",
        auto / "results" / "mask_overlays",
        auto / "results" / "verification",
    ]:
        ensure_dir(d)

    seed_all(seed)
    mask0, mask1 = make_checkerboard_masks(res, n)
    val = validate_complementary(mask0, mask1)
    (assert_writable(auto / "results" / "checkerboard_validation.json")).write_text(
        json.dumps(val, indent=2), encoding="utf-8"
    )
    m0 = (mask0.squeeze().numpy() * 255).astype(np.uint8)
    m1 = (mask1.squeeze().numpy() * 255).astype(np.uint8)
    save_png(m0, auto / "masks" / "checkerboard_mask_0" / "template_512.png")
    save_png(m1, auto / "masks" / "checkerboard_mask_1" / "template_512.png")

    print("Loading paper-faithful LDM adapter...")
    adapter = OfficialLDMAdapter(steps=steps)
    rows = load_rows()
    out = []
    for r in rows:
        fn = r["filename"]
        stem = Path(fn).stem
        att_path = ROOT / "source_data" / "attacked_images" / fn
        print(f"=== AUTOMATIC {fn} ===")
        t0 = time.perf_counter()
        seed_all(seed)
        regen = adapter.regenerate_pair(att_path, mask0, mask1, size=res, steps=steps)
        gen, inp = regen["generated_01"], regen["input_01"]
        ori_w, ori_h = regen["ori_size"]
        gen_orig = resize_to_original(to_pil_rgb(gen), (ori_w, ori_h))
        save_jpg(gen_orig, auto / "regenerated_images" / fn)

        adv, dist, km = detect_adversarial_mask(
            inp, gen, size=res, num_grids=n, kmeans_random_state=seed
        )
        raw = torch.norm(inp - gen, p=2, dim=0).detach().cpu().numpy()
        sm = dist.detach().cpu().numpy()
        save_png(to_vis_png(raw), auto / "results" / "difference_maps" / "raw" / f"{stem}.png")
        save_png(to_vis_png(sm), auto / "results" / "difference_maps" / "smoothed" / f"{stem}.png")

        mask_orig = mask_to_uint8_orig(adv, (ori_w, ori_h))
        save_png(mask_orig, auto / "masks" / "predicted_masks" / f"{stem}.png")

        attacked = np.array(Image.open(att_path).convert("RGB"))
        save_jpg(overlay_mask(attacked, mask_orig), auto / "results" / "mask_overlays" / fn)

        tmask = rect_mask(
            attacked.shape[0], attacked.shape[1], int(r["patch_x1"]), int(r["patch_y1"]), int(r["patch_x2"]), int(r["patch_y2"])
        )
        save_png(tmask, auto / "masks" / "true_patch_masks_evaluation_only" / f"{stem}.png")
        loc = localization_metrics(mask_orig, tmask)

        rect = rectify(adv, gen, inp)
        rest = resize_to_original(to_pil_rgb(rect), (ori_w, ori_h))
        save_jpg(rest, auto / "restored_images" / fn)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        rec = {
            "filename": fn,
            "image_id": r["image_id"],
            "target_class": r["target_class"],
            "case_type": r["case_type"],
            "sample_mode": regen.get("sample_mode"),
            "inference_time_sec": time.perf_counter() - t0,
            **loc,
            "status": "completed",
        }
        out.append(rec)
        print(f"  mask_iou={loc['mask_iou']:.4f} P={loc['precision']:.4f} R={loc['recall']:.4f}")

    _write_csv(auto / "results" / "per_image_results.csv", out)
    succ = [x for x in out if x["case_type"] == "attack_success"]
    summary = (
        "AUTOMATIC PAPER-FAITHFUL DISPATCH (diagnostic subset — NOT DRR)\n"
        f"N={n} steps={steps} Gaussian k={gaussian_kernel_size(res, n)} sigma=1.0 MORPH_OPEN=3x3\n"
        f"mean mask IoU={float(np.mean([x['mask_iou'] for x in out])):.4f}\n"
        f"mean precision={float(np.mean([x['precision'] for x in out])):.4f}\n"
        f"mean recall={float(np.mean([x['recall'] for x in out])):.4f}\n"
        "True patch was evaluation-only. LDM input = attacked image only.\n"
    )
    (assert_writable(auto / "results" / "summary.txt")).write_text(summary, encoding="utf-8")
    print(summary)


def run_known_location() -> None:
    cfg = _prepare_paths()
    dd = Path(cfg["dispatch_root"])
    if str(dd) not in sys.path:
        sys.path.insert(0, str(dd))
    from scripts.ldm_inpaint_adapter import LDMInpaintAdapter

    seed = int(cfg["seed"])
    res = int(cfg["resolution"])
    steps = int(cfg["ddim_steps"])
    kl = ROOT / "known_location"
    for sub in [
        "masks/exact_mask",
        "masks/expanded_4px",
        "ldm_outputs/exact_mask",
        "ldm_outputs/expanded_4px",
        "restored_images/exact_mask",
        "restored_images/expanded_4px",
        "results/verification/exact_mask",
        "results/verification/expanded_4px",
    ]:
        ensure_dir(kl / sub)

    seed_all(seed)
    print("Loading CompVis inpaint adapter for Known Location...")
    adapter = LDMInpaintAdapter(steps=steps, seed=seed)
    rows = load_rows()
    out = []
    for r in rows:
        fn = r["filename"]
        print(f"=== KNOWN LOCATION {fn} ===")
        attacked = np.array(Image.open(ROOT / "source_data" / "attacked_images" / fn).convert("RGB"))
        h, w = attacked.shape[:2]
        x1, y1, x2, y2 = int(r["patch_x1"]), int(r["patch_y1"]), int(r["patch_x2"]), int(r["patch_y2"])
        exact = rect_mask(h, w, x1, y1, x2, y2)
        ex = expand_xyxy(x1, y1, x2, y2, 4, w, h)
        expm = rect_mask(h, w, *ex)
        save_png(exact, kl / "masks" / "exact_mask" / f"{Path(fn).stem}.png")
        save_png(expm, kl / "masks" / "expanded_4px" / f"{Path(fn).stem}.png")

        rec = {
            "filename": fn,
            "image_id": r["image_id"],
            "target_class": r["target_class"],
            "case_type": r["case_type"],
        }
        t0 = time.perf_counter()
        for name, mask in (("exact_mask", exact), ("expanded_4px", expm)):
            seed_all(seed)
            proc_img = np.array(Image.fromarray(attacked).resize((res, res), Image.Resampling.LANCZOS))
            proc_mask = np.array(Image.fromarray(mask, mode="L").resize((res, res), Image.Resampling.NEAREST))
            ldm512 = adapter.inpaint(proc_img, proc_mask, steps=steps)
            ldm_up = np.array(Image.fromarray(ldm512).resize((w, h), Image.Resampling.LANCZOS))
            save_jpg(ldm_up, kl / "ldm_outputs" / name / fn)
            m = (mask > 127)[..., None]
            restored = np.where(m, ldm_up, attacked).astype(np.uint8)
            save_jpg(restored, kl / "restored_images" / name / fn)
            rec[f"{name}_outside_mae"] = outside_mae(attacked, restored, mask)
            save_jpg(restored, kl / "results" / "verification" / name / fn)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        rec["inference_time_sec"] = time.perf_counter() - t0
        rec["status"] = "completed"
        out.append(rec)
        print(f"  exact_mae={rec['exact_mask_outside_mae']:.6f} +4_mae={rec['expanded_4px_outside_mae']:.6f}")

    _write_csv(kl / "results" / "per_image_results.csv", out)
    txt = (
        "KNOWN LOCATION (oracle mask — diagnostic, not Automatic DISPATCH)\n"
        "LDM input = attacked image + binary mask. Clean pixels never composited.\n"
        f"mean outside MAE exact={float(np.mean([x['exact_mask_outside_mae'] for x in out])):.8f}\n"
        f"mean outside MAE +4={float(np.mean([x['expanded_4px_outside_mae'] for x in out])):.8f}\n"
    )
    (assert_writable(kl / "results" / "summary.txt")).write_text(txt, encoding="utf-8")
    print(txt)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path = assert_writable(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in {"automatic", "all"}:
        run_automatic()
    if mode in {"known", "all"}:
        run_known_location()
