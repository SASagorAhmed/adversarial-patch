#!/usr/bin/env python
"""Paper/source-faithful Automatic DisPatch on the isolated diagnostic set.

Writes ONLY under D:\\project CS\\DISPATCH-Paper-Faithful\\
Never uses true patch coords until after the predicted mask is frozen.
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToPILImage

ROOT = Path(r"D:\project CS\DISPATCH-Paper-Faithful").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.reference_config import (
    CURRENT_ATTACKED,
    CURRENT_CLEAN,
    CURRENT_SELECTED,
    DIFFUSION_STEPS,
    NUM_GRIDS,
    OFFICIAL_COMMIT,
    RESOLUTION,
    RUN_SEED,
    SELECTED_IDS,
    assert_isolated,
)
from implementation.evaluation import detect, load_hyper_yolo, localization_metrics, yolo_predict
from implementation.ldm_adapter import OfficialLDMAdapter
from implementation.localization import detect_adversarial_mask, gaussian_kernel_size
from implementation.rectification import mask_to_uint8_orig, rectify, resize_to_original, to_pil_rgb
from implementation.regeneration import make_checkerboard_masks, validate_complementary

DIAG = ROOT / "diagnostic"
LOGS = ROOT / "logs"


def seed_all(seed: int = RUN_SEED) -> None:
    import os
    import random

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_selected() -> list[dict]:
    rows = []
    with CURRENT_SELECTED.open(encoding="utf-8", newline="") as f:
        by = {r["image_id"]: r for r in csv.DictReader(f)}
    for iid in SELECTED_IDS:
        if iid not in by:
            raise RuntimeError(f"STOP: {iid} missing from current diagnostic selected_images.csv")
        rows.append(by[iid])
    return rows


def copy_sources(rows: list[dict]) -> None:
    att_dir = assert_isolated(DIAG / "source_attacked_images")
    cln_dir = assert_isolated(DIAG / "source_clean_images")
    meta_dir = assert_isolated(DIAG / "source_attack_data")
    att_dir.mkdir(parents=True, exist_ok=True)
    cln_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CURRENT_SELECTED, meta_dir / "selected_images.csv")
    for r in rows:
        fn = r["filename"]
        src_a = CURRENT_ATTACKED / fn
        src_c = CURRENT_CLEAN / fn
        if not src_a.is_file() or not src_c.is_file():
            raise FileNotFoundError(fn)
        dst_a = att_dir / fn
        dst_c = cln_dir / fn
        if not dst_a.exists():
            shutil.copy2(src_a, dst_a)
        if not dst_c.exists():
            shutil.copy2(src_c, dst_c)


def true_mask(h, w, r) -> np.ndarray:
    x1, y1, x2, y2 = int(r["patch_x1"]), int(r["patch_y1"]), int(r["patch_x2"]), int(r["patch_y2"])
    m = np.zeros((h, w), dtype=np.uint8)
    m[max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)] = 255
    return m


def gt_xyxy(r) -> list[float]:
    x, y, w, h = float(r["gt_x"]), float(r["gt_y"]), float(r["gt_width"]), float(r["gt_height"])
    return [x, y, x + w, y + h]


def save_png(arr, path: Path, mode="L") -> None:
    path = assert_isolated(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode=mode).save(path)


def tensor_to_uint8_hwc(t: torch.Tensor) -> np.ndarray:
    x = t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return (x * 255.0 + 0.5).astype(np.uint8)


def main() -> int:
    seed_all()
    rows = load_selected()
    copy_sources(rows)

    for d in [
        DIAG / "regenerated_images",
        DIAG / "masks" / "checkerboard_m0",
        DIAG / "masks" / "checkerboard_m1",
        DIAG / "masks" / "predicted_masks",
        DIAG / "masks" / "true_patch_masks_evaluation_only",
        DIAG / "restored_images",
        DIAG / "predictions" / "clean" / "images",
        DIAG / "predictions" / "clean" / "labels",
        DIAG / "predictions" / "attacked" / "images",
        DIAG / "predictions" / "attacked" / "labels",
        DIAG / "predictions" / "restored" / "images",
        DIAG / "predictions" / "restored" / "labels",
        DIAG / "results",
        LOGS,
    ]:
        assert_isolated(d).mkdir(parents=True, exist_ok=True)

    mask0, mask1 = make_checkerboard_masks(RESOLUTION, NUM_GRIDS)
    val = validate_complementary(mask0, mask1)
    (DIAG / "results" / "checkerboard_validation.json").write_text(json.dumps(val, indent=2), encoding="utf-8")
    if not val["sum_allclose_1"] or val["overlap_both_one_pixels"] or val["hole_both_zero_pixels"]:
        raise RuntimeError(f"Checkerboard validation failed: {val}")

    m0_vis = (mask0.squeeze().numpy() * 255).astype(np.uint8)
    m1_vis = (mask1.squeeze().numpy() * 255).astype(np.uint8)
    save_png(m0_vis, DIAG / "masks" / "checkerboard_m0" / "template_512.png")
    save_png(m1_vis, DIAG / "masks" / "checkerboard_m1" / "template_512.png")

    print("Loading official-style LDM adapter (read-only CompVis ckpt)...")
    adapter = OfficialLDMAdapter(steps=DIFFUSION_STEPS)
    print("Loading Hyper-YOLO (read-only weights)...")
    yolo = load_hyper_yolo()

    out_rows = []
    for r in rows:
        fn = r["filename"]
        stem = r["image_id"]
        att_path = DIAG / "source_attacked_images" / fn
        cln_path = DIAG / "source_clean_images" / fn
        done_json = DIAG / "results" / f"{stem}_reference.json"
        print(f"\n=== REFERENCE AUTOMATIC {fn} ({r['case_type']} {r['target_class']}) ===")
        if done_json.is_file():
            print("skip completed")
            out_rows.append(json.loads(done_json.read_text(encoding="utf-8")))
            continue

        t0 = time.perf_counter()
        seed_all()
        regen = adapter.regenerate_pair(att_path, mask0, mask1, size=RESOLUTION)
        gen = regen["generated_01"]
        inp = regen["input_01"]
        ori_w, ori_h = regen["ori_size"]

        gen_pil = to_pil_rgb(gen)
        gen_orig = resize_to_original(gen_pil, (ori_w, ori_h))
        gen_orig.save(assert_isolated(DIAG / "regenerated_images" / fn), quality=95)

        adv, dist, km_stats = detect_adversarial_mask(
            inp, gen, size=RESOLUTION, num_grids=NUM_GRIDS, kmeans_random_state=RUN_SEED
        )
        # Freeze predicted mask BEFORE looking at true patch.
        mask_orig = mask_to_uint8_orig(adv, (ori_w, ori_h))
        save_png(mask_orig, DIAG / "masks" / "predicted_masks" / f"{stem}.png")

        attacked = np.array(Image.open(att_path).convert("RGB"))
        h, w = attacked.shape[:2]
        tmask = true_mask(h, w, r)
        save_png(tmask, DIAG / "masks" / "true_patch_masks_evaluation_only" / f"{stem}.png")
        loc = localization_metrics(mask_orig, tmask)

        rect = rectify(adv, gen, inp)
        rect_orig = resize_to_original(to_pil_rgb(rect), (ori_w, ori_h))
        rest_path = assert_isolated(DIAG / "restored_images" / fn)
        rect_orig.save(rest_path, quality=95)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gt = gt_xyxy(r)
        cls = r["target_class"]
        clean_pred = yolo_predict(
            yolo, cln_path, DIAG / "predictions" / "clean" / "images", DIAG / "predictions" / "clean" / "labels", fn
        )
        att_pred = yolo_predict(
            yolo, att_path, DIAG / "predictions" / "attacked" / "images", DIAG / "predictions" / "attacked" / "labels", fn
        )
        rest_pred = yolo_predict(
            yolo, rest_path, DIAG / "predictions" / "restored" / "images", DIAG / "predictions" / "restored" / "labels", fn
        )
        c_det, c_conf, c_iou = detect(clean_pred["predictions"], cls, gt)
        a_det, a_conf, a_iou = detect(att_pred["predictions"], cls, gt)
        r_det, r_conf, r_iou = detect(rest_pred["predictions"], cls, gt)

        recovered = bool(r["case_type"] == "attack_success" and r_det)
        preserved = bool(r["case_type"] == "control" and r_det)
        elapsed = time.perf_counter() - t0

        rec = {
            "image_id": stem,
            "filename": fn,
            "target_class": cls,
            "case_type": r["case_type"],
            "patch_size": r["patch_size"],
            "official_commit": OFFICIAL_COMMIT,
            "gaussian_kernel_size": gaussian_kernel_size(RESOLUTION, NUM_GRIDS),
            "kmeans": km_stats,
            "reference_mask_iou": loc["mask_iou"],
            "reference_precision": loc["mask_precision"],
            "reference_recall": loc["mask_recall"],
            "reference_predicted_area": loc["predicted_mask_area"],
            "true_patch_area": loc["true_patch_area"],
            "area_ratio_reference": loc["area_ratio"],
            "clean_detected": c_det,
            "clean_confidence": c_conf,
            "clean_iou": c_iou,
            "attacked_detected": a_det,
            "attacked_confidence": a_conf,
            "attacked_iou": a_iou,
            "reference_detected": r_det,
            "reference_confidence": r_conf,
            "reference_iou": r_iou,
            "reference_recovered": recovered,
            "reference_control_preserved": preserved,
            "inference_time_sec": elapsed,
            "sample_mode": regen.get("sample_mode"),
            "status": "completed",
        }
        done_json.write_text(json.dumps(rec, indent=2), encoding="utf-8")
        out_rows.append(rec)
        print(
            f"  mask_iou={loc['mask_iou']:.4f} P={loc['mask_precision']:.4f} R={loc['mask_recall']:.4f} "
            f"area={loc['predicted_mask_area']}/{loc['true_patch_area']} restored={r_det} recovered={recovered}"
        )

    fields = list(out_rows[0].keys()) if out_rows else []
    csv_path = assert_isolated(DIAG / "results" / "per_image_results.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in out_rows:
            flat = dict(row)
            if isinstance(flat.get("kmeans"), dict):
                flat["kmeans"] = json.dumps(flat["kmeans"])
            w.writerow(flat)

    succ = [x for x in out_rows if x["case_type"] == "attack_success"]
    ctrl = [x for x in out_rows if x["case_type"] == "control"]
    nrec = sum(1 for x in succ if x["reference_recovered"])
    npre = sum(1 for x in ctrl if x["reference_control_preserved"])
    ious = [float(x["reference_mask_iou"]) for x in out_rows]
    summary = {
        "note": "Paper/source-faithful Automatic DisPatch on testing mit02 diagnostic subset — NOT paper DRR.",
        "official_commit": OFFICIAL_COMMIT,
        "n_images": len(out_rows),
        "attack_success_recovered": f"{nrec} / {len(succ)}",
        "controls_preserved": f"{npre} / {len(ctrl)}",
        "mean_mask_iou": float(np.mean(ious)) if ious else None,
        "median_mask_iou": float(np.median(ious)) if ious else None,
        "checkerboard_validation": val,
    }
    (DIAG / "results" / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "PAPER-FAITHFUL REFERENCE AUTOMATIC (isolated)",
        f"commit={OFFICIAL_COMMIT}",
        f"N={NUM_GRIDS} steps={DIFFUSION_STEPS} resolution={RESOLUTION}",
        f"Gaussian kernel={gaussian_kernel_size(RESOLUTION, NUM_GRIDS)} sigma=1.0 MORPH_OPEN=3x3",
        f"Attack-success recovered: {nrec} / {len(succ)}  (diagnostic — NOT paper metric)",
        f"Controls preserved: {npre} / {len(ctrl)}",
        f"Mean mask IoU: {summary['mean_mask_iou']}",
        "",
    ]
    for x in out_rows:
        lines.append(
            f"  {x['filename']}  {x['target_class']}  {x['case_type']}  "
            f"mask_iou={x['reference_mask_iou']:.4f} P={x['reference_precision']:.4f} "
            f"R={x['reference_recall']:.4f} area={x['reference_predicted_area']}/{x['true_patch_area']} "
            f"det={x['reference_detected']} rec={x['reference_recovered']}"
        )
    (DIAG / "results" / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
