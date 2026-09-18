#!/usr/bin/env python
"""Automatic DISPATCH branch — testing_mitigations/mitigation_01/automatic.

Does NOT receive true patch coordinates during localization.
Writes ONLY under this Automatic branch.
"""
from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

BRANCH = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
PROJECT = Path(r"D:\project CS\DISPATCH-Defense")
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from config.dispatch_config import (  # noqa: E402
    CHECKERBOARD_GRID_N,
    DIFFUSION_STEPS,
    DISPATCH_SEED,
    RESOLUTION,
    seed_everything,
)
from scripts.combine_regeneration import combine_regeneration  # noqa: E402
from scripts.compute_difference_map import compute_l2_difference, smooth_difference, to_vis_png  # noqa: E402
from scripts.evaluate_dispatch import localization_metrics  # noqa: E402
from scripts.generate_checkerboard_masks import generate_checkerboard_masks, validate_masks  # noqa: E402
from scripts.ldm_inpaint_adapter import LDMInpaintAdapter  # noqa: E402
from scripts.predict_adversarial_mask import predict_adversarial_mask  # noqa: E402
from scripts.rectify_image import rectify_image  # noqa: E402
from scripts.run_hyper_yolo_adapter import load_hyper_yolo_model  # noqa: E402

from common_lib import (  # noqa: E402
    FROZEN_MIT02,
    TESTING_ROOT,
    assert_testing_output,
    copy_branch_sources,
    detect,
    ensure_dirs,
    hash_frozen,
    load_selected_from_frozen,
    make_grid,
    make_rect_mask,
    overlay_mask,
    patch_similarity,
    preflight_ok,
    print_preflight,
    save_jpg,
    save_png,
    sha256_file,
    write_csv,
    write_json,
    yolo_predict,
)

AUTO_DIRS = [
    BRANCH / "masks" / "checkerboard_mask_0",
    BRANCH / "masks" / "checkerboard_mask_1",
    BRANCH / "masks" / "predicted_masks",
    BRANCH / "masks" / "true_patch_masks_evaluation_only",
    BRANCH / "neutralized_inputs",
    BRANCH / "restored_images",
    BRANCH / "results" / "regenerated_images",
    BRANCH / "results" / "difference_maps" / "raw",
    BRANCH / "results" / "difference_maps" / "smoothed",
    BRANCH / "results" / "mask_overlays",
    BRANCH / "results" / "clean_predictions" / "images",
    BRANCH / "results" / "clean_predictions" / "labels",
    BRANCH / "results" / "attacked_predictions" / "images",
    BRANCH / "results" / "attacked_predictions" / "labels",
    BRANCH / "results" / "restored_predictions" / "images",
    BRANCH / "results" / "restored_predictions" / "labels",
    BRANCH / "results" / "verification",
    BRANCH / "results" / "comparison",
    BRANCH / "source_attack_data",
    BRANCH / "source_attacked_images",
    BRANCH / "source_clean_images",
    BRANCH / "scripts",
    BRANCH / "config",
    BRANCH / "logs",
]


def main() -> int:
    assert_testing_output(BRANCH)
    if FROZEN_MIT02.exists():
        print("STOP: mitigations/mitigation_02 exists")
        return 2
    preflight_ok()
    ensure_dirs(AUTO_DIRS)
    before = hash_frozen()
    write_json(BRANCH / "logs" / "frozen_mitigation01_hashes_before.json", before)

    selected = load_selected_from_frozen()
    copy_branch_sources(BRANCH, selected)
    write_csv(
        BRANCH / "results" / "selected_images.csv",
        selected,
        [
            "image_id", "filename", "target_class", "case_type", "selection_reason",
            "image_width", "image_height", "gt_x", "gt_y", "gt_width", "gt_height",
            "patch_x1", "patch_y1", "patch_x2", "patch_y2", "patch_size",
            "patch_scale", "source_attack_id",
            "previous_mitigation01_recovered", "previous_mitigation01_mask_iou",
        ],
    )
    print_preflight()
    print(f"checkerboard_N={CHECKERBOARD_GRID_N} steps={DIFFUSION_STEPS} seed={DISPATCH_SEED}")

    seed_everything(DISPATCH_SEED)
    adapter = LDMInpaintAdapter(steps=DIFFUSION_STEPS, seed=DISPATCH_SEED)
    yolo = load_hyper_yolo_model()
    rows = []

    for s in selected:
        fn = s["filename"]
        stem = Path(fn).stem
        print(f"\n=== AUTOMATIC {fn} ({s['case_type']} {s['target_class']}) ===")
        attacked = np.array(Image.open(BRANCH / "source_attacked_images" / fn).convert("RGB"))
        clean = np.array(Image.open(BRANCH / "source_clean_images" / fn).convert("RGB"))
        h, w = attacked.shape[:2]
        gt = [s["gt_x"], s["gt_y"], s["gt_x"] + s["gt_width"], s["gt_y"] + s["gt_height"]]
        x1, y1, x2, y2 = s["patch_x1"], s["patch_y1"], s["patch_x2"], s["patch_y2"]
        true_mask = make_rect_mask(h, w, x1, y1, x2, y2)
        save_png(true_mask, BRANCH / "masks" / "true_patch_masks_evaluation_only" / f"{stem}.png", mode="L")

        clean_pred = yolo_predict(
            yolo, BRANCH / "source_clean_images" / fn,
            BRANCH / "results" / "clean_predictions" / "images",
            BRANCH / "results" / "clean_predictions" / "labels", fn,
        )
        att_pred = yolo_predict(
            yolo, BRANCH / "source_attacked_images" / fn,
            BRANCH / "results" / "attacked_predictions" / "images",
            BRANCH / "results" / "attacked_predictions" / "labels", fn,
        )
        clean_det, clean_conf, clean_iou = detect(clean_pred["predictions"], s["target_class"], gt)
        att_det, att_conf, att_iou = detect(att_pred["predictions"], s["target_class"], gt)

        t0 = time.perf_counter()
        seed_everything(DISPATCH_SEED)
        proc = np.array(Image.fromarray(attacked).resize((RESOLUTION, RESOLUTION), Image.Resampling.LANCZOS))
        m0, m1 = generate_checkerboard_masks(RESOLUTION, RESOLUTION, CHECKERBOARD_GRID_N)
        validate_masks(m0, m1)
        save_png((m0 * 255).astype(np.uint8), BRANCH / "masks" / "checkerboard_mask_0" / f"{stem}.png", mode="L")
        save_png((m1 * 255).astype(np.uint8), BRANCH / "masks" / "checkerboard_mask_1" / f"{stem}.png", mode="L")
        pass0 = adapter.inpaint(proc, m0, steps=DIFFUSION_STEPS)
        pass1 = adapter.inpaint(proc, m1, steps=DIFFUSION_STEPS)
        full512 = combine_regeneration(pass0, pass1, m0, m1)
        regen_orig = np.array(Image.fromarray(full512).resize((w, h), Image.Resampling.LANCZOS))
        save_jpg(regen_orig, BRANCH / "results" / "regenerated_images" / fn)

        raw = compute_l2_difference(proc, full512)
        sm = smooth_difference(raw)
        save_png(to_vis_png(raw), BRANCH / "results" / "difference_maps" / "raw" / f"{stem}.png", mode="L")
        save_png(to_vis_png(sm), BRANCH / "results" / "difference_maps" / "smoothed" / f"{stem}.png", mode="L")
        adv512, _km = predict_adversarial_mask(sm)
        adv_orig = np.array(Image.fromarray(adv512, mode="L").resize((w, h), Image.Resampling.NEAREST))
        save_png(adv_orig, BRANCH / "masks" / "predicted_masks" / f"{stem}.png", mode="L")
        save_jpg(overlay_mask(attacked, adv_orig, (255, 220, 0)), BRANCH / "results" / "mask_overlays" / fn)

        loc = localization_metrics(adv_orig, true_mask)
        write_json(BRANCH / "results" / f"{stem}_localization_eval.json", {
            "note": "True patch compared AFTER predicted mask was frozen. Evaluation only.",
            "image_id": s["image_id"],
            **loc,
        })

        rect512 = rectify_image(proc, full512, adv512)
        auto_rest = np.array(Image.fromarray(rect512).resize((w, h), Image.Resampling.LANCZOS))
        save_jpg(auto_rest, BRANCH / "restored_images" / fn)
        t_auto = time.perf_counter() - t0

        rest_pred = yolo_predict(
            yolo, BRANCH / "restored_images" / fn,
            BRANCH / "results" / "restored_predictions" / "images",
            BRANCH / "results" / "restored_predictions" / "labels", fn,
        )
        rest_det, rest_conf, rest_iou = detect(rest_pred["predictions"], s["target_class"], gt)
        recovered = s["case_type"] == "attack_success" and clean_det and (not att_det) and rest_det
        preserved = s["case_type"] == "control" and att_det and rest_det
        regression = s["case_type"] == "control" and att_det and (not rest_det)
        sim = patch_similarity(clean, auto_rest, true_mask)

        make_grid(
            [
                ("CLEAN — EVALUATION ONLY", Image.fromarray(clean)),
                ("ATTACKED", Image.fromarray(attacked)),
                ("TRUE PATCH MASK — EVAL ONLY", Image.fromarray(overlay_mask(attacked, true_mask))),
                ("AUTOMATIC PREDICTED MASK", Image.fromarray(overlay_mask(attacked, adv_orig, (255, 220, 0)))),
                ("REGENERATED — NOT FINAL", Image.fromarray(regen_orig)),
                ("FINAL AUTOMATIC RESTORED", Image.fromarray(auto_rest)),
                ("HYPER-YOLO RESULT", Image.open(rest_pred["viz"])),
            ],
            BRANCH / "results" / "verification" / fn,
            f"{fn}  {s['target_class']}  {s['case_type']}  |  AUTOMATIC DISPATCH (testing; NOT final DRR)",
            cols=4,
        )

        rows.append({
            "image_id": s["image_id"],
            "filename": fn,
            "target_class": s["target_class"],
            "case_type": s["case_type"],
            "image_width": w,
            "image_height": h,
            "gt_x": s["gt_x"],
            "gt_y": s["gt_y"],
            "gt_width": s["gt_width"],
            "gt_height": s["gt_height"],
            "patch_x1": x1, "patch_y1": y1, "patch_x2": x2, "patch_y2": y2,
            "patch_size": s["patch_size"],
            "clean_detected": clean_det,
            "clean_confidence": clean_conf,
            "clean_iou": clean_iou,
            "attacked_detected": att_det,
            "attacked_confidence": att_conf,
            "attacked_iou": att_iou,
            "automatic_mask_iou": loc["mask_iou"],
            "automatic_mask_precision": loc["mask_precision"],
            "automatic_mask_recall": loc["mask_recall"],
            "predicted_mask_area": loc["predicted_mask_area"],
            "true_patch_area": loc["true_patch_area"],
            "intersection_area": loc["mask_intersection"],
            "restored_detected": rest_det,
            "restored_confidence": rest_conf,
            "restored_iou": rest_iou,
            "automatic_recovered": recovered,
            "automatic_control_preserved": preserved,
            "automatic_control_regression": regression,
            "patch_mae_vs_clean_eval_only": sim["mae"],
            "patch_mse_vs_clean_eval_only": sim["mse"],
            "patch_psnr_vs_clean_eval_only": sim["psnr"],
            "patch_ssim_vs_clean_eval_only": sim["ssim"],
            "inference_time_sec": t_auto,
            "manual_visual_review_required": True,
            "status": "completed",
            "error_message": "",
        })
        write_csv(BRANCH / "results" / "per_image_results.csv", rows)

    succ = [r for r in rows if r["case_type"] == "attack_success"]
    ctrl = [r for r in rows if r["case_type"] == "control"]
    rec_n = sum(1 for r in succ if r["automatic_recovered"])
    ious = [r["automatic_mask_iou"] for r in rows]
    ctrl_yes = "YES" if ctrl and ctrl[0]["automatic_control_preserved"] else "NO"
    summary = {
        "branch": "automatic",
        "n_images": len(rows),
        "attack_success_recovered": f"{rec_n} / {len(succ)}",
        "diagnostic_subset_recovery_count": rec_n,
        "note": "This is a 5-image diagnostic subset — NOT final DISPATCH DRR.",
        "mean_mask_iou": float(np.mean(ious)),
        "median_mask_iou": float(np.median(ious)),
        "min_mask_iou": float(min(ious)),
        "max_mask_iou": float(max(ious)),
        "control_preserved": ctrl_yes,
        "manual_visual_review_required": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    write_json(BRANCH / "results" / "summary.json", summary)
    txt = f"""TESTING MITIGATION 01 — AUTOMATIC DISPATCH
===========================================

This branch is fully self-contained.
True patch coordinates were NOT supplied during localization.

Images: {len(rows)}
Attack-success diagnostic cases: {len(succ)}
Controls: {len(ctrl)}

Victim: Hyper-YOLO-N  conf=0.25 iou=0.70 imgsz=640 device=cpu
Diffusion: CompVis LDM inpainting  DDIM steps=5  seed={DISPATCH_SEED}
Checkerboard N={CHECKERBOARD_GRID_N}  KMeans k=2
Clean pixels used for restoration: NO

Mean predicted-mask IoU: {np.mean(ious):.6f}
Median predicted-mask IoU: {np.median(ious):.6f}
Min / max mask IoU: {min(ious):.6f} / {max(ious):.6f}

Attack-success cases recovered: {rec_n} / {len(succ)}
  (diagnostic subset recovery count — NOT final DISPATCH DRR)
Control preserved: {ctrl_yes}

Per image:
""" + "\n".join(
        f"  {r['filename']}  {r['target_class']}  {r['case_type']}  patch={r['patch_size']}  "
        f"mask_iou={r['automatic_mask_iou']:.4f}  P={r['automatic_mask_precision']:.4f}  "
        f"R={r['automatic_mask_recall']:.4f}  restored={r['restored_detected']}  "
        f"conf={r['restored_confidence']}  iou={r['restored_iou']}  recovered={r['automatic_recovered']}"
        for r in rows
    ) + """

=====================================================
AUTOMATIC FOLDER GUIDE
=====================================================

source_clean_images
= clean evaluation references

source_attacked_images
= actual Automatic adversarial inputs

source_attack_data
= source attack metadata for these five cases

masks/predicted_masks
= automatically localized suspicious masks

masks/true_patch_masks_evaluation_only
= true patch used ONLY for evaluation

results/regenerated_images
= intermediate whole regenerated images
= NOT final

restored_images
= FINAL AUTOMATIC RESTORED IMAGES

results/restored_predictions
= Hyper-YOLO predictions from FINAL restored images

neutralized_inputs
= empty compatibility folder (no SD2.1 neutralization)
"""
    (BRANCH / "results" / "summary.txt").write_text(txt, encoding="utf-8")
    print(txt)

    after = hash_frozen()
    write_json(BRANCH / "logs" / "frozen_mitigation01_hashes_after.json", after)
    write_json(BRANCH / "logs" / "frozen_hash_compare.json", {"unchanged": before == after, "before": before, "after": after})
    if before != after:
        print("STOP: frozen original mitigation_01 hashes changed")
        return 5
    print("FROZEN original mitigation_01 hashes UNCHANGED: YES")
    print(f"mitigation_02 exists: {FROZEN_MIT02.exists()}")
    print(f"TESTING_ROOT={TESTING_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
