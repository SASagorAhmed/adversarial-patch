#!/usr/bin/env python
"""Known-Location LDM branch — testing_mitigations/mitigation_02/known_location.

Receives saved true patch coordinates. Does NOT use L2/KMeans localization.
Writes ONLY under this Known Location branch. LDM input is the attacked image.
"""
from __future__ import annotations

import gc
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image

BRANCH = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
PROJECT = Path(r"D:\project CS\DISPATCH-Defense")
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from config.dispatch_config import DIFFUSION_STEPS, DISPATCH_SEED, RESOLUTION, seed_everything  # noqa: E402
from scripts.ldm_inpaint_adapter import LDMInpaintAdapter  # noqa: E402
from scripts.run_hyper_yolo_adapter import load_hyper_yolo_model  # noqa: E402

from common_lib import (  # noqa: E402
    FROZEN_MIT02,
    TESTING_ROOT,
    assert_testing_output,
    copy_branch_sources,
    detect,
    ensure_dirs,
    expand_rect,
    hash_frozen,
    load_csv,
    load_selected_from_frozen,
    make_grid,
    make_rect_mask,
    overlay_mask,
    patch_similarity,
    preflight_ok,
    print_preflight,
    restore_known_at_original,
    save_jpg,
    save_png,
    write_csv,
    write_json,
    yn,
    yolo_predict,
)

EXPAND = 4
OUTSIDE_FAIL = 1.0

KNOWN_DIRS = [
    BRANCH / "masks" / "exact_mask",
    BRANCH / "masks" / "expanded_4px",
    BRANCH / "neutralized_inputs",
    BRANCH / "restored_images" / "exact_mask",
    BRANCH / "restored_images" / "expanded_4px",
    BRANCH / "results" / "ldm_outputs" / "exact_mask",
    BRANCH / "results" / "ldm_outputs" / "expanded_4px",
    BRANCH / "results" / "mask_overlays" / "exact_mask",
    BRANCH / "results" / "mask_overlays" / "expanded_4px",
    BRANCH / "results" / "clean_predictions" / "images",
    BRANCH / "results" / "clean_predictions" / "labels",
    BRANCH / "results" / "attacked_predictions" / "images",
    BRANCH / "results" / "attacked_predictions" / "labels",
    BRANCH / "results" / "restored_predictions" / "exact_mask" / "images",
    BRANCH / "results" / "restored_predictions" / "exact_mask" / "labels",
    BRANCH / "results" / "restored_predictions" / "expanded_4px" / "images",
    BRANCH / "results" / "restored_predictions" / "expanded_4px" / "labels",
    BRANCH / "results" / "verification" / "exact_mask",
    BRANCH / "results" / "verification" / "expanded_4px",
    BRANCH / "results" / "verification" / "comparison",
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
    ensure_dirs(KNOWN_DIRS)
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
    print(f"steps={DIFFUSION_STEPS} seed={DISPATCH_SEED} expand={EXPAND}")
    print("LDM input: ATTACKED images only. Class matching: per-image target_class (not car-only).")

    rows = []
    existing_csv = BRANCH / "results" / "per_image_results.csv"
    if existing_csv.is_file():
        prev = load_csv(existing_csv)
        rows.extend(prev)
        done = {r["image_id"] for r in prev if str(r.get("status")) == "completed"}
        print(f"Resuming known_location; already completed: {sorted(done)}")
        selected = [s for s in selected if s["image_id"] not in done]

    adapter = None
    yolo = None
    if selected:
        seed_everything(DISPATCH_SEED)
        adapter = LDMInpaintAdapter(steps=DIFFUSION_STEPS, seed=DISPATCH_SEED)
        yolo = load_hyper_yolo_model()

    for s in selected:
        fn = s["filename"]
        stem = Path(fn).stem
        print(f"\n=== KNOWN LOCATION {fn} ({s['case_type']} {s['target_class']}) ===")
        attacked = np.array(Image.open(BRANCH / "source_attacked_images" / fn).convert("RGB"))
        clean = np.array(Image.open(BRANCH / "source_clean_images" / fn).convert("RGB"))
        h, w = attacked.shape[:2]
        gt = [s["gt_x"], s["gt_y"], s["gt_x"] + s["gt_width"], s["gt_y"] + s["gt_height"]]
        x1, y1, x2, y2 = s["patch_x1"], s["patch_y1"], s["patch_x2"], s["patch_y2"]
        true_mask = make_rect_mask(h, w, x1, y1, x2, y2)
        ex4 = expand_rect(x1, y1, x2, y2, EXPAND, w, h)
        exp_mask = make_rect_mask(h, w, *ex4)
        save_png(true_mask, BRANCH / "masks" / "exact_mask" / f"{stem}.png", mode="L")
        save_png(exp_mask, BRANCH / "masks" / "expanded_4px" / f"{stem}.png", mode="L")
        save_jpg(overlay_mask(attacked, true_mask, (255, 0, 0)), BRANCH / "results" / "mask_overlays" / "exact_mask" / fn)
        save_jpg(overlay_mask(attacked, exp_mask, (0, 180, 255)), BRANCH / "results" / "mask_overlays" / "expanded_4px" / fn)

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

        seed_everything(DISPATCH_SEED)
        t0 = time.perf_counter()
        kex = restore_known_at_original(adapter, attacked, true_mask, DIFFUSION_STEPS, RESOLUTION)
        t_kex = time.perf_counter() - t0
        seed_everything(DISPATCH_SEED)
        t0 = time.perf_counter()
        k4 = restore_known_at_original(adapter, attacked, exp_mask, DIFFUSION_STEPS, RESOLUTION)
        t_k4 = time.perf_counter() - t0

        save_jpg(kex["ldm_up"], BRANCH / "results" / "ldm_outputs" / "exact_mask" / fn)
        save_jpg(k4["ldm_up"], BRANCH / "results" / "ldm_outputs" / "expanded_4px" / fn)
        save_jpg(kex["restored"], BRANCH / "restored_images" / "exact_mask" / fn)
        save_jpg(k4["restored"], BRANCH / "restored_images" / "expanded_4px" / fn)

        kex_fail = kex["outside_mask_mae"] > OUTSIDE_FAIL
        k4_fail = k4["outside_mask_mae"] > OUTSIDE_FAIL

        kex_pred = yolo_predict(
            yolo, BRANCH / "restored_images" / "exact_mask" / fn,
            BRANCH / "results" / "restored_predictions" / "exact_mask" / "images",
            BRANCH / "results" / "restored_predictions" / "exact_mask" / "labels", fn,
        )
        k4_pred = yolo_predict(
            yolo, BRANCH / "restored_images" / "expanded_4px" / fn,
            BRANCH / "results" / "restored_predictions" / "expanded_4px" / "images",
            BRANCH / "results" / "restored_predictions" / "expanded_4px" / "labels", fn,
        )
        kex_det, kex_conf, kex_iou = detect(kex_pred["predictions"], s["target_class"], gt)
        k4_det, k4_conf, k4_iou = detect(k4_pred["predictions"], s["target_class"], gt)
        if kex_fail:
            kex_det = False
        if k4_fail:
            k4_det = False

        kex_rec = s["case_type"] == "attack_success" and clean_det and (not att_det) and kex_det
        k4_rec = s["case_type"] == "attack_success" and clean_det and (not att_det) and k4_det
        kex_pres = s["case_type"] == "control" and att_det and kex_det
        k4_pres = s["case_type"] == "control" and att_det and k4_det
        kex_reg = s["case_type"] == "control" and att_det and (not kex_det)
        k4_reg = s["case_type"] == "control" and att_det and (not k4_det)
        kex_sim = patch_similarity(clean, kex["restored"], true_mask)
        k4_sim = patch_similarity(clean, k4["restored"], true_mask)

        make_grid(
            [
                ("CLEAN — EVALUATION ONLY", Image.fromarray(clean)),
                ("ATTACKED", Image.fromarray(attacked)),
                ("EXACT TRUE MASK", Image.fromarray(overlay_mask(attacked, true_mask))),
                ("RAW LDM — NOT FINAL", Image.fromarray(kex["ldm_up"])),
                ("FINAL EXACT RESTORED", Image.fromarray(kex["restored"])),
                ("HYPER-YOLO RESULT", Image.open(kex_pred["viz"])),
            ],
            BRANCH / "results" / "verification" / "exact_mask" / fn,
            f"{fn}  KNOWN EXACT  {s['target_class']}  {s['case_type']}",
            cols=3,
        )
        make_grid(
            [
                ("CLEAN — EVALUATION ONLY", Image.fromarray(clean)),
                ("ATTACKED", Image.fromarray(attacked)),
                ("EXPANDED +4PX MASK", Image.fromarray(overlay_mask(attacked, exp_mask, (0, 180, 255)))),
                ("RAW LDM — NOT FINAL", Image.fromarray(k4["ldm_up"])),
                ("FINAL +4PX RESTORED", Image.fromarray(k4["restored"])),
                ("HYPER-YOLO RESULT", Image.open(k4_pred["viz"])),
            ],
            BRANCH / "results" / "verification" / "expanded_4px" / fn,
            f"{fn}  KNOWN +4PX  {s['target_class']}  {s['case_type']}",
            cols=3,
        )
        make_grid(
            [
                ("ATTACKED", Image.fromarray(attacked)),
                ("EXACT MASK", Image.fromarray(overlay_mask(attacked, true_mask))),
                ("FINAL EXACT RESTORED", Image.fromarray(kex["restored"])),
                ("EXACT HYPER-YOLO", Image.open(kex_pred["viz"])),
                ("+4PX MASK", Image.fromarray(overlay_mask(attacked, exp_mask, (0, 180, 255)))),
                ("FINAL +4PX RESTORED", Image.fromarray(k4["restored"])),
                ("+4PX HYPER-YOLO", Image.open(k4_pred["viz"])),
            ],
            BRANCH / "results" / "verification" / "comparison" / fn,
            f"{fn}  EXACT vs +4PX  {s['target_class']}  {s['case_type']}",
            cols=4,
        )

        err = ""
        if kex_fail:
            err += f"known_exact compositing_validation_failed mae={kex['outside_mask_mae']}; "
        if k4_fail:
            err += f"known_expand4 compositing_validation_failed mae={k4['outside_mask_mae']}"

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
            "known_exact_detected": kex_det,
            "known_exact_confidence": kex_conf,
            "known_exact_iou": kex_iou,
            "known_exact_recovered": kex_rec,
            "known_exact_control_preserved": kex_pres,
            "known_exact_control_regression": kex_reg,
            "known_exact_outside_mask_mae": kex["outside_mask_mae"],
            "known_expand4_detected": k4_det,
            "known_expand4_confidence": k4_conf,
            "known_expand4_iou": k4_iou,
            "known_expand4_recovered": k4_rec,
            "known_expand4_control_preserved": k4_pres,
            "known_expand4_control_regression": k4_reg,
            "known_expand4_outside_mask_mae": k4["outside_mask_mae"],
            "known_exact_patch_mae_vs_clean_eval_only": kex_sim["mae"],
            "known_expand4_patch_mae_vs_clean_eval_only": k4_sim["mae"],
            "known_exact_patch_psnr_vs_clean_eval_only": kex_sim["psnr"],
            "known_expand4_patch_psnr_vs_clean_eval_only": k4_sim["psnr"],
            "known_exact_patch_ssim_vs_clean_eval_only": kex_sim["ssim"],
            "known_expand4_patch_ssim_vs_clean_eval_only": k4_sim["ssim"],
            "exact_inference_time_sec": t_kex,
            "expanded_inference_time_sec": t_k4,
            "compositing_validation_failed": bool(kex_fail or k4_fail),
            "manual_visual_review_required": True,
            "status": "completed",
            "error_message": err,
        })
        write_csv(BRANCH / "results" / "per_image_results.csv", rows)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    succ = [r for r in rows if r["case_type"] == "attack_success"]
    ctrl = [r for r in rows if r["case_type"] == "control"]
    kex_n = sum(1 for r in succ if yn(r["known_exact_recovered"]))
    k4_n = sum(1 for r in succ if yn(r["known_expand4_recovered"]))
    kex_ctrl_n = sum(1 for r in ctrl if yn(r["known_exact_control_preserved"]))
    k4_ctrl_n = sum(1 for r in ctrl if yn(r["known_expand4_control_preserved"]))
    kex_ctrl = f"{kex_ctrl_n} / {len(ctrl)}" if ctrl else "n/a"
    k4_ctrl = f"{k4_ctrl_n} / {len(ctrl)}" if ctrl else "n/a"

    def rec_by_cls(key: str, cls: str) -> str:
        sub = [r for r in succ if str(r["target_class"]).lower() == cls]
        n = sum(1 for r in sub if yn(r[key]))
        return f"{n} / {len(sub)}"

    outside_ok = all(
        float(r["known_exact_outside_mask_mae"]) <= OUTSIDE_FAIL and float(r["known_expand4_outside_mask_mae"]) <= OUTSIDE_FAIL
        for r in rows
    )
    summary = {
        "branch": "known_location",
        "n_images": len(rows),
        "known_exact_recovered": f"{kex_n} / {len(succ)}",
        "known_expand4_recovered": f"{k4_n} / {len(succ)}",
        "person_exact_recovered": rec_by_cls("known_exact_recovered", "person"),
        "car_exact_recovered": rec_by_cls("known_exact_recovered", "car"),
        "person_expand4_recovered": rec_by_cls("known_expand4_recovered", "person"),
        "car_expand4_recovered": rec_by_cls("known_expand4_recovered", "car"),
        "note": "This is a 10-image diagnostic subset — NOT final DISPATCH DRR. Restoration is class-agnostic; attack_02 only contains person+car.",
        "control_exact_preserved": kex_ctrl,
        "control_expand4_preserved": k4_ctrl,
        "outside_mask_integrity": "PASS" if outside_ok else "FAIL",
        "ldm_input": "attacked_image_only",
        "manual_visual_review_required": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    write_json(BRANCH / "results" / "summary.json", summary)
    txt = f"""TESTING MITIGATION 02 — KNOWN LOCATION LDM
===========================================

This branch is fully self-contained.
True saved patch coordinates WERE supplied.
Automatic L2/KMeans localization was NOT used.
LDM input = ATTACKED / PATCHED image only (clean is evaluation-only).
Target matching uses each image's own target_class (not hardcoded car/person).

Images: {len(rows)}
Attack-success diagnostic cases: {len(succ)}
Controls: {len(ctrl)}

Victim: Hyper-YOLO-N  conf=0.25 iou=0.70 imgsz=640 device=cpu
Diffusion: CompVis LDM inpainting  DDIM steps=5  seed={DISPATCH_SEED}
Clean pixels used for restoration: NO

Known Exact recovered: {kex_n} / {len(succ)}   (diagnostic subset — NOT DISPATCH DRR)
  person: {rec_by_cls("known_exact_recovered", "person")}
  car: {rec_by_cls("known_exact_recovered", "car")}
Known Exact control preserved: {kex_ctrl}

Known +4px recovered: {k4_n} / {len(succ)}   (diagnostic subset — NOT DISPATCH DRR)
  person: {rec_by_cls("known_expand4_recovered", "person")}
  car: {rec_by_cls("known_expand4_recovered", "car")}
Known +4px control preserved: {k4_ctrl}

Outside-mask integrity: {'PASS' if outside_ok else 'FAIL'}
""" + "\n".join(
        f"  {r['filename']} exact_mae={float(r['known_exact_outside_mask_mae']):.6f}  "
        f"expand4_mae={float(r['known_expand4_outside_mask_mae']):.6f}"
        for r in rows
    ) + "\n\nPer image:\n" + "\n".join(
        f"  {r['filename']}  {r['target_class']}  patch={r['patch_size']}  "
        f"exact_det={r['known_exact_detected']} conf={r['known_exact_confidence']} iou={r['known_exact_iou']} rec={r['known_exact_recovered']}  |  "
        f"+4_det={r['known_expand4_detected']} conf={r['known_expand4_confidence']} iou={r['known_expand4_iou']} rec={r['known_expand4_recovered']}"
        for r in rows
    ) + """

=====================================================
KNOWN LOCATION FOLDER GUIDE
=====================================================

source_clean_images
= clean evaluation references

source_attacked_images
= actual Known Location LDM inputs

source_attack_data
= source attack metadata for these ten cases

masks/exact_mask
= exact saved patch location

masks/expanded_4px
= true location expanded by four pixels

results/ldm_outputs
= RAW INTERMEDIATE LDM output
= NOT final

restored_images/exact_mask
= FINAL exact-mask restored images

restored_images/expanded_4px
= FINAL +4px restored images

results/restored_predictions
= Hyper-YOLO predictions from FINAL restored images

neutralized_inputs
= empty compatibility folder (attacked image + known mask used directly)
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
    print(f"real mitigations/mitigation_02 exists: {FROZEN_MIT02.exists()}")
    print(f"TESTING_ROOT={TESTING_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
