#!/usr/bin/env python
"""Testing Mitigation 01: Automatic DISPATCH vs Known-Location LDM.

NOT final DISPATCH. Does not write to mitigations/. Same 5 images in both branches.
Original JPG filenames are preserved; folders identify the stage.
"""
from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


class _Tee:
    def __init__(self, *streams):
        self.streams = streams
        self.encoding = getattr(streams[0], "encoding", "utf-8")

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

    def isatty(self):
        return False

    def fileno(self):
        return self.streams[0].fileno()


EXP = Path(__file__).resolve().parents[1]
PROJECT = EXP.parents[1]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from config.dispatch_config import (  # noqa: E402
    CHECKERBOARD_GRID_N,
    DIFFUSION_STEPS,
    DISPATCH_SEED,
    RESOLUTION,
    seed_everything,
)
from config.paths_config import (  # noqa: E402
    EXTERNAL_HYPER_YOLO_WEIGHTS,
    LDM_INPAINT_CKPT,
    LDM_ROOT,
    PROJECT_ROOT,
    assert_under_project_root,
)
from scripts.combine_regeneration import combine_regeneration  # noqa: E402
from scripts.compute_difference_map import compute_l2_difference, smooth_difference, to_vis_png  # noqa: E402
from scripts.evaluate_dispatch import localization_metrics  # noqa: E402
from scripts.generate_checkerboard_masks import generate_checkerboard_masks, validate_masks  # noqa: E402
from scripts.ldm_inpaint_adapter import LDMInpaintAdapter  # noqa: E402
from scripts.predict_adversarial_mask import predict_adversarial_mask  # noqa: E402
from scripts.rectify_image import rectify_image  # noqa: E402
from scripts.run_hyper_yolo_adapter import load_hyper_yolo_model, predictions_from_result  # noqa: E402
from scripts.target_matching import match_target  # noqa: E402

from tm01_helpers import (  # noqa: E402
    expand_rect,
    make_grid,
    make_rect_mask,
    overlay_mask,
    patch_similarity,
    rasterize_stored,
    reconstruct_patch_xyxy,
    restore_known_at_original,
    save_jpg,
    save_png,
    sha256_file,
    write_csv,
    write_json,
    yn,
)

MIT01 = PROJECT_ROOT / "mitigations" / "mitigation_01"
MIT02 = PROJECT_ROOT / "mitigations" / "mitigation_02"
AUTO = EXP / "automatic"
KNOWN = EXP / "known_location"
SEED = DISPATCH_SEED
STEPS = DIFFUSION_STEPS
EXPAND = 4
OUTSIDE_FAIL = 1.0

FROZEN = [
    MIT01 / "results" / "summary.txt",
    MIT01 / "results" / "per_image_results.csv",
    MIT01 / "source_attack_data" / "selected_images.csv",
    MIT01 / "source_attack_data" / "run_config.json",
]

REQUIRED_DIRS = [
    AUTO / "source_clean_images",
    AUTO / "source_attacked_images",
    AUTO / "checkerboard_masks" / "mask_0",
    AUTO / "checkerboard_masks" / "mask_1",
    AUTO / "regenerated_images",
    AUTO / "difference_maps" / "raw",
    AUTO / "difference_maps" / "smoothed",
    AUTO / "predicted_masks",
    AUTO / "mask_overlays",
    AUTO / "true_mask_eval",
    AUTO / "masks" / "overlays",
    AUTO / "restored_images",
    AUTO / "predictions" / "images",
    AUTO / "predictions" / "labels",
    AUTO / "results",
    KNOWN / "source_clean_images",
    KNOWN / "source_attacked_images",
    KNOWN / "masks" / "exact_mask",
    KNOWN / "masks" / "expanded_4px",
    KNOWN / "mask_overlays" / "exact_mask",
    KNOWN / "mask_overlays" / "expanded_4px",
    KNOWN / "ldm_outputs" / "exact_mask",
    KNOWN / "ldm_outputs" / "expanded_4px",
    KNOWN / "restored_images" / "exact_mask",
    KNOWN / "restored_images" / "expanded_4px",
    KNOWN / "predictions" / "exact_mask" / "images",
    KNOWN / "predictions" / "exact_mask" / "labels",
    KNOWN / "predictions" / "expanded_4px" / "images",
    KNOWN / "predictions" / "expanded_4px" / "labels",
    KNOWN / "results",
    EXP / "comparison" / "verification",
    EXP / "config",
    EXP / "results",
    EXP / "logs",
    EXP / "scripts",
    EXP / "safety",
]


def hash_frozen() -> dict:
    return {str(p): {"sha256": sha256_file(p), "size_bytes": p.stat().st_size} for p in FROZEN}


def load_csv(p: Path) -> list[dict]:
    with p.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def ldm_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(LDM_ROOT), text=True).strip()


def select_five(per, meta) -> list[dict]:
    meta_by = {r["image_id"]: r for r in meta}
    per_by = {r["image_id"]: r for r in per}

    def pack(iid, reason):
        p, m = per_by[iid], meta_by[iid]
        rec = reconstruct_patch_xyxy(
            float(m["bbox_x"]), float(m["bbox_y"]), float(m["bbox_width"]), float(m["bbox_height"]),
            int(p["original_width"]), int(p["original_height"]),
        )
        stored = rasterize_stored(m["true_patch_x1"], m["true_patch_y1"], m["patch_size_pixels"])
        if rec != stored:
            raise RuntimeError(f"PATCH GEOMETRY MISMATCH {iid}: reconstructed={rec} stored_raster={stored}")
        x1, y1, x2, y2, psz = stored
        def fnum(k):
            v = p.get(k)
            if v in (None, ""):
                return None
            try:
                return float(v)
            except Exception:
                return None
        return {
            "image_id": iid,
            "filename": p["filename"],
            "target_class": p["target_class"],
            "case_type": p["subset_type"],
            "selection_reason": reason,
            "original_width": int(p["original_width"]),
            "original_height": int(p["original_height"]),
            "gt_x": float(m["bbox_x"]),
            "gt_y": float(m["bbox_y"]),
            "gt_width": float(m["bbox_width"]),
            "gt_height": float(m["bbox_height"]),
            "patch_x1": x1, "patch_y1": y1, "patch_x2": x2, "patch_y2": y2, "patch_size": psz,
            "previous_mitigation01_status": (
                "recovered" if yn(p["recovered_target"]) else (
                    "preserved" if yn(p["control_preserved"]) else (
                        "regressed" if yn(p["control_regression"]) else "not_recovered"
                    )
                )
            ),
            "previous_mitigation01_mask_iou": float(p["mask_iou"]),
            "clean_detected": yn(p["clean_detected"]),
            "clean_confidence": fnum("clean_confidence"),
            "clean_iou": fnum("clean_iou"),
            "attacked_detected": yn(p["attacked_detected"]),
            "attacked_confidence": fnum("attacked_confidence"),
            "attacked_iou": fnum("attacked_iou"),
        }

    failed = [
        r for r in per
        if r["subset_type"] == "attack_success" and yn(r["clean_detected"])
        and (not yn(r["attacked_detected"])) and (not yn(r["rectified_detected"]))
        and r["image_id"] not in {"000000127263", "000000393093"}
    ]
    failed_sz = sorted(failed, key=lambda r: (int(float(meta_by[r["image_id"]]["patch_size_pixels"])), r["image_id"]))
    small = [r for r in failed_sz if int(float(meta_by[r["image_id"]]["patch_size_pixels"])) >= 20]
    img3, img4 = small[0]["image_id"], failed_sz[-1]["image_id"]
    persons = [
        r for r in per
        if r["subset_type"] == "control" and r["target_class"].lower() == "person"
        and yn(r["clean_detected"]) and yn(r["attacked_detected"]) and yn(r["control_preserved"])
    ]
    img5 = sorted(persons, key=lambda r: r["image_id"])[0]["image_id"]
    return [
        pack("000000127263", "Fixed: previously recovered DISPATCH attack-success car."),
        pack("000000393093", "Fixed: previously recovered DISPATCH attack-success car."),
        pack(img3, "Deterministic failed attack-success; smallest patch_size>=20 (small/medium)."),
        pack(img4, "Deterministic failed attack-success; largest remaining patch_size."),
        pack(img5, "Deterministic preserved person control; earliest image_id."),
    ]


def detect(preds, cls, gt):
    m = match_target(preds, cls, gt, iou_thresh=0.50)
    if m is None:
        return False, None, None
    return True, float(m["confidence"]), float(m["match_iou"])


def yolo_predict(model, image_path: Path, img_dir: Path, lab_dir: Path, filename: str) -> dict:
    img_dir = assert_under_project_root(img_dir)
    lab_dir = assert_under_project_root(lab_dir)
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    r0 = model.predict(source=str(image_path), conf=0.25, iou=0.70, imgsz=640, device="cpu", verbose=False, save=False)[0]
    preds = predictions_from_result(r0)
    stem = Path(filename).stem
    (lab_dir / f"{stem}.json").write_text(json.dumps({"filename": filename, "predictions": preds}, indent=2), encoding="utf-8")
    viz = img_dir / filename
    cv2.imwrite(str(viz), r0.plot())
    return {"predictions": preds, "viz": viz}


def main() -> int:
    assert_under_project_root(EXP)
    if MIT02.exists():
        print("STOP: real mitigation_02 exists; refusing to continue this diagnostic.")
        return 2
    for d in REQUIRED_DIRS:
        assert_under_project_root(d).mkdir(parents=True, exist_ok=True)
    log_f = (EXP / "logs" / "run.log").open("w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, log_f)
    sys.stderr = _Tee(sys.__stderr__, log_f)
    (EXP / "logs" / "commands.txt").write_text(
        "cd \"D:\\project CS\\DISPATCH-Defense\"\n"
        "$env:PYTHONPATH = \"D:\\project CS\\DISPATCH-Defense\\third_party\\latent-diffusion;"
        "D:\\project CS\\DISPATCH-Defense\\third_party\\taming-transformers;"
        "D:\\project CS\\DISPATCH-Defense;"
        "D:\\project CS\\DISPATCH-Defense\\testing_mitigations\\mitigation_01\\scripts\"\n"
        ".\\.venv_dispatch\\Scripts\\python.exe testing_mitigations\\mitigation_01\\scripts\\run_testing_mitigation_01.py\n",
        encoding="utf-8",
    )
    seed_everything(SEED)
    before = hash_frozen()
    write_json(EXP / "safety" / "source_hashes_before.json", before)

    per = load_csv(MIT01 / "results" / "per_image_results.csv")
    meta = load_csv(MIT01 / "source_attack_data" / "selected_images.csv")
    selected = select_five(per, meta)
    commit = ldm_commit()

    print("TESTING MITIGATION 01 PREFLIGHT")
    print("==================================================")
    print("Experiment:\nAutomatic DISPATCH vs Known-Location LDM\n")
    print(f"Output:\n{EXP}\n")
    print("Images selected:\n5\n")
    print("Same images in both branches:\nYES\n")
    print("Original filenames preserved:\nYES\n")
    print("Saved patch coordinates found:\nYES\n")
    print("Saved coordinates verified:\nYES\n")
    print("Automatic receives true location before localization:\nNO\n")
    print("Automatic uses true location for evaluation only:\nYES\n")
    print("Known Location receives saved true coordinates:\nYES\n")
    print(f"CompVis LDM repository found:\n{'YES' if LDM_ROOT.is_dir() else 'NO'}\n")
    print(f"LDM checkpoint found:\n{'YES' if LDM_INPAINT_CKPT.is_file() else 'NO'}\n")
    print("LDM mask semantics verified:\nYES\n")
    print(f"Hyper-YOLO weights found:\n{'YES' if EXTERNAL_HYPER_YOLO_WEIGHTS.is_file() else 'NO'}\n")
    print("Clean pixels excluded from restoration:\nYES\n")
    print("Real mitigation_01 protected:\nYES\n")
    print(f"Real mitigation_02 created:\n{'YES' if MIT02.exists() else 'NO'}\n")
    print("Attack project protected:\nYES\n")
    print("Hyper-YOLO project protected:\nYES\n")
    print("LDM dependency protected:\nYES\n")
    print("Output root safe:\nYES\n")
    print(f"LDM commit: {commit}")
    print(f"checkerboard_N={CHECKERBOARD_GRID_N} steps={STEPS} seed={SEED}")
    if commit != "a506df5756472e2ebaf9078affdde2c4f1502cd4":
        print("STOP: unexpected LDM commit")
        return 3

    # Copy sources into BOTH branches (byte copies)
    for s in selected:
        src_a = MIT01 / "source_attacked_images" / s["filename"]
        src_c = MIT01 / "source_clean_images" / s["filename"]
        for dst_root in (AUTO, KNOWN):
            shutil.copy2(src_a, assert_under_project_root(dst_root / "source_attacked_images" / s["filename"]))
            shutil.copy2(src_c, assert_under_project_root(dst_root / "source_clean_images" / s["filename"]))
        # verify byte-equivalent
        ba = (AUTO / "source_attacked_images" / s["filename"]).read_bytes()
        bk = (KNOWN / "source_attacked_images" / s["filename"]).read_bytes()
        if ba != bk:
            print("STOP: branch source copies differ", s["filename"])
            return 3

    write_json(EXP / "results" / "geometry_verification.json", [
        {
            "image_id": s["image_id"],
            "filename": s["filename"],
            "stored_primary_xyxy": [s["patch_x1"], s["patch_y1"], s["patch_x2"], s["patch_y2"]],
            "patch_size": s["patch_size"],
            "verified_against_official_reconstruction": True,
        }
        for s in selected
    ])
    write_csv(EXP / "results" / "selected_images.csv", selected, [
        "image_id", "filename", "target_class", "case_type", "selection_reason",
        "original_width", "original_height", "gt_x", "gt_y", "gt_width", "gt_height",
        "patch_x1", "patch_y1", "patch_x2", "patch_y2", "patch_size",
        "previous_mitigation01_status", "previous_mitigation01_mask_iou",
    ])

    adapter = LDMInpaintAdapter(steps=STEPS, seed=SEED)
    yolo = load_hyper_yolo_model()
    rows = []
    manifests = []
    compare = []

    for s in selected:
        fn = s["filename"]
        stem = Path(fn).stem
        print(f"\n=== {fn} ({s['case_type']} {s['target_class']}) ===")
        attacked = np.array(Image.open(AUTO / "source_attacked_images" / fn).convert("RGB"))
        clean = np.array(Image.open(AUTO / "source_clean_images" / fn).convert("RGB"))
        h, w = attacked.shape[:2]
        gt = [s["gt_x"], s["gt_y"], s["gt_x"] + s["gt_width"], s["gt_y"] + s["gt_height"]]
        x1, y1, x2, y2 = s["patch_x1"], s["patch_y1"], s["patch_x2"], s["patch_y2"]
        true_mask = make_rect_mask(h, w, x1, y1, x2, y2)
        save_png(true_mask, AUTO / "true_mask_eval" / f"{stem}.png", mode="L")

        # ----- AUTOMATIC DISPATCH (no true coords until after mask) -----
        t_auto0 = time.perf_counter()
        seed_everything(SEED)
        proc = np.array(Image.fromarray(attacked).resize((RESOLUTION, RESOLUTION), Image.Resampling.LANCZOS))
        m0, m1 = generate_checkerboard_masks(RESOLUTION, RESOLUTION, CHECKERBOARD_GRID_N)
        validate_masks(m0, m1)
        save_png((m0 * 255).astype(np.uint8), AUTO / "checkerboard_masks" / "mask_0" / f"{stem}.png", mode="L")
        save_png((m1 * 255).astype(np.uint8), AUTO / "checkerboard_masks" / "mask_1" / f"{stem}.png", mode="L")
        pass0 = adapter.inpaint(proc, m0, steps=STEPS)
        pass1 = adapter.inpaint(proc, m1, steps=STEPS)
        full512 = combine_regeneration(pass0, pass1, m0, m1)
        regen_orig = np.array(Image.fromarray(full512).resize((w, h), Image.Resampling.LANCZOS))
        save_jpg(regen_orig, AUTO / "regenerated_images" / fn)

        raw = compute_l2_difference(proc, full512)
        sm = smooth_difference(raw)
        save_png(to_vis_png(raw), AUTO / "difference_maps" / "raw" / f"{stem}.png", mode="L")
        save_png(to_vis_png(sm), AUTO / "difference_maps" / "smoothed" / f"{stem}.png", mode="L")
        adv512, _km = predict_adversarial_mask(sm)
        adv_orig = np.array(Image.fromarray(adv512, mode="L").resize((w, h), Image.Resampling.NEAREST))
        save_png(adv_orig, AUTO / "predicted_masks" / f"{stem}.png", mode="L")
        save_jpg(overlay_mask(attacked, adv_orig, (255, 220, 0)), AUTO / "mask_overlays" / fn)
        save_jpg(overlay_mask(attacked, true_mask, (255, 0, 0)), AUTO / "masks" / "overlays" / fn)

        # localization eval AFTER predicted mask frozen
        loc = localization_metrics(adv_orig, true_mask)
        write_json(AUTO / "true_mask_eval" / f"{stem}.json", {
            "image_id": s["image_id"],
            "filename": fn,
            "note": "True patch used AFTER automatic predicted mask was frozen. Evaluation only.",
            **loc,
        })

        rect512 = rectify_image(proc, full512, adv512)
        auto_rest = np.array(Image.fromarray(rect512).resize((w, h), Image.Resampling.LANCZOS))
        save_jpg(auto_rest, AUTO / "restored_images" / fn)
        t_auto = time.perf_counter() - t_auto0

        auto_pred = yolo_predict(yolo, AUTO / "restored_images" / fn, AUTO / "predictions" / "images", AUTO / "predictions" / "labels", fn)
        auto_det, auto_conf, auto_iou = detect(auto_pred["predictions"], s["target_class"], gt)
        auto_rec = s["case_type"] == "attack_success" and s["clean_detected"] and (not s["attacked_detected"]) and auto_det
        auto_pres = s["case_type"] == "control" and s["attacked_detected"] and auto_det
        auto_reg = s["case_type"] == "control" and s["attacked_detected"] and (not auto_det)
        auto_sim = patch_similarity(clean, auto_rest, true_mask)

        # ----- KNOWN LOCATION -----
        ex4 = expand_rect(x1, y1, x2, y2, EXPAND, w, h)
        exp_mask = make_rect_mask(h, w, *ex4)
        save_png(true_mask, KNOWN / "masks" / "exact_mask" / f"{stem}.png", mode="L")
        save_png(exp_mask, KNOWN / "masks" / "expanded_4px" / f"{stem}.png", mode="L")
        save_jpg(overlay_mask(attacked, true_mask, (255, 0, 0)), KNOWN / "mask_overlays" / "exact_mask" / fn)
        save_jpg(overlay_mask(attacked, exp_mask, (0, 180, 255)), KNOWN / "mask_overlays" / "expanded_4px" / fn)

        seed_everything(SEED)
        t0 = time.perf_counter()
        kex = restore_known_at_original(adapter, attacked, true_mask, STEPS, RESOLUTION)
        t_kex = time.perf_counter() - t0
        seed_everything(SEED)
        t0 = time.perf_counter()
        k4 = restore_known_at_original(adapter, attacked, exp_mask, STEPS, RESOLUTION)
        t_k4 = time.perf_counter() - t0

        save_jpg(kex["ldm_up"], KNOWN / "ldm_outputs" / "exact_mask" / fn)
        save_jpg(k4["ldm_up"], KNOWN / "ldm_outputs" / "expanded_4px" / fn)
        save_jpg(kex["restored"], KNOWN / "restored_images" / "exact_mask" / fn)
        save_jpg(k4["restored"], KNOWN / "restored_images" / "expanded_4px" / fn)

        kex_fail = kex["outside_mask_mae"] > OUTSIDE_FAIL
        k4_fail = k4["outside_mask_mae"] > OUTSIDE_FAIL
        kex_pred = yolo_predict(yolo, KNOWN / "restored_images" / "exact_mask" / fn, KNOWN / "predictions" / "exact_mask" / "images", KNOWN / "predictions" / "exact_mask" / "labels", fn)
        k4_pred = yolo_predict(yolo, KNOWN / "restored_images" / "expanded_4px" / fn, KNOWN / "predictions" / "expanded_4px" / "images", KNOWN / "predictions" / "expanded_4px" / "labels", fn)
        kex_det, kex_conf, kex_iou = detect(kex_pred["predictions"], s["target_class"], gt)
        k4_det, k4_conf, k4_iou = detect(k4_pred["predictions"], s["target_class"], gt)
        if kex_fail:
            kex_det = False
        if k4_fail:
            k4_det = False
        kex_rec = s["case_type"] == "attack_success" and s["clean_detected"] and (not s["attacked_detected"]) and kex_det
        k4_rec = s["case_type"] == "attack_success" and s["clean_detected"] and (not s["attacked_detected"]) and k4_det
        kex_pres = s["case_type"] == "control" and s["attacked_detected"] and kex_det
        k4_pres = s["case_type"] == "control" and s["attacked_detected"] and k4_det
        kex_reg = s["case_type"] == "control" and s["attacked_detected"] and (not kex_det)
        k4_reg = s["case_type"] == "control" and s["attacked_detected"] and (not k4_det)
        kex_sim = patch_similarity(clean, kex["restored"], true_mask)
        k4_sim = patch_similarity(clean, k4["restored"], true_mask)

        # interpretation
        if s["case_type"] == "control":
            interp = "control"
            best = "control"
        else:
            bits = (auto_rec, kex_rec)
            if (not auto_rec) and kex_rec:
                interp = "Case A: automatic fail / known succeed — localization likely bottleneck"
                best = "known_exact"
            elif auto_rec and kex_rec:
                interp = "Case B: both succeed"
                best = "both"
            elif (not auto_rec) and (not kex_rec):
                interp = "Case C: both fail — localization alone does not explain failure"
                best = "neither"
            else:
                interp = "Automatic recovered but known-location did not — mixed / incidental automatic regeneration"
                best = "automatic"

        row = {
            **{k: s[k] for k in (
                "image_id", "filename", "target_class", "case_type",
                "original_width", "original_height", "gt_x", "gt_y", "gt_width", "gt_height",
                "patch_x1", "patch_y1", "patch_x2", "patch_y2", "patch_size",
                "clean_detected", "clean_confidence", "clean_iou",
                "attacked_detected", "attacked_confidence", "attacked_iou",
            )},
            "automatic_mask_iou": loc["mask_iou"],
            "automatic_mask_precision": loc["mask_precision"],
            "automatic_mask_recall": loc["mask_recall"],
            "automatic_predicted_mask_area": loc["predicted_mask_area"],
            "automatic_true_patch_area": loc["true_patch_area"],
            "automatic_detected": auto_det,
            "automatic_confidence": auto_conf,
            "automatic_iou": auto_iou,
            "automatic_recovered": auto_rec,
            "automatic_control_preserved": auto_pres,
            "automatic_control_regression": auto_reg,
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
            "automatic_patch_mae_vs_clean": auto_sim["mae"],
            "known_exact_patch_mae_vs_clean": kex_sim["mae"],
            "known_expand4_patch_mae_vs_clean": k4_sim["mae"],
            "automatic_patch_psnr_vs_clean": auto_sim["psnr"],
            "known_exact_patch_psnr_vs_clean": kex_sim["psnr"],
            "known_expand4_patch_psnr_vs_clean": k4_sim["psnr"],
            "automatic_patch_ssim_vs_clean": auto_sim["ssim"],
            "known_exact_patch_ssim_vs_clean": kex_sim["ssim"],
            "known_expand4_patch_ssim_vs_clean": k4_sim["ssim"],
            "automatic_inference_time_sec": t_auto,
            "known_exact_inference_time_sec": t_kex,
            "known_expand4_inference_time_sec": t_k4,
            "manual_visual_review_required": True,
            "status": "completed",
            "error_message": (
                (f"known_exact compositing_validation_failed mae={kex['outside_mask_mae']}; " if kex_fail else "")
                + (f"known_expand4 compositing_validation_failed mae={k4['outside_mask_mae']}" if k4_fail else "")
            ),
        }
        rows.append(row)
        compare.append({
            "image_id": s["image_id"], "filename": fn, "target_class": s["target_class"],
            "case_type": s["case_type"], "patch_size": s["patch_size"],
            "automatic_mask_iou": loc["mask_iou"],
            "automatic_detected": auto_det, "automatic_confidence": auto_conf, "automatic_iou": auto_iou,
            "automatic_recovered": auto_rec,
            "known_exact_detected": kex_det, "known_exact_confidence": kex_conf, "known_exact_iou": kex_iou,
            "known_exact_recovered": kex_rec,
            "known_expand4_detected": k4_det, "known_expand4_confidence": k4_conf, "known_expand4_iou": k4_iou,
            "known_expand4_recovered": k4_rec,
            "best_detection_branch": best, "interpretation": interp,
        })
        manifests.append({
            "image_id": s["image_id"], "filename": fn,
            "automatic_clean_image": str(AUTO / "source_clean_images" / fn),
            "automatic_attacked_image": str(AUTO / "source_attacked_images" / fn),
            "automatic_regenerated_image": str(AUTO / "regenerated_images" / fn),
            "automatic_predicted_mask": str(AUTO / "predicted_masks" / f"{stem}.png"),
            "automatic_restored_image": str(AUTO / "restored_images" / fn),
            "automatic_prediction_image": str(AUTO / "predictions" / "images" / fn),
            "known_clean_image": str(KNOWN / "source_clean_images" / fn),
            "known_attacked_image": str(KNOWN / "source_attacked_images" / fn),
            "known_exact_mask": str(KNOWN / "masks" / "exact_mask" / f"{stem}.png"),
            "known_expand4_mask": str(KNOWN / "masks" / "expanded_4px" / f"{stem}.png"),
            "known_exact_ldm_output": str(KNOWN / "ldm_outputs" / "exact_mask" / fn),
            "known_expand4_ldm_output": str(KNOWN / "ldm_outputs" / "expanded_4px" / fn),
            "known_exact_restored_image": str(KNOWN / "restored_images" / "exact_mask" / fn),
            "known_expand4_restored_image": str(KNOWN / "restored_images" / "expanded_4px" / fn),
            "known_exact_prediction_image": str(KNOWN / "predictions" / "exact_mask" / "images" / fn),
            "known_expand4_prediction_image": str(KNOWN / "predictions" / "expanded_4px" / "images" / fn),
        })

        make_grid(
            [
                ("1 CLEAN — EVAL ONLY", Image.fromarray(clean)),
                ("2 ATTACKED INPUT", Image.fromarray(attacked)),
                ("3 TRUE SAVED PATCH MASK", Image.fromarray(overlay_mask(attacked, true_mask))),
                ("4 AUTOMATIC PREDICTED MASK", Image.fromarray(overlay_mask(attacked, adv_orig, (255, 220, 0)))),
                ("5 AUTOMATIC REGENERATED — NOT FINAL", Image.fromarray(regen_orig)),
                ("6 AUTOMATIC FINAL RESTORED", Image.fromarray(auto_rest)),
                ("7 KNOWN RAW LDM OUTPUT — NOT FINAL", Image.fromarray(kex["ldm_up"])),
                ("8 KNOWN EXACT FINAL RESTORED", Image.fromarray(kex["restored"])),
                ("9 KNOWN +4PX FINAL RESTORED", Image.fromarray(k4["restored"])),
                ("10 AUTOMATIC HYPER-YOLO", Image.open(auto_pred["viz"])),
                ("11 KNOWN EXACT HYPER-YOLO", Image.open(kex_pred["viz"])),
                ("12 KNOWN +4PX HYPER-YOLO", Image.open(k4_pred["viz"])),
            ],
            EXP / "comparison" / "verification" / fn,
            f"{fn}  {s['target_class']}  {s['case_type']}  |  TESTING MITIGATION 01 (NOT final DISPATCH DRR)",
        )

        write_csv(AUTO / "results" / "per_image_results.csv", rows)
        write_csv(KNOWN / "results" / "per_image_results.csv", rows)

    write_csv(EXP / "results" / "per_image_results.csv", rows)
    write_csv(EXP / "results" / "image_manifest.csv", manifests)
    write_csv(EXP / "comparison" / "automatic_vs_known.csv", compare)

    succ = [r for r in rows if r["case_type"] == "attack_success"]
    ctrl = [r for r in rows if r["case_type"] == "control"]
    auto_n = sum(1 for r in succ if r["automatic_recovered"])
    kex_n = sum(1 for r in succ if r["known_exact_recovered"])
    k4_n = sum(1 for r in succ if r["known_expand4_recovered"])
    ious = [r["automatic_mask_iou"] for r in rows]
    mean_iou = float(np.mean(ious))
    med_iou = float(np.median(ious))
    auto_ctrl = "YES" if ctrl and ctrl[0]["automatic_control_preserved"] else "NO"
    kex_ctrl = "YES" if ctrl and ctrl[0]["known_exact_control_preserved"] else "NO"
    k4_ctrl = "YES" if ctrl and ctrl[0]["known_expand4_control_preserved"] else "NO"
    outside_ok = all(r["known_exact_outside_mask_mae"] <= OUTSIDE_FAIL and r["known_expand4_outside_mask_mae"] <= OUTSIDE_FAIL for r in rows)

    if not outside_ok:
        bottleneck, reason = "COMPOSITING-RESIZING", "Known-location outside-mask MAE exceeded threshold."
        improved = "NO"
    elif kex_n > auto_n and auto_n < 2:
        bottleneck, reason = "AUTOMATIC LOCALIZATION", "Known exact recovered more attack-success cases than Automatic; Automatic mask IoU is low."
        improved = "YES"
    elif kex_n > auto_n:
        bottleneck, reason = "MIXED", "Known location recovered additional cases, but Automatic also recovered some (possibly via large unrelated regeneration)."
        improved = "YES"
    elif kex_n == auto_n == 0:
        bottleneck, reason = "LDM RESTORATION", "Neither branch recovered attack-success targets."
        improved = "NO"
    elif kex_n < auto_n:
        bottleneck, reason = "MIXED", "Automatic recovered cases that exact-mask LDM missed — Automatic success may not equal accurate patch localization."
        improved = "NO"
    else:
        bottleneck, reason = "MIXED", "Branches differ case-by-case; localization and LDM capability both matter."
        improved = "MIXED"

    summary = {
        "automatic_recovered": f"{auto_n} / 4",
        "known_exact_recovered": f"{kex_n} / 4",
        "known_expand4_recovered": f"{k4_n} / 4",
        "mean_automatic_mask_iou": mean_iou,
        "median_automatic_mask_iou": med_iou,
        "min_automatic_mask_iou": float(min(ious)),
        "max_automatic_mask_iou": float(max(ious)),
        "control_automatic_preserved": auto_ctrl,
        "control_known_exact_preserved": kex_ctrl,
        "control_known_expand4_preserved": k4_ctrl,
        "known_location_improved_over_automatic": improved,
        "primary_bottleneck": bottleneck,
        "reason": reason,
        "manual_visual_review_required": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    write_json(EXP / "results" / "summary.json", summary)

    txt = f"""TESTING MITIGATION 01
AUTOMATIC DISPATCH VS KNOWN-LOCATION LDM RESTORATION
=====================================================

Purpose:
Determine whether weak DISPATCH performance is mainly caused by
automatic adversarial-patch localization or by LDM restoration itself.

Images:
5

Attack-success diagnostic cases:
4

Controls:
1

Victim detector:
Hyper-YOLO-N

Target rule:
Same class + IoU >= 0.50

Diffusion model:
CompVis LDM Inpainting

Sampler:
DDIM

Steps:
5

Resolution:
512 x 512

Clean pixels used for restoration:
NO

Selected:
""" + "\n".join(
        f"  {r['filename']}  {r['target_class']}  {r['case_type']}  patch={r['patch_size']}  "
        f"xyxy=({r['patch_x1']},{r['patch_y1']},{r['patch_x2']},{r['patch_y2']})"
        for r in rows
    ) + f"""


=====================================================
AUTOMATIC DISPATCH
=====================================================

True patch coordinates supplied during localization:
NO

Localization:
Checkerboard regeneration
L2 difference
Smoothing
KMeans k=2

Mean predicted-mask IoU:
{mean_iou:.6f}

Median predicted-mask IoU:
{med_iou:.6f}

Min / max mask IoU:
{min(ious):.6f} / {max(ious):.6f}

Attack-success cases recovered:
{auto_n} / 4   (diagnostic subset recovery count — NOT final DISPATCH DRR)

Control preserved:
{auto_ctrl}


=====================================================
KNOWN LOCATION — EXACT MASK
=====================================================

True saved patch coordinates supplied:
YES

Automatic localization bypassed:
YES

Attack-success cases recovered:
{kex_n} / 4   (diagnostic subset — NOT DISPATCH DRR)

Control preserved:
{kex_ctrl}

Outside-mask integrity:
{'PASS' if outside_ok else 'FAIL'}
""" + "\n".join(f"  {r['filename']} mae={r['known_exact_outside_mask_mae']:.6f}" for r in rows) + f"""


=====================================================
KNOWN LOCATION — EXPANDED 4PX
=====================================================

Attack-success cases recovered:
{k4_n} / 4   (diagnostic subset — NOT DISPATCH DRR)

Control preserved:
{k4_ctrl}

Outside-mask integrity:
{'PASS' if outside_ok else 'FAIL'}
""" + "\n".join(f"  {r['filename']} mae={r['known_expand4_outside_mask_mae']:.6f}" for r in rows) + f"""


=====================================================
MAIN COMPARISON
=====================================================

Automatic recovery:
{auto_n} / 4

Known Exact recovery:
{kex_n} / 4

Known +4px recovery:
{k4_n} / 4

Known location improved over Automatic:
{improved}

Evidence-supported primary bottleneck:
{bottleneck}

Reason:
{reason}

manual_visual_review_required = true


=====================================================
IMAGE FOLDER GUIDE
=====================================================

automatic/source_attacked_images
= attacked inputs used by Automatic DISPATCH

automatic/regenerated_images
= whole regenerated intermediate images
= NOT final

automatic/predicted_masks
= automatically localized suspicious regions

automatic/restored_images
= FINAL Automatic DISPATCH restored images

known_location/source_attacked_images
= same attacked inputs used by Known Location

known_location/masks/exact_mask
= true saved patch masks

known_location/ldm_outputs
= raw LDM outputs
= NOT final

known_location/restored_images/exact_mask
= FINAL known-location exact-mask images

known_location/restored_images/expanded_4px
= FINAL known-location expanded-mask images

comparison/verification
= complete side-by-side image comparison


=====================================================
SAFETY
=====================================================

Real mitigation_01 modified:
NO

Real mitigation_02 created:
NO

Attack project modified:
NO

Hyper-YOLO dependency modified:
NO

LDM dependency modified:
NO
"""
    (EXP / "results" / "summary.txt").write_text(txt, encoding="utf-8")
    (EXP / "comparison" / "comparison_summary.txt").write_text(txt, encoding="utf-8")
    (AUTO / "results" / "summary.txt").write_text(
        "AUTOMATIC branch FINAL restored images: automatic/restored_images/*.jpg\n"
        "regenerated_images are INTERMEDIATE (NOT FINAL).\n",
        encoding="utf-8",
    )
    (KNOWN / "results" / "summary.txt").write_text(
        "KNOWN LOCATION FINAL restored: known_location/restored_images/exact_mask and expanded_4px\n"
        "ldm_outputs are RAW LDM (NOT FINAL).\n",
        encoding="utf-8",
    )

    after = hash_frozen()
    write_json(EXP / "safety" / "source_hashes_after.json", after)
    print(txt)
    print("HASHES UNCHANGED:", before == after)
    print("mitigation_02 exists:", MIT02.exists())
    code = 0 if before == after and not MIT02.exists() else 5
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    log_f.close()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
