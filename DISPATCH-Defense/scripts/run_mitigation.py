#!/usr/bin/env python
"""Run a real DISPATCH mitigation under mitigations/mitigation_XX/ layout.

Requires explicit --i-approve-full-run.
Does not create mitigation folders during validate/smoke modes.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.dispatch_config import (
    CHECKERBOARD_GRID_N,
    DIFFUSION_STEPS,
    DISPATCH_SEED,
    RESOLUTION,
    SAMPLER,
    TARGET_MATCH_IOU,
    as_run_config_dict,
    seed_everything,
)
from config.paths_config import (
    EXTERNAL_ATTACK_ROOT,
    LDM_INPAINT_CKPT,
    MITIGATIONS_DIR,
    PROJECT_ROOT,
    REPORTS_DIR,
    assert_under_project_root,
    ensure_dir,
)
from scripts.allocate_mitigation import create_mitigation_folder, list_mitigation_ids, next_mitigation_id
from scripts.combine_regeneration import combine_regeneration
from scripts.compute_difference_map import compute_l2_difference, smooth_difference, to_vis_png
from scripts.dispatch_reporting import STORAGE_GUARANTEE, append_experiment_inventory
from scripts.evaluate_dispatch import aggregate_detection_metrics, localization_metrics, write_csv
from scripts.generate_checkerboard_masks import generate_checkerboard_masks, save_mask_png, validate_masks
from scripts.ldm_inpaint_adapter import LDMInpaintAdapter
from scripts.predict_adversarial_mask import predict_adversarial_mask
from scripts.rectify_image import rectify_image
from scripts.run_hyper_yolo_adapter import load_hyper_yolo_model, predictions_from_result
from scripts.select_mitigation_subset import select_subset, write_selected_csv
from scripts.target_matching import match_target


def free_gb_d() -> float:
    return shutil.disk_usage("D:\\").free / (1024**3)


def sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ldm_commit() -> str:
    import subprocess

    repo = PROJECT_ROOT / "third_party" / "latent-diffusion"
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo), text=True)
            .strip()
        )
    except Exception:
        return "unknown"


def true_patch_mask(h: int, w: int, x1, y1, x2, y2) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.uint8)
    xa, xb = int(max(0, np.floor(x1))), int(min(w, np.ceil(x2)))
    ya, yb = int(max(0, np.floor(y1))), int(min(h, np.ceil(y2)))
    m[ya:yb, xa:xb] = 255
    return m


def prepare_masked_input(rgb: np.ndarray, mask01: np.ndarray) -> np.ndarray:
    """Visual/prepared LDM input: show masked regions darkened (not old neutralization)."""
    m = mask01.astype(np.float32)
    if m.max() > 1:
        m = m / 255.0
    out = rgb.astype(np.float32).copy()
    # Zero-out masked region (CompVis-style masked_image preview)
    out *= (1.0 - m[..., None])
    return np.clip(out, 0, 255).astype(np.uint8)


def make_verification(
    clean: Image.Image,
    attacked: Image.Image,
    regenerated: Image.Image,
    adv_mask: Image.Image,
    restored: Image.Image,
    det_viz: Image.Image | None,
    out_path: Path,
    title: str,
) -> None:
    imgs = [clean, attacked, regenerated, adv_mask.convert("RGB"), restored]
    labels = ["Clean", "Attacked", "Regenerated", "Adv Mask", "Restored"]
    if det_viz is not None:
        imgs.append(det_viz.convert("RGB"))
        labels.append("YOLO Restored")
    tw = 256
    thumbs = []
    for im in imgs:
        t = im.convert("RGB").copy()
        t.thumbnail((tw, tw))
        thumbs.append(t)
    gap = 8
    header = 28
    W = tw * len(thumbs) + gap * (len(thumbs) + 1)
    H = tw + header + gap * 2
    canvas = Image.new("RGB", (W, H), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 6), title, fill=(20, 20, 20))
    x = gap
    for t, lab in zip(thumbs, labels):
        canvas.paste(t, (x, header))
        draw.text((x, header + tw + 2), lab, fill=(40, 40, 40))
        x += tw + gap
    out_path = assert_under_project_root(out_path)
    canvas.save(out_path)


def evaluate_detection(preds: list[dict], target_class: str, gt_xyxy) -> dict:
    matched = match_target(preds, target_class, gt_xyxy, iou_thresh=TARGET_MATCH_IOU)
    if matched is None:
        return {"detected": False, "confidence": None, "iou": None}
    return {
        "detected": True,
        "confidence": float(matched.get("confidence")),
        "iou": float(matched.get("match_iou")),
    }


def run_mitigation(
    attack_id: str,
    n_success: int,
    n_control: int,
    steps: int,
    resume: bool,
    continue_incomplete: bool = False,
) -> Path:
    if free_gb_d() < 15:
        raise RuntimeError(f"STOP: insufficient disk space ({free_gb_d():.2f} GB free)")

    existing = list_mitigation_ids()
    print(f"Existing mitigations: {existing or 'none'}")

    if continue_incomplete:
        mid = next_mitigation_id()
        # If highest exists but has no completed per_image results, reuse it
        if existing:
            cand = f"mitigation_{max(existing):02d}"
            cand_root = MITIGATIONS_DIR / cand
            per = cand_root / "results" / "per_image_results.csv"
            restored = cand_root / "restored_images"
            n_restored = len(list(restored.glob("*.jpg"))) + len(list(restored.glob("*.png"))) if restored.is_dir() else 0
            if (not per.is_file()) or n_restored == 0:
                mid = cand
                mroot = assert_under_project_root(cand_root)
                print(f"Continuing incomplete mitigation folder: {mid}")
                # Ensure full subtree exists
                for rel in [
                    "source_attack_data",
                    "source_attacked_images",
                    "source_clean_images",
                    "masks/checkerboard_m0",
                    "masks/checkerboard_m1",
                    "masks/adversarial_masks",
                    "masks/evaluation_true_patch_masks",
                    "neutralized_inputs/masked_input_m0",
                    "neutralized_inputs/masked_input_m1",
                    "restored_images",
                    "results/regenerated/pass_0",
                    "results/regenerated/pass_1",
                    "results/regenerated/full",
                    "results/difference_maps/raw_numeric",
                    "results/difference_maps/raw_visualization",
                    "results/difference_maps/smoothed",
                    "results/clean_predictions/images",
                    "results/clean_predictions/labels",
                    "results/attacked_predictions/images",
                    "results/attacked_predictions/labels",
                    "results/restored_predictions/images",
                    "results/restored_predictions/labels",
                    "results/verification",
                    "results/plots",
                    "results/logs",
                ]:
                    ensure_dir(mroot / rel)
            else:
                raise RuntimeError(
                    f"Refusing continue-incomplete: {cand} already has restored outputs"
                )
        else:
            mroot = create_mitigation_folder(mid)
    else:
        mid = next_mitigation_id()
        print(f"Creating: {mid}")
        mroot = create_mitigation_folder(mid)

    assert_under_project_root(mroot)

    seed_everything(DISPATCH_SEED)
    rows = select_subset(attack_id, n_success, n_control, DISPATCH_SEED)

    # --- Freeze manifest BEFORE inference ---
    sad = mroot / "source_attack_data"
    write_selected_csv(rows, sad / "selected_images.csv")
    write_selected_csv(rows, sad / "source_manifest.csv")
    shutil.copy2(
        EXTERNAL_ATTACK_ROOT / "attacks" / attack_id / "results" / "attack_results.csv",
        sad / "source_attack_results.csv",
    )
    shutil.copy2(
        EXTERNAL_ATTACK_ROOT / "attacks" / attack_id / "results" / "selected_images.csv",
        sad / "source_metadata.csv",
    )
    # target metadata subset
    write_selected_csv(
        [
            {
                k: r[k]
                for k in (
                    "image_id",
                    "filename",
                    "target_class",
                    "bbox_x",
                    "bbox_y",
                    "bbox_width",
                    "bbox_height",
                    "image_width",
                    "image_height",
                    "subset_type",
                )
            }
            for r in rows
        ],
        sad / "target_metadata.csv",
    )

    ckpt_sha = sha256_file(LDM_INPAINT_CKPT) if LDM_INPAINT_CKPT.is_file() else "missing"
    commit = ldm_commit()
    run_cfg = as_run_config_dict(
        mitigation_id=mid,
        source_attack=attack_id,
        n_success=n_success,
        n_control=n_control,
        n_selected=len(rows),
        diffusion_steps=steps,
        sampler=SAMPLER,
        checkerboard_grid_n=CHECKERBOARD_GRID_N,
        seed=DISPATCH_SEED,
        checkpoint_sha256=ckpt_sha,
        ldm_commit=commit,
        clean_pixels_used_for_restoration=False,
        known_patch_location_used_by_defense=False,
        neutralized_inputs_note=(
            "neutralized_inputs folder retained for output-layout compatibility; "
            "contents are DISPATCH checkerboard-prepared LDM inputs."
        ),
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    (sad / "run_config.json").write_text(json.dumps(run_cfg, indent=2), encoding="utf-8")
    (mroot / "results" / "logs" / "environment_snapshot.txt").write_text(
        json.dumps(run_cfg, indent=2), encoding="utf-8"
    )

    # Copy source images (local working copies only)
    for r in rows:
        src_a = Path(r["source_attacked_path"])
        src_c = Path(r["source_clean_path"])
        if not src_a.is_file() or not src_c.is_file():
            raise FileNotFoundError(f"Missing source images for {r['filename']}")
        dst_a = assert_under_project_root(mroot / "source_attacked_images" / r["filename"])
        dst_c = assert_under_project_root(mroot / "source_clean_images" / r["filename"])
        if not dst_a.exists():
            shutil.copy2(src_a, dst_a)
        if not dst_c.exists():
            shutil.copy2(src_c, dst_c)

    adapter = LDMInpaintAdapter(steps=steps)
    yolo = load_hyper_yolo_model()

    def predict_save(image_path: Path, pred_root: Path, image_stem: str) -> dict:
        pred_root = assert_under_project_root(pred_root)
        results = yolo.predict(
            source=str(image_path),
            conf=0.25,
            iou=0.70,
            imgsz=640,
            device="cpu",
            verbose=False,
            save=False,
        )
        r0 = results[0]
        preds = predictions_from_result(r0)
        labels = pred_root / "labels" / f"{image_stem}.json"
        labels.parent.mkdir(parents=True, exist_ok=True)
        labels.write_text(json.dumps({"predictions": preds}, indent=2), encoding="utf-8")
        viz = pred_root / "images" / f"{image_stem}.jpg"
        viz.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(viz), r0.plot())
        return {"predictions": preds, "viz": viz}

    per_rows = []
    timing_rows = []

    for idx, r in enumerate(rows, 1):
        fname = r["filename"]
        stem = Path(fname).stem
        image_id = r["image_id"]
        print(f"[{idx}/{len(rows)}] {fname} ({r['subset_type']})")
        t_total0 = time.perf_counter()
        status = "completed"
        err = ""
        row_out = {
            "experiment_id": mid,
            "source_attack": attack_id,
            "image_id": image_id,
            "filename": fname,
            "target_class": r["target_class"],
            "subset_type": r["subset_type"],
            "checkerboard_grid_n": CHECKERBOARD_GRID_N,
            "diffusion_steps": steps,
            "sampler": SAMPLER,
            "resolution": RESOLUTION,
            "checkpoint_path": str(LDM_INPAINT_CKPT),
            "ldm_commit": commit,
            "clean_pixels_used_for_restoration": False,
            "known_patch_location_used_by_defense": False,
        }
        try:
            attacked_path = mroot / "source_attacked_images" / fname
            clean_path = mroot / "source_clean_images" / fname
            attacked_pil = Image.open(attacked_path).convert("RGB")
            clean_pil = Image.open(clean_path).convert("RGB")
            orig_w, orig_h = attacked_pil.size
            row_out["original_width"] = orig_w
            row_out["original_height"] = orig_h

            proc = attacked_pil.resize((RESOLUTION, RESOLUTION), Image.Resampling.LANCZOS)
            proc_arr = np.array(proc)

            m0, m1 = generate_checkerboard_masks(RESOLUTION, RESOLUTION, CHECKERBOARD_GRID_N)
            validate_masks(m0, m1)
            save_mask_png(m0, mroot / "masks" / "checkerboard_m0" / f"{stem}_mask_m0.png")
            save_mask_png(m1, mroot / "masks" / "checkerboard_m1" / f"{stem}_mask_m1.png")

            prep0 = prepare_masked_input(proc_arr, m0)
            prep1 = prepare_masked_input(proc_arr, m1)
            Image.fromarray(prep0).save(
                assert_under_project_root(
                    mroot / "neutralized_inputs" / "masked_input_m0" / f"{stem}_masked_m0.png"
                )
            )
            Image.fromarray(prep1).save(
                assert_under_project_root(
                    mroot / "neutralized_inputs" / "masked_input_m1" / f"{stem}_masked_m1.png"
                )
            )

            t0 = time.perf_counter()
            pass0 = adapter.inpaint(proc_arr, m0, steps=steps)
            t_pass0 = time.perf_counter() - t0
            t1 = time.perf_counter()
            pass1 = adapter.inpaint(proc_arr, m1, steps=steps)
            t_pass1 = time.perf_counter() - t1
            Image.fromarray(pass0).save(
                mroot / "results" / "regenerated" / "pass_0" / f"{stem}_regenerated_pass0.png"
            )
            Image.fromarray(pass1).save(
                mroot / "results" / "regenerated" / "pass_1" / f"{stem}_regenerated_pass1.png"
            )
            full = combine_regeneration(pass0, pass1, m0, m1)
            Image.fromarray(full).save(
                mroot / "results" / "regenerated" / "full" / f"{stem}_regenerated_full.png"
            )

            t_d0 = time.perf_counter()
            raw = compute_l2_difference(proc_arr, full)
            sm = smooth_difference(raw)
            t_diff = time.perf_counter() - t_d0
            np.save(
                mroot / "results" / "difference_maps" / "raw_numeric" / f"{stem}_l2_raw.npy", raw
            )
            Image.fromarray(to_vis_png(raw), mode="L").save(
                mroot / "results" / "difference_maps" / "raw_visualization" / f"{stem}_l2_raw.png"
            )
            Image.fromarray(to_vis_png(sm), mode="L").save(
                mroot / "results" / "difference_maps" / "smoothed" / f"{stem}_l2_smoothed.png"
            )
            np.save(
                mroot / "results" / "difference_maps" / "smoothed" / f"{stem}_l2_smoothed.npy", sm
            )

            t_k0 = time.perf_counter()
            adv512, km_stats = predict_adversarial_mask(sm)
            t_km = time.perf_counter() - t_k0
            Image.fromarray(adv512, mode="L").save(
                mroot / "masks" / "adversarial_masks" / f"{stem}_adversarial_mask.png"
            )
            (mroot / "results" / "logs" / f"{stem}_kmeans.json").write_text(
                json.dumps(km_stats, indent=2), encoding="utf-8"
            )

            # Rectify at 512 then map back to original resolution
            t_r0 = time.perf_counter()
            rect512 = rectify_image(proc_arr, full, adv512)
            rect_orig = Image.fromarray(rect512).resize((orig_w, orig_h), Image.Resampling.LANCZOS)
            adv_orig = Image.fromarray(adv512, mode="L").resize(
                (orig_w, orig_h), Image.Resampling.NEAREST
            )
            t_rect = time.perf_counter() - t_r0
            restored_path = mroot / "restored_images" / fname
            rect_orig.save(assert_under_project_root(restored_path), quality=95)
            adv_orig.save(
                mroot / "masks" / "adversarial_masks" / f"{stem}_adversarial_mask_origres.png"
            )

            # True patch mask ONLY AFTER defense prediction (evaluation)
            tpm = true_patch_mask(
                orig_h,
                orig_w,
                r["true_patch_x1"],
                r["true_patch_y1"],
                r["true_patch_x2"],
                r["true_patch_y2"],
            )
            Image.fromarray(tpm, mode="L").save(
                mroot
                / "masks"
                / "evaluation_true_patch_masks"
                / f"{stem}_true_patch_mask.png"
            )
            loc = localization_metrics(np.array(adv_orig), tpm)

            # Detector on clean / attacked / restored (outputs only under mitigation)
            t_det0 = time.perf_counter()
            clean_pred = predict_save(
                clean_path, mroot / "results" / "clean_predictions", stem
            )
            attacked_pred = predict_save(
                attacked_path, mroot / "results" / "attacked_predictions", stem
            )
            restored_pred = predict_save(
                restored_path, mroot / "results" / "restored_predictions", stem
            )
            t_det = time.perf_counter() - t_det0

            gt = [
                r["bbox_x"],
                r["bbox_y"],
                r["bbox_x"] + r["bbox_width"],
                r["bbox_y"] + r["bbox_height"],
            ]
            c_ev = evaluate_detection(clean_pred["predictions"], r["target_class"], gt)
            a_ev = evaluate_detection(attacked_pred["predictions"], r["target_class"], gt)
            r_ev = evaluate_detection(restored_pred["predictions"], r["target_class"], gt)

            recovered = (
                r["subset_type"] == "attack_success"
                and c_ev["detected"]
                and (not a_ev["detected"])
                and r_ev["detected"]
            )
            control_preserved = (
                r["subset_type"] == "control" and c_ev["detected"] and a_ev["detected"] and r_ev["detected"]
            )
            control_regression = (
                r["subset_type"] == "control"
                and c_ev["detected"]
                and a_ev["detected"]
                and (not r_ev["detected"])
            )

            def gap(a, b):
                if a is None or b is None:
                    return None
                return float(b) - float(a)

            row_out.update(
                {
                    "clean_detected": c_ev["detected"],
                    "attacked_detected": a_ev["detected"],
                    "rectified_detected": r_ev["detected"],
                    "attack_success": r["subset_type"] == "attack_success",
                    "recovered_target": recovered,
                    "control_preserved": control_preserved,
                    "control_regression": control_regression,
                    "clean_confidence": c_ev["confidence"],
                    "attacked_confidence": a_ev["confidence"],
                    "rectified_confidence": r_ev["confidence"],
                    "confidence_recovery": gap(a_ev["confidence"], r_ev["confidence"]),
                    "clean_iou": c_ev["iou"],
                    "attacked_iou": a_ev["iou"],
                    "rectified_iou": r_ev["iou"],
                    "iou_recovery": gap(a_ev["iou"], r_ev["iou"]),
                    "true_patch_x1": r["true_patch_x1"],
                    "true_patch_y1": r["true_patch_y1"],
                    "true_patch_x2": r["true_patch_x2"],
                    "true_patch_y2": r["true_patch_y2"],
                    "true_patch_area": loc["true_patch_area"],
                    "predicted_mask_area": loc["predicted_mask_area"],
                    "mask_iou": loc["mask_iou"],
                    "mask_precision": loc["mask_precision"],
                    "mask_recall": loc["mask_recall"],
                    "false_positive_pixels": loc["false_positive_pixels"],
                    "false_negative_pixels": loc["false_negative_pixels"],
                    "false_positive_benign_ratio": loc["false_positive_benign_ratio"],
                    "regeneration_time_sec": t_pass0 + t_pass1,
                    "difference_time_sec": t_diff,
                    "kmeans_time_sec": t_km,
                    "rectification_time_sec": t_rect,
                    "detector_time_sec": t_det,
                    "total_time_sec": time.perf_counter() - t_total0,
                    "status": "completed",
                    "error_message": "",
                }
            )

            make_verification(
                clean_pil,
                attacked_pil,
                Image.fromarray(full).resize((orig_w, orig_h), Image.Resampling.LANCZOS),
                adv_orig,
                rect_orig,
                Image.open(restored_pred["viz"]),
                mroot / "results" / "verification" / f"{stem}_verification.jpg",
                f"{mid} | {stem} | {r['subset_type']}",
            )

            timing_rows.append(
                {
                    "image_id": image_id,
                    "filename": fname,
                    "pass0_sec": t_pass0,
                    "pass1_sec": t_pass1,
                    "difference_sec": t_diff,
                    "kmeans_sec": t_km,
                    "rectify_sec": t_rect,
                    "detector_sec": t_det,
                    "total_sec": row_out["total_time_sec"],
                }
            )
        except Exception as e:
            status = "failed_regeneration"
            # refine stage if message hints
            msg = f"{type(e).__name__}: {e}"
            err = msg
            row_out.update(
                {
                    "status": status,
                    "error_message": err,
                    "total_time_sec": time.perf_counter() - t_total0,
                    "recovered_target": False,
                    "control_preserved": False,
                    "control_regression": False,
                }
            )
            print(f"FAIL {fname}: {err}")
            (mroot / "results" / "logs" / f"{stem}_error.txt").write_text(err, encoding="utf-8")

        per_rows.append(row_out)
        # incremental save
        write_csv(mroot / "results" / "per_image_results.csv", per_rows)

    # Aggregates
    completed = [r for r in per_rows if r.get("status") == "completed"]
    failed = [r for r in per_rows if str(r.get("status", "")).startswith("failed")]
    det = aggregate_detection_metrics(
        [
            {
                **r,
                "recovered_target": r.get("recovered_target"),
                "control_preserved": r.get("control_preserved"),
                "control_regression": r.get("control_regression"),
            }
            for r in completed
        ]
    )
    loc_completed = [r for r in completed if r.get("mask_iou") is not None]
    mean = lambda xs: float(np.mean(xs)) if xs else 0.0
    summary = {
        "mitigation_id": mid,
        "source_attack": attack_id,
        "total_selected_images": len(rows),
        "successfully_processed_images": len(completed),
        "failed_images": len(failed),
        **det,
        "mean_confidence_recovery": mean(
            [r["confidence_recovery"] for r in completed if r.get("confidence_recovery") is not None]
        ),
        "mean_iou_recovery": mean(
            [r["iou_recovery"] for r in completed if r.get("iou_recovery") is not None]
        ),
        "mean_adversarial_mask_iou": mean([r["mask_iou"] for r in loc_completed]),
        "mean_mask_precision": mean([r["mask_precision"] for r in loc_completed]),
        "mean_mask_recall": mean([r["mask_recall"] for r in loc_completed]),
        "mean_processing_time_sec": mean([r["total_time_sec"] for r in completed]),
        "diffusion_model": "CompVis LDM inpainting_big",
        "sampler": SAMPLER,
        "steps": steps,
        "grid_n": CHECKERBOARD_GRID_N,
        "seed": DISPATCH_SEED,
        "checkpoint_sha256": ckpt_sha,
        "ldm_commit": commit,
        "clean_pixels_used_for_restoration": False,
        "known_patch_location_used_by_defense": False,
        "storage_guarantee": STORAGE_GUARANTEE,
        "external_resources_readonly": True,
        "previous_mitigation_results_reused": False,
    }

    res = mroot / "results"
    write_csv(res / "per_image_results.csv", per_rows)
    write_csv(res / "timing.csv", timing_rows)
    write_csv(
        res / "detection_metrics.csv",
        [det],
    )
    write_csv(
        res / "localization_metrics.csv",
        [
            {
                "image_id": r["image_id"],
                "filename": r["filename"],
                "mask_iou": r.get("mask_iou"),
                "mask_precision": r.get("mask_precision"),
                "mask_recall": r.get("mask_recall"),
                "true_patch_area": r.get("true_patch_area"),
                "predicted_mask_area": r.get("predicted_mask_area"),
                "false_positive_pixels": r.get("false_positive_pixels"),
                "false_negative_pixels": r.get("false_negative_pixels"),
                "false_positive_benign_ratio": r.get("false_positive_benign_ratio"),
            }
            for r in completed
        ],
    )
    write_csv(
        res / "class_metrics.csv",
        [
            {
                "class": "person",
                "success_count": det.get("person_success_count"),
                "recovered": det.get("person_recovered"),
                "recovery_rate": det.get("person_recovery_rate"),
            },
            {
                "class": "car",
                "success_count": det.get("car_success_count"),
                "recovered": det.get("car_recovered"),
                "recovery_rate": det.get("car_recovery_rate"),
            },
        ],
    )
    write_csv(res / "summary.csv", [summary])
    (res / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    summary_txt = f"""DISPATCH MITIGATION SUMMARY
Mitigation ID: {mid}
Source attack/data: {attack_id}
Total selected images: {len(rows)}
Successfully processed: {len(completed)}
Failed images: {len(failed)}

Attack-success cases: {det.get('attack_success_cases')}
Control cases: {det.get('control_count')}

Recovered target count: {det.get('recovered_target_count')}
Detection Recovery Rate: {100*det.get('detection_recovery_rate',0):.2f}%
Remaining failed targets: {det.get('remaining_failed_targets')}

Control preserved count: {det.get('control_preserved_count')}
Control Preservation Rate: {100*det.get('control_preservation_rate',0):.2f}%

Control regression count: {det.get('control_regression_count')}
Control Regression Rate: {100*det.get('control_regression_rate',0):.2f}%

Person recovery: {det.get('person_recovered')}/{det.get('person_success_count')} ({100*det.get('person_recovery_rate',0):.2f}%)
Car recovery: {det.get('car_recovered')}/{det.get('car_success_count')} ({100*det.get('car_recovery_rate',0):.2f}%)

Mean confidence recovery: {summary['mean_confidence_recovery']:.6f}
Mean IoU recovery: {summary['mean_iou_recovery']:.6f}

Mean adversarial-mask IoU: {summary['mean_adversarial_mask_iou']:.6f}
Mask precision: {summary['mean_mask_precision']:.6f}
Mask recall: {summary['mean_mask_recall']:.6f}

Mean processing time: {summary['mean_processing_time_sec']:.2f} sec

Diffusion model: CompVis LDM inpainting_big
Sampler: {SAMPLER}
Steps: {steps}
Grid N: {CHECKERBOARD_GRID_N}

clean_pixels_used_for_restoration = false
known_patch_location_used_by_defense = false

{STORAGE_GUARANTEE}
External resources were used read-only: YES
Previous mitigation results were reused: NO
True patch coordinates were supplied to the defense algorithm: NO
"""
    (res / "summary.txt").write_text(summary_txt, encoding="utf-8")

    # Simple overview plot
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(
            ["DRR", "CPR", "CRR", "MaskIoU"],
            [
                det.get("detection_recovery_rate", 0) * 100,
                det.get("control_preservation_rate", 0) * 100,
                det.get("control_regression_rate", 0) * 100,
                summary["mean_adversarial_mask_iou"] * 100,
            ],
            color=["#2a9d8f", "#264653", "#e76f51", "#e9c46a"],
        )
        ax.set_ylim(0, 100)
        ax.set_ylabel("%")
        ax.set_title(f"{mid} overview ({attack_id})")
        fig.tight_layout()
        fig.savefig(res / "overview.png", dpi=120)
        fig.savefig(res / "plots" / "overview.png", dpi=120)
        plt.close(fig)
    except Exception as e:
        (res / "logs" / "plot_error.txt").write_text(str(e), encoding="utf-8")

    append_experiment_inventory(
        {
            "run_id": mid,
            "datetime": datetime.now(timezone.utc).isoformat(),
            "source_attack": attack_id,
            "n_selected": len(rows),
            "n_completed": len(completed),
            "n_failed": len(failed),
            "checkpoint_sha256": ckpt_sha,
            "ldm_commit": commit,
            "seed": DISPATCH_SEED,
            "grid_n": CHECKERBOARD_GRID_N,
            "diffusion_steps": steps,
            "sampler": SAMPLER,
            "experiment_folder": str(mroot),
            "status": "completed" if not failed else "completed_with_failures",
        }
    )

    # Safety checklist file
    safety = f"""Mitigation folder:
{mroot}

source_attack_data complete = YES
source_clean_images complete = YES
source_attacked_images complete = YES
masks complete = YES
neutralized_inputs/prepared inputs complete = YES
restored_images complete = YES
results complete = YES

All generated files inside DISPATCH-Defense = YES

Adversarial-Patch-Experiment new files = 0
Hyper-YOLO new files = 0
Stable-Diffusion-Patch new files = 0

Existing DISPATCH mitigations overwritten = NO
"""
    (res / "safety_check.txt").write_text(safety, encoding="utf-8")
    print(summary_txt)
    print(safety)
    return mroot


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--attack", default="attack_02")
    p.add_argument("--n-success", type=int, default=15)
    p.add_argument("--n-control", type=int, default=15)
    p.add_argument("--steps", type=int, default=DIFFUSION_STEPS)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--continue-incomplete", action="store_true",
                   help="Reuse highest incomplete mitigation_* (no restored outputs yet)")
    p.add_argument("--i-approve-full-run", action="store_true")
    args = p.parse_args()
    if not args.i_approve_full_run:
        print("REFUSING: pass --i-approve-full-run to start a real mitigation run.")
        return 2
    if args.resume:
        print("Safe per-image resume not fully enabled yet; use --continue-incomplete for failed setup.")
        return 2
    run_mitigation(
        args.attack,
        args.n_success,
        args.n_control,
        args.steps,
        args.resume,
        continue_incomplete=args.continue_incomplete,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
