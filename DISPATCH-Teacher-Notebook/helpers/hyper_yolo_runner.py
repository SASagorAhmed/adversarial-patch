#!/usr/bin/env python
"""Hyper-YOLO runner. Weights/code read-only; all outputs under Teacher-Notebook."""
from __future__ import annotations

import os

os.environ["MPLBACKEND"] = "Agg"

import csv
import json
import sys
from pathlib import Path

import cv2
from PIL import Image

ROOT = Path(r"D:\project CS\DISPATCH-Teacher-Notebook").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from helpers.evaluation import control_preserved, detect, gt_xyxy, recovered
from helpers.safety import assert_writable, ensure_dir, load_config
from helpers.visualization import hstack, labeled_panel, save_jpg


def _purge_ultralytics() -> None:
    for k in list(sys.modules):
        if k == "ultralytics" or k.startswith("ultralytics."):
            del sys.modules[k]


def load_model():
    cfg = load_config()
    hy = str(Path(cfg["hyper_yolo_root"]))
    _purge_ultralytics()
    if hy in sys.path:
        sys.path.remove(hy)
    sys.path.insert(0, hy)
    from ultralytics import YOLO  # type: ignore

    return YOLO(cfg["hyper_yolo_weights"]), cfg


def predict(model, cfg, image_path: Path, img_dir: Path, lab_dir: Path, filename: str) -> list[dict]:
    ensure_dir(img_dir)
    ensure_dir(lab_dir)
    r0 = model.predict(
        source=str(image_path),
        conf=float(cfg["hyper_yolo_conf"]),
        iou=float(cfg["hyper_yolo_iou"]),
        imgsz=int(cfg["hyper_yolo_imgsz"]),
        device=cfg["hyper_yolo_device"],
        verbose=False,
        save=False,
    )[0]
    names = r0.names if hasattr(r0, "names") else {}
    preds = []
    if r0.boxes is not None and len(r0.boxes):
        xyxy = r0.boxes.xyxy.cpu().numpy()
        confs = r0.boxes.conf.cpu().numpy()
        clss = r0.boxes.cls.cpu().numpy().astype(int)
        for box, c, cls_id in zip(xyxy, confs, clss):
            preds.append(
                {
                    "class_id": int(cls_id),
                    "class_name": str(names.get(int(cls_id), cls_id)),
                    "confidence": float(c),
                    "box_xyxy": [float(x) for x in box.tolist()],
                }
            )
    stem = Path(filename).stem
    (assert_writable(lab_dir / f"{stem}.json")).write_text(
        json.dumps({"filename": filename, "predictions": preds}, indent=2), encoding="utf-8"
    )
    cv2.imwrite(str(assert_writable(img_dir / filename)), r0.plot())
    return preds


def pack(det, conf, iou):
    return det, conf, iou


def run_yolo() -> None:
    model, cfg = load_model()
    sel = ROOT / "source_data" / "source_attack_data" / "selected_images.csv"
    with sel.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    branches = {
        "clean": ROOT / "source_data" / "clean_images",
        "attacked": ROOT / "source_data" / "attacked_images",
        "automatic": ROOT / "automatic" / "restored_images",
        "known_exact": ROOT / "known_location" / "restored_images" / "exact_mask",
        "known_expand4": ROOT / "known_location" / "restored_images" / "expanded_4px",
    }
    pred_dirs = {
        "clean": (ROOT / "automatic" / "results" / "clean_predictions", ROOT / "known_location" / "results" / "clean_predictions"),
        "attacked": (ROOT / "automatic" / "results" / "attacked_predictions", ROOT / "known_location" / "results" / "attacked_predictions"),
        "automatic": (ROOT / "automatic" / "results" / "restored_predictions",),
        "known_exact": (ROOT / "known_location" / "results" / "restored_predictions" / "exact_mask",),
        "known_expand4": (ROOT / "known_location" / "results" / "restored_predictions" / "expanded_4px",),
    }

    auto_loc = {}
    with (ROOT / "automatic" / "results" / "per_image_results.csv").open(encoding="utf-8", newline="") as f:
        auto_loc = {r["filename"]: r for r in csv.DictReader(f)}
    known_loc = {}
    with (ROOT / "known_location" / "results" / "per_image_results.csv").open(encoding="utf-8", newline="") as f:
        known_loc = {r["filename"]: r for r in csv.DictReader(f)}

    combined = []
    for r in rows:
        fn = r["filename"]
        gt = gt_xyxy(r)
        cls = r["target_class"]
        dets = {}
        for key, src in branches.items():
            preds = None
            for out_root in pred_dirs[key]:
                preds = predict(model, cfg, src / fn, out_root / "images", out_root / "labels", fn)
            dets[key] = detect(preds, cls, gt)

        al = auto_loc[fn]
        kn = known_loc[fn]
        c_det, c_conf, c_iou = dets["clean"]
        a_det, a_conf, a_iou = dets["attacked"]
        u_det, u_conf, u_iou = dets["automatic"]
        e_det, e_conf, e_iou = dets["known_exact"]
        x_det, x_conf, x_iou = dets["known_expand4"]
        rec = {
            "filename": fn,
            "target": cls,
            "case": r["case_type"],
            "clean_detected": c_det,
            "clean_confidence": c_conf,
            "clean_iou": c_iou,
            "attacked_detected": a_det,
            "attacked_confidence": a_conf,
            "attacked_iou": a_iou,
            "automatic_mask_iou": al["mask_iou"],
            "automatic_precision": al["precision"],
            "automatic_recall": al["recall"],
            "automatic_detected": u_det,
            "automatic_confidence": u_conf,
            "automatic_iou": u_iou,
            "automatic_recovered": recovered(r["case_type"], u_det),
            "known_exact_detected": e_det,
            "known_exact_confidence": e_conf,
            "known_exact_iou": e_iou,
            "known_exact_recovered": recovered(r["case_type"], e_det),
            "known_expand4_detected": x_det,
            "known_expand4_confidence": x_conf,
            "known_expand4_iou": x_iou,
            "known_expand4_recovered": recovered(r["case_type"], x_det),
            "outside_mask_mae_exact": kn["exact_mask_outside_mae"],
            "outside_mask_mae_expand4": kn["expanded_4px_outside_mae"],
            "automatic_control_preserved": control_preserved(r["case_type"], u_det),
            "known_exact_control_preserved": control_preserved(r["case_type"], e_det),
            "known_expand4_control_preserved": control_preserved(r["case_type"], x_det),
        }
        if r["case_type"] == "attack_success":
            if rec["automatic_recovered"] and not rec["known_exact_recovered"]:
                interp = "Automatic recovered; known exact did not (mixed / incidental)."
            elif (not rec["automatic_recovered"]) and rec["known_exact_recovered"]:
                interp = "Case A: Automatic fail / known succeed — localization likely bottleneck."
            elif (not rec["automatic_recovered"]) and rec["known_expand4_recovered"]:
                interp = "Case C-ish: exact fail, +4 succeed — tight mask may be insufficient."
            elif not rec["automatic_recovered"] and not rec["known_expand4_recovered"]:
                interp = "Case C: both fail — localization alone does not explain failure."
            else:
                interp = "Both Automatic and known recovered, or mixed."
        else:
            interp = "control"
        rec["interpretation"] = interp
        combined.append(rec)

        panels = [
            labeled_panel(Image.open(ROOT / "source_data" / "clean_images" / fn), "CLEAN"),
            labeled_panel(Image.open(ROOT / "source_data" / "attacked_images" / fn), "ATTACKED"),
            labeled_panel(Image.open(ROOT / "automatic" / "masks" / "predicted_masks" / f"{Path(fn).stem}.png").convert("RGB"), "AUTO MASK"),
            labeled_panel(Image.open(ROOT / "automatic" / "restored_images" / fn), "AUTO FINAL"),
            labeled_panel(Image.open(ROOT / "known_location" / "masks" / "exact_mask" / f"{Path(fn).stem}.png").convert("RGB"), "KNOWN EXACT MASK"),
            labeled_panel(Image.open(ROOT / "known_location" / "restored_images" / "exact_mask" / fn), "KNOWN EXACT FINAL"),
            labeled_panel(Image.open(ROOT / "known_location" / "restored_images" / "expanded_4px" / fn), "KNOWN +4 FINAL"),
        ]
        yolo_auto = ROOT / "automatic" / "results" / "restored_predictions" / "images" / fn
        if yolo_auto.is_file():
            panels.append(labeled_panel(Image.open(yolo_auto), "YOLO AUTO"))
        save_jpg(hstack(panels), ROOT / "comparison" / "images" / fn)

    ensure_dir(ROOT / "comparison")
    _csv(ROOT / "comparison" / "automatic_vs_known.csv", combined)
    _csv(ROOT / "automatic" / "results" / "notebook_per_image_results.csv", combined)
    _csv(ROOT / "comparison" / "per_image_results.csv", combined)

    succ = [x for x in combined if x["case"] == "attack_success"]
    ctrl = [x for x in combined if x["case"] == "control"]
    n_s = len(succ)
    lines = [
        "DIAGNOSTIC SUBSET RECOVERY RATE (5-image notebook — NOT DISPATCH DRR, NOT paper mAP)",
        f"Automatic recovered: {sum(1 for x in succ if x['automatic_recovered'])} / {n_s}",
        f"Known Exact recovered: {sum(1 for x in succ if x['known_exact_recovered'])} / {n_s}",
        f"Known +4px recovered: {sum(1 for x in succ if x['known_expand4_recovered'])} / {n_s}",
        f"Automatic mean mask IoU: {sum(float(x['automatic_mask_iou']) for x in combined)/len(combined):.4f}",
        f"Automatic controls preserved: {sum(1 for x in ctrl if x['automatic_control_preserved'])} / {len(ctrl)}",
        f"Known exact controls preserved: {sum(1 for x in ctrl if x['known_exact_control_preserved'])} / {len(ctrl)}",
        f"Known +4 controls preserved: {sum(1 for x in ctrl if x['known_expand4_control_preserved'])} / {len(ctrl)}",
    ]
    (assert_writable(ROOT / "comparison" / "summary.txt")).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def _csv(path: Path, rows: list[dict]) -> None:
    path = assert_writable(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    run_yolo()
