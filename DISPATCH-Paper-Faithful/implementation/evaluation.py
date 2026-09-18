"""Hyper-YOLO evaluation (read-only weights) + canonical target matching."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

import cv2
import numpy as np

ROOT = Path(r"D:\project CS\DISPATCH-Paper-Faithful").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.reference_config import (
    HYPER_YOLO_CONF,
    HYPER_YOLO_DEVICE,
    HYPER_YOLO_IMGSZ,
    HYPER_YOLO_NMS_IOU,
    HYPER_YOLO_ROOT,
    HYPER_YOLO_WEIGHTS,
    TARGET_MATCH_IOU,
    assert_isolated,
)


def box_iou(a: Iterable[float], b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def match_target(
    predictions: list[dict[str, Any]],
    target_class: str,
    gt_box_xyxy: Iterable[float],
    iou_thresh: float = TARGET_MATCH_IOU,
) -> Optional[dict[str, Any]]:
    best = None
    best_iou = -1.0
    for pred in predictions:
        if str(pred.get("class_name", "")).lower() != str(target_class).lower():
            continue
        iou = box_iou(pred["box_xyxy"], gt_box_xyxy)
        if iou >= iou_thresh and iou > best_iou:
            best_iou = iou
            best = {**pred, "match_iou": iou}
    return best


def localization_metrics(pred_mask: np.ndarray, true_mask: np.ndarray) -> dict:
    p = pred_mask > 127
    t = true_mask > 127
    inter = int((p & t).sum())
    union = int((p | t).sum())
    pred_area = int(p.sum())
    true_area = int(t.sum())
    precision = float(inter / pred_area) if pred_area else 0.0
    recall = float(inter / true_area) if true_area else 0.0
    iou = float(inter / union) if union else 0.0
    return {
        "mask_iou": iou,
        "mask_precision": precision,
        "mask_recall": recall,
        "predicted_mask_area": pred_area,
        "true_patch_area": true_area,
        "intersection_area": inter,
        "area_ratio": float(pred_area / true_area) if true_area else None,
    }


def _purge_ultralytics() -> None:
    for k in list(sys.modules):
        if k == "ultralytics" or k.startswith("ultralytics."):
            del sys.modules[k]


def load_hyper_yolo():
    if not HYPER_YOLO_ROOT.is_dir():
        raise FileNotFoundError(HYPER_YOLO_ROOT)
    if not HYPER_YOLO_WEIGHTS.is_file():
        raise FileNotFoundError(HYPER_YOLO_WEIGHTS)
    _purge_ultralytics()
    hy = str(HYPER_YOLO_ROOT)
    if hy in sys.path:
        sys.path.remove(hy)
    sys.path.insert(0, hy)
    from ultralytics import YOLO  # type: ignore

    return YOLO(str(HYPER_YOLO_WEIGHTS))


def predictions_from_result(r0) -> list[dict[str, Any]]:
    preds: list[dict[str, Any]] = []
    names = r0.names if hasattr(r0, "names") else {}
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
    return preds


def yolo_predict(model, image_path: Path, img_dir: Path, lab_dir: Path, filename: str) -> dict:
    img_dir = assert_isolated(img_dir)
    lab_dir = assert_isolated(lab_dir)
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    r0 = model.predict(
        source=str(image_path),
        conf=HYPER_YOLO_CONF,
        iou=HYPER_YOLO_NMS_IOU,
        imgsz=HYPER_YOLO_IMGSZ,
        device=HYPER_YOLO_DEVICE,
        verbose=False,
        save=False,
    )[0]
    preds = predictions_from_result(r0)
    stem = Path(filename).stem
    (lab_dir / f"{stem}.json").write_text(
        json.dumps({"filename": filename, "predictions": preds}, indent=2),
        encoding="utf-8",
    )
    cv2.imwrite(str(img_dir / filename), r0.plot())
    return {"predictions": preds}


def detect(preds, cls, gt):
    m = match_target(preds, cls, gt, iou_thresh=TARGET_MATCH_IOU)
    if m is None:
        return False, None, None
    return True, float(m["confidence"]), float(m["match_iou"])
