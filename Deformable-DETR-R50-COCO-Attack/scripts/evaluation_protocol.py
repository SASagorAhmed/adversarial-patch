"""Shared evaluation protocol (identical to Faster R-CNN / RT-DETRv2 / D-FINE / YOLOv12)."""
from __future__ import annotations

from typing import Any, Iterable

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.50


def box_iou_xyxy(a: Iterable[float], b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union else 0.0


def gt_xywh_to_xyxy(x: float, y: float, w: float, h: float) -> list[float]:
    return [float(x), float(y), float(x) + float(w), float(y) + float(h)]


def match_target(
    detections: list[dict[str, Any]],
    target_class: str,
    gt_xyxy: Iterable[float],
    *,
    conf_thresh: float = CONF_THRESHOLD,
    iou_thresh: float = IOU_THRESHOLD,
) -> dict[str, Any]:
    candidates = [
        d
        for d in detections
        if float(d["confidence"]) >= conf_thresh
        and str(d.get("class_name", "")).lower() == str(target_class).lower()
    ]
    if not candidates:
        return {
            "detected": False,
            "confidence": None,
            "iou": None,
            "bbox": None,
            "class_id": None,
            "class_name": None,
        }

    def sort_key(d):
        iou = box_iou_xyxy(d["bbox"], gt_xyxy)
        return (iou, float(d["confidence"]))

    best = max(candidates, key=sort_key)
    best_iou = box_iou_xyxy(best["bbox"], gt_xyxy)
    if best_iou < iou_thresh:
        return {
            "detected": False,
            "confidence": None,
            "iou": None,
            "bbox": None,
            "class_id": None,
            "class_name": None,
        }
    return {
        "detected": True,
        "confidence": float(best["confidence"]),
        "iou": float(best_iou),
        "bbox": [float(x) for x in best["bbox"]],
        "class_id": int(best["class_id"]),
        "class_name": str(best["class_name"]),
    }
