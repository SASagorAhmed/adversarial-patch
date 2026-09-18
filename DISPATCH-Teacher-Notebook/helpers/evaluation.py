"""Evaluation helpers: mask metrics + canonical target matching. Writes only under Teacher-Notebook."""
from __future__ import annotations

from typing import Any, Iterable, Optional

import numpy as np

from helpers.safety import load_config


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
    return float(inter / union) if union else 0.0


def match_target(predictions: list[dict[str, Any]], target_class: str, gt_xyxy: Iterable[float], iou_thresh: float | None = None):
    cfg = load_config()
    if iou_thresh is None:
        iou_thresh = float(cfg["target_match_iou"])
    best = None
    best_iou = -1.0
    for pred in predictions:
        if str(pred.get("class_name", "")).lower() != str(target_class).lower():
            continue
        iou = box_iou(pred["box_xyxy"], gt_xyxy)
        if iou >= iou_thresh and iou > best_iou:
            best_iou = iou
            best = {**pred, "match_iou": iou}
    return best


def detect(preds, cls, gt):
    m = match_target(preds, cls, gt)
    if m is None:
        return False, None, None
    return True, float(m["confidence"]), float(m["match_iou"])


def localization_metrics(pred_mask: np.ndarray, true_mask: np.ndarray) -> dict:
    p = pred_mask > 127
    t = true_mask > 127
    inter = int((p & t).sum())
    union = int((p | t).sum())
    pred_area = int(p.sum())
    true_area = int(t.sum())
    return {
        "mask_iou": float(inter / union) if union else 0.0,
        "precision": float(inter / pred_area) if pred_area else 0.0,
        "recall": float(inter / true_area) if true_area else 0.0,
        "predicted_mask_area": pred_area,
        "true_patch_area": true_area,
    }


def gt_xyxy(row: dict) -> list[float]:
    x, y, w, h = float(row["gt_x"]), float(row["gt_y"]), float(row["gt_width"]), float(row["gt_height"])
    return [x, y, x + w, y + h]


def recovered(case_type: str, detected: bool) -> bool:
    return bool(case_type == "attack_success" and detected)


def control_preserved(case_type: str, detected: bool) -> bool:
    return bool(case_type == "control" and detected)
