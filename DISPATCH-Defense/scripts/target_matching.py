#!/usr/bin/env python
"""Target matching: same class + IoU >= 0.50 (canonical project rule)."""
from __future__ import annotations

from typing import Any, Iterable, Optional


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
    target_class: str | int,
    gt_box_xyxy: Iterable[float],
    iou_thresh: float = 0.50,
) -> Optional[dict[str, Any]]:
    """Return best-IoU prediction matching class with IoU>=thresh, else None.

    Each prediction dict needs: class_name or class_id, box_xyxy, confidence.
    """
    best = None
    best_iou = -1.0
    for pred in predictions:
        pname = pred.get("class_name", pred.get("name"))
        pid = pred.get("class_id", pred.get("cls"))
        if isinstance(target_class, str):
            ok = str(pname).lower() == str(target_class).lower()
        else:
            ok = int(pid) == int(target_class)
        if not ok:
            continue
        iou = box_iou(pred["box_xyxy"], gt_box_xyxy)
        if iou >= iou_thresh and iou > best_iou:
            best_iou = iou
            best = {**pred, "match_iou": iou}
    return best
