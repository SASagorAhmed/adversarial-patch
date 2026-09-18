#!/usr/bin/env python
"""Evaluation helpers: detection metrics + post-hoc patch localization."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np


def localization_metrics(
    predicted_mask: np.ndarray,
    true_patch_mask: np.ndarray,
) -> dict[str, Any]:
    """Compare predicted A vs true patch (EVALUATION ONLY; never during defense)."""
    p = (predicted_mask.astype(np.float32) > 0).astype(np.uint8)
    t = (true_patch_mask.astype(np.float32) > 0).astype(np.uint8)
    if p.shape != t.shape:
        raise ValueError(f"mask shape mismatch {p.shape} vs {t.shape}")
    inter = int(np.logical_and(p, t).sum())
    union = int(np.logical_or(p, t).sum())
    pred_area = int(p.sum())
    true_area = int(t.sum())
    fp = int(np.logical_and(p == 1, t == 0).sum())
    fn = int(np.logical_and(p == 0, t == 1).sum())
    benign = max(int((t == 0).sum()), 1)
    return {
        "true_patch_area": true_area,
        "predicted_mask_area": pred_area,
        "mask_intersection": inter,
        "mask_union": union,
        "mask_iou": float(inter / union) if union else 0.0,
        "mask_precision": float(inter / pred_area) if pred_area else 0.0,
        "mask_recall": float(inter / true_area) if true_area else 0.0,
        "false_positive_pixels": fp,
        "false_negative_pixels": fn,
        "false_positive_benign_ratio": float(fp / benign),
    }


def aggregate_detection_metrics(rows: list[dict]) -> dict[str, Any]:
    """Compute DRR/CPR/CRR style metrics from per-image result dicts."""
    total = len(rows)
    success = [r for r in rows if str(r.get("subset_type")) == "attack_success"]
    control = [r for r in rows if str(r.get("subset_type")) == "control"]
    recovered = [r for r in success if _truthy(r.get("recovered_target"))]
    preserved = [r for r in control if _truthy(r.get("control_preserved"))]
    regressed = [r for r in control if _truthy(r.get("control_regression"))]
    failed = [r for r in rows if str(r.get("status", "")).startswith("failed")]
    completed = [r for r in rows if str(r.get("status")) == "completed"]

    def rate(num, den):
        return float(num / den) if den else 0.0

    person_s = [r for r in success if str(r.get("target_class")).lower() == "person"]
    car_s = [r for r in success if str(r.get("target_class")).lower() == "car"]
    return {
        "total_images": total,
        "completed_images": len(completed),
        "failed_images": len(failed),
        "attack_success_cases": len(success),
        "recovered_target_count": len(recovered),
        "detection_recovery_rate": rate(len(recovered), len(success)),
        "remaining_failed_targets": len(success) - len(recovered),
        "control_count": len(control),
        "control_preserved_count": len(preserved),
        "control_preservation_rate": rate(len(preserved), len(control)),
        "control_regression_count": len(regressed),
        "control_regression_rate": rate(len(regressed), len(control)),
        "person_success_count": len(person_s),
        "person_recovered": sum(1 for r in person_s if _truthy(r.get("recovered_target"))),
        "person_recovery_rate": rate(
            sum(1 for r in person_s if _truthy(r.get("recovered_target"))), len(person_s)
        ),
        "car_success_count": len(car_s),
        "car_recovered": sum(1 for r in car_s if _truthy(r.get("recovered_target"))),
        "car_recovery_rate": rate(
            sum(1 for r in car_s if _truthy(r.get("recovered_target"))), len(car_s)
        ),
    }


def _truthy(v) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if not rows and not fieldnames:
        path.write_text("", encoding="utf-8")
        return
    fields = fieldnames or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
