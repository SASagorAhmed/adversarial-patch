#!/usr/bin/env python3
"""Reporting helpers for mitigation experiments."""

from __future__ import annotations

import csv
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mitigation_config import RESULTS_DIR, TARGET_MATCH_IOU_THRESHOLD
from select_mitigation_subset import SubsetRow

MITIGATION_RESULTS_COLUMNS = [
    "mitigation_id",
    "source_attack_id",
    "image_id",
    "filename",
    "target_class",
    "case_type",
    "clean_detected",
    "attacked_detected",
    "mitigated_detected",
    "clean_confidence",
    "attacked_confidence",
    "mitigated_confidence",
    "confidence_recovery",
    "clean_to_mitigated_confidence_gap",
    "clean_iou",
    "attacked_iou",
    "mitigated_iou",
    "iou_recovery",
    "clean_to_mitigated_iou_gap",
    "attack_success",
    "recovered_after_mitigation",
    "control_preserved",
    "control_regressed",
    "patch_size_pixels",
    "patch_x1",
    "patch_y1",
    "patch_x2",
    "patch_y2",
    "restoration_seed",
]

GLOBAL_MITIGATION_COLUMNS = [
    "mitigation_id",
    "source_attack_id",
    "subset_total",
    "source_attack_success_count",
    "recovered_target_count",
    "detection_recovery_rate",
    "remaining_failed_targets",
    "control_count",
    "control_preserved_count",
    "control_preservation_rate",
    "control_regression_count",
    "control_regression_rate",
    "person_recovery_rate",
    "car_recovery_rate",
    "mean_confidence_recovery_success",
    "mean_iou_recovery_success",
    "mean_clean_to_mitigated_confidence_gap_success",
    "mean_clean_to_mitigated_iou_gap_success",
    "target_match_iou_threshold",
]


@dataclass
class MitigationResultRow:
    mitigation_id: str
    source_attack_id: str
    image_id: str
    filename: str
    target_class: str
    case_type: str
    clean_detected: str
    attacked_detected: str
    mitigated_detected: str
    clean_confidence: float
    attacked_confidence: float
    mitigated_confidence: float
    confidence_recovery: float
    clean_to_mitigated_confidence_gap: float
    clean_iou: float
    attacked_iou: float
    mitigated_iou: float
    iou_recovery: float
    clean_to_mitigated_iou_gap: float
    attack_success: str
    recovered_after_mitigation: str
    control_preserved: str
    control_regressed: str
    patch_size_pixels: int
    patch_x1: int
    patch_y1: int
    patch_x2: int
    patch_y2: int
    restoration_seed: int


def build_mitigation_result_row(
    subset_row: SubsetRow,
    mitigated_detected: str,
    mitigated_confidence: float,
    mitigated_iou: float,
    restoration_seed: int,
) -> MitigationResultRow:
    recovered = (
        "Yes"
        if subset_row.subset_role == "success" and mitigated_detected == "Yes"
        else "No"
    )
    control_preserved = (
        "Yes"
        if subset_row.subset_role == "control" and mitigated_detected == "Yes"
        else "No"
    )
    control_regressed = (
        "Yes"
        if subset_row.subset_role == "control" and mitigated_detected == "No"
        else "No"
    )
    return MitigationResultRow(
        mitigation_id=subset_row.mitigation_id,
        source_attack_id=subset_row.source_attack_id,
        image_id=subset_row.image_id,
        filename=subset_row.filename,
        target_class=subset_row.target_class,
        case_type=subset_row.subset_role,
        clean_detected=subset_row.clean_detected,
        attacked_detected=subset_row.attacked_detected,
        mitigated_detected=mitigated_detected,
        clean_confidence=subset_row.clean_confidence,
        attacked_confidence=subset_row.attacked_confidence,
        mitigated_confidence=mitigated_confidence,
        confidence_recovery=mitigated_confidence - subset_row.attacked_confidence,
        clean_to_mitigated_confidence_gap=subset_row.clean_confidence - mitigated_confidence,
        clean_iou=subset_row.clean_iou,
        attacked_iou=subset_row.attacked_iou,
        mitigated_iou=mitigated_iou,
        iou_recovery=mitigated_iou - subset_row.attacked_iou,
        clean_to_mitigated_iou_gap=subset_row.clean_iou - mitigated_iou,
        attack_success=subset_row.attack_success,
        recovered_after_mitigation=recovered,
        control_preserved=control_preserved,
        control_regressed=control_regressed,
        patch_size_pixels=subset_row.reconstructed_patch_size,
        patch_x1=subset_row.reconstructed_x1,
        patch_y1=subset_row.reconstructed_y1,
        patch_x2=subset_row.reconstructed_x2,
        patch_y2=subset_row.reconstructed_y2,
        restoration_seed=restoration_seed,
    )


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def compute_mitigation_metrics(rows: list[MitigationResultRow]) -> dict[str, float | int | str]:
    successes = [row for row in rows if row.case_type == "success"]
    controls = [row for row in rows if row.case_type == "control"]
    recovered = [row for row in successes if row.recovered_after_mitigation == "Yes"]
    preserved = [row for row in controls if row.control_preserved == "Yes"]
    person_success = [row for row in successes if row.target_class == "person"]
    car_success = [row for row in successes if row.target_class == "car"]
    person_recovered = [row for row in recovered if row.target_class == "person"]
    car_recovered = [row for row in recovered if row.target_class == "car"]

    source_attack_success_count = len(successes)
    recovered_target_count = len(recovered)
    control_count = len(controls)
    control_preserved_count = len(preserved)
    control_regression_count = control_count - control_preserved_count

    detection_recovery_rate = (
        recovered_target_count / source_attack_success_count * 100.0
        if source_attack_success_count
        else 0.0
    )
    control_preservation_rate = (
        control_preserved_count / control_count * 100.0 if control_count else 0.0
    )
    control_regression_rate = (
        control_regression_count / control_count * 100.0 if control_count else 0.0
    )
    person_recovered_count = len(person_recovered)
    car_recovered_count = len(car_recovered)
    person_success_count = len(person_success)
    car_success_count = len(car_success)
    person_recovery_rate = (
        person_recovered_count / person_success_count * 100.0 if person_success_count else 0.0
    )
    car_recovery_rate = (
        car_recovered_count / car_success_count * 100.0 if car_success_count else 0.0
    )

    return {
        "mitigation_id": rows[0].mitigation_id if rows else "",
        "source_attack_id": rows[0].source_attack_id if rows else "",
        "subset_total": len(rows),
        "source_attack_success_count": source_attack_success_count,
        "recovered_target_count": recovered_target_count,
        "detection_recovery_rate": detection_recovery_rate,
        "remaining_failed_targets": source_attack_success_count - recovered_target_count,
        "control_count": control_count,
        "control_preserved_count": control_preserved_count,
        "control_preservation_rate": control_preservation_rate,
        "control_regression_count": control_regression_count,
        "control_regression_rate": control_regression_rate,
        "person_success_count": person_success_count,
        "person_recovered_count": person_recovered_count,
        "person_recovery_rate": person_recovery_rate,
        "car_success_count": car_success_count,
        "car_recovered_count": car_recovered_count,
        "car_recovery_rate": car_recovery_rate,
        "mean_clean_confidence": _mean([r.clean_confidence for r in successes]),
        "mean_attacked_confidence": _mean([r.attacked_confidence for r in successes]),
        "mean_mitigated_confidence": _mean([r.mitigated_confidence for r in successes]),
        "mean_confidence_recovery": _mean([r.confidence_recovery for r in successes]),
        "mean_clean_iou": _mean([r.clean_iou for r in successes]),
        "mean_attacked_iou": _mean([r.attacked_iou for r in successes]),
        "mean_mitigated_iou": _mean([r.mitigated_iou for r in successes]),
        "mean_iou_recovery": _mean([r.iou_recovery for r in successes]),
        "mean_clean_confidence_success": _mean([r.clean_confidence for r in successes]),
        "mean_attacked_confidence_success": _mean([r.attacked_confidence for r in successes]),
        "mean_mitigated_confidence_success": _mean([r.mitigated_confidence for r in successes]),
        "mean_confidence_recovery_success": _mean([r.confidence_recovery for r in successes]),
        "mean_clean_iou_success": _mean([r.clean_iou for r in successes]),
        "mean_attacked_iou_success": _mean([r.attacked_iou for r in successes]),
        "mean_mitigated_iou_success": _mean([r.mitigated_iou for r in successes]),
        "mean_iou_recovery_success": _mean([r.iou_recovery for r in successes]),
        "mean_clean_to_mitigated_confidence_gap_success": _mean(
            [r.clean_to_mitigated_confidence_gap for r in successes]
        ),
        "mean_clean_to_mitigated_iou_gap_success": _mean(
            [r.clean_to_mitigated_iou_gap for r in successes]
        ),
        "target_match_iou_threshold": TARGET_MATCH_IOU_THRESHOLD,
    }


def write_mitigation_results_csv(path: Path, rows: list[MitigationResultRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MITIGATION_RESULTS_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_mitigation_summary(path: Path, metrics: dict[str, float | int | str], config: dict) -> None:
    lines = [
        "Mitigation Summary",
        "=" * 72,
        f"Method: {config.get('method_name', '')}",
        f"Mitigation ID: {metrics['mitigation_id']}",
        f"Source attack: {metrics['source_attack_id']}",
        f"Target match IoU threshold: {metrics['target_match_iou_threshold']}",
        "",
        f"Subset total: {metrics['subset_total']}",
        f"source_attack_success_count: {metrics['source_attack_success_count']}",
        f"recovered_target_count: {metrics['recovered_target_count']}",
        f"detection_recovery_rate: {float(metrics['detection_recovery_rate']):.2f}%",
        f"remaining_failed_targets: {metrics['remaining_failed_targets']}",
        "",
        f"control_count: {metrics['control_count']}",
        f"control_preserved_count: {metrics['control_preserved_count']}",
        f"control_preservation_rate: {float(metrics['control_preservation_rate']):.2f}%",
        f"control_regression_count: {metrics['control_regression_count']}",
        f"control_regression_rate: {float(metrics['control_regression_rate']):.2f}%",
        "",
        f"person_recovered_count: {metrics['person_recovered_count']} / {metrics['person_success_count']}",
        f"person_recovery_rate: {float(metrics['person_recovery_rate']):.2f}%",
        f"car_recovered_count: {metrics['car_recovered_count']} / {metrics['car_success_count']}",
        f"car_recovery_rate: {float(metrics['car_recovery_rate']):.2f}%",
        "",
        f"mean_clean_confidence (success): {float(metrics['mean_clean_confidence']):.4f}",
        f"mean_attacked_confidence (success): {float(metrics['mean_attacked_confidence']):.4f}",
        f"mean_mitigated_confidence (success): {float(metrics['mean_mitigated_confidence']):.4f}",
        f"mean_confidence_recovery (success): {float(metrics['mean_confidence_recovery']):.4f}",
        f"mean_clean_iou (success): {float(metrics['mean_clean_iou']):.4f}",
        f"mean_attacked_iou (success): {float(metrics['mean_attacked_iou']):.4f}",
        f"mean_mitigated_iou (success): {float(metrics['mean_mitigated_iou']):.4f}",
        f"mean_iou_recovery (success): {float(metrics['mean_iou_recovery']):.4f}",
        f"mean_clean_to_mitigated_confidence_gap (success): "
        f"{float(metrics['mean_clean_to_mitigated_confidence_gap_success']):.4f}",
        f"mean_clean_to_mitigated_iou_gap (success): "
        f"{float(metrics['mean_clean_to_mitigated_iou_gap_success']):.4f}",
        "",
        "Note: detection_recovery_rate is a subset metric, not full-dataset ASR after defense.",
        "",
        "Restoration config:",
        json.dumps(config, indent=2),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_comparison_plots(
    comparison_dir: Path,
    metrics: dict[str, float | int | str],
    rows: list[MitigationResultRow] | None = None,
) -> None:
    comparison_dir.mkdir(parents=True, exist_ok=True)

    # Detection recovery
    plt.figure(figsize=(8, 5))
    labels = ["Recovered", "Remaining failed"]
    values = [
        int(metrics["recovered_target_count"]),
        int(metrics["remaining_failed_targets"]),
    ]
    plt.bar(labels, values, color=["#54A24B", "#E45756"])
    plt.title(
        f"Detection Recovery — {metrics['mitigation_id']} "
        f"({float(metrics['detection_recovery_rate']):.1f}%)"
    )
    plt.ylabel("Count (success subset)")
    plt.tight_layout()
    plt.savefig(comparison_dir / "detection_recovery.png", dpi=150)
    plt.close()

    # Control preservation/regression
    plt.figure(figsize=(8, 5))
    labels = ["Preserved", "Regressed"]
    values = [
        int(metrics["control_preserved_count"]),
        int(metrics["control_regression_count"]),
    ]
    plt.bar(labels, values, color=["#4C78A8", "#F58518"])
    plt.title(
        f"Control Preservation — {metrics['mitigation_id']} "
        f"({float(metrics['control_preservation_rate']):.1f}%)"
    )
    plt.ylabel("Count (control subset)")
    plt.tight_layout()
    plt.savefig(comparison_dir / "control_preservation.png", dpi=150)
    plt.close()

    # Person vs car recovery
    plt.figure(figsize=(8, 5))
    plt.bar(
        ["Person recovery %", "Car recovery %"],
        [float(metrics["person_recovery_rate"]), float(metrics["car_recovery_rate"])],
        color=["#72B7B2", "#B279A2"],
    )
    plt.ylim(0, 100)
    plt.title(f"Person vs Car Recovery — {metrics['mitigation_id']}")
    plt.tight_layout()
    plt.savefig(comparison_dir / "person_vs_car_recovery.png", dpi=150)
    plt.close()

    if rows:
        successes = [row for row in rows if row.case_type == "success"]
        if successes:
            # Confidence clean/attacked/mitigated means on success subset
            plt.figure(figsize=(8, 5))
            conf_vals = [
                _mean([r.clean_confidence for r in successes]),
                _mean([r.attacked_confidence for r in successes]),
                _mean([r.mitigated_confidence for r in successes]),
            ]
            plt.bar(["Clean", "Attacked", "Mitigated"], conf_vals, color=["#54A24B", "#E45756", "#4C78A8"])
            plt.ylabel("Mean confidence (success subset)")
            plt.title(f"Confidence Clean vs Attacked vs Mitigated — {metrics['mitigation_id']}")
            plt.ylim(0, 1.0)
            plt.tight_layout()
            plt.savefig(comparison_dir / "clean_vs_attacked_vs_mitigated_confidence.png", dpi=150)
            plt.close()

            plt.figure(figsize=(8, 5))
            iou_vals = [
                _mean([r.clean_iou for r in successes]),
                _mean([r.attacked_iou for r in successes]),
                _mean([r.mitigated_iou for r in successes]),
            ]
            plt.bar(["Clean", "Attacked", "Mitigated"], iou_vals, color=["#54A24B", "#E45756", "#4C78A8"])
            plt.ylabel("Mean IoU (success subset)")
            plt.title(f"IoU Clean vs Attacked vs Mitigated — {metrics['mitigation_id']}")
            plt.ylim(0, 1.0)
            plt.tight_layout()
            plt.savefig(comparison_dir / "clean_vs_attacked_vs_mitigated_iou.png", dpi=150)
            plt.close()


def _annotate_panel(
    image: np.ndarray,
    title: str,
    subset_row: SubsetRow,
    detected: str,
    confidence: float,
    iou: float,
    draw_patch: bool,
) -> np.ndarray:
    panel = image.copy()
    gx = int(round(subset_row.bbox_x))
    gy = int(round(subset_row.bbox_y))
    gw = int(round(subset_row.bbox_width))
    gh = int(round(subset_row.bbox_height))
    cv2.rectangle(panel, (gx, gy), (gx + gw, gy + gh), (0, 255, 0), 2)
    if draw_patch:
        cv2.rectangle(
            panel,
            (subset_row.reconstructed_x1, subset_row.reconstructed_y1),
            (subset_row.reconstructed_x2, subset_row.reconstructed_y2),
            (0, 255, 255),
            2,
        )
    lines = [
        title,
        f"Target: {subset_row.target_class}",
        f"Detected: {detected}",
        f"Conf: {confidence:.3f}",
        f"IoU: {iou:.3f}",
    ]
    y = 22
    for line in lines:
        cv2.putText(panel, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(panel, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        y += 20
    return panel


def draw_mitigation_verification(
    clean_image_path: Path,
    attacked_image_path: Path,
    restored_image_path: Path,
    output_path: Path,
    subset_row: SubsetRow,
    result_row: MitigationResultRow,
) -> None:
    """Create a labelled CLEAN | ATTACKED | MITIGATED comparison image."""
    clean = cv2.imread(str(clean_image_path))
    attacked = cv2.imread(str(attacked_image_path))
    restored = cv2.imread(str(restored_image_path))
    if clean is None:
        raise FileNotFoundError(f"Could not read clean image: {clean_image_path}")
    if attacked is None:
        raise FileNotFoundError(f"Could not read attacked image: {attacked_image_path}")
    if restored is None:
        raise FileNotFoundError(f"Could not read restored image: {restored_image_path}")

    h = max(clean.shape[0], attacked.shape[0], restored.shape[0])
    w = max(clean.shape[1], attacked.shape[1], restored.shape[1])

    def _resize(img: np.ndarray) -> np.ndarray:
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

    clean_p = _annotate_panel(
        _resize(clean),
        "CLEAN",
        subset_row,
        result_row.clean_detected,
        result_row.clean_confidence,
        result_row.clean_iou,
        draw_patch=False,
    )
    attacked_p = _annotate_panel(
        _resize(attacked),
        "ATTACKED",
        subset_row,
        result_row.attacked_detected,
        result_row.attacked_confidence,
        result_row.attacked_iou,
        draw_patch=True,
    )
    mitigated_p = _annotate_panel(
        _resize(restored),
        "MITIGATED",
        subset_row,
        result_row.mitigated_detected,
        result_row.mitigated_confidence,
        result_row.mitigated_iou,
        draw_patch=True,
    )

    gap = 8
    header_h = 36
    canvas = np.full((h + header_h, w * 3 + gap * 2, 3), 255, dtype=np.uint8)
    canvas[header_h : header_h + h, 0:w] = clean_p
    canvas[header_h : header_h + h, w + gap : 2 * w + gap] = attacked_p
    canvas[header_h : header_h + h, 2 * (w + gap) : 2 * (w + gap) + w] = mitigated_p

    if result_row.case_type == "success":
        case_label = "SUCCESS CASE"
        outcome = (
            "RECOVERED"
            if result_row.recovered_after_mitigation == "Yes"
            else "NOT RECOVERED"
        )
    else:
        case_label = "CONTROL CASE"
        outcome = "PRESERVED" if result_row.control_preserved == "Yes" else "REGRESSED"
    header = f"{subset_row.filename} | {case_label} | {outcome}"
    cv2.putText(
        canvas,
        header,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


def create_mitigation_overview(
    verification_dir: Path,
    rows: list[MitigationResultRow],
    output_path: Path,
    max_thumbs: int = 30,
) -> None:
    if not rows:
        return
    sample = rows[:max_thumbs]
    cols = 5
    rows_count = math.ceil(len(sample) / cols)
    thumb_w, thumb_h, label_h, margin = 280, 210, 28, 8
    canvas_w = cols * thumb_w + (cols + 1) * margin
    canvas_h = rows_count * (thumb_h + label_h) + (rows_count + 1) * margin
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for index, row in enumerate(sample):
        grid_r = index // cols
        grid_c = index % cols
        x = margin + grid_c * (thumb_w + margin)
        y = margin + grid_r * (thumb_h + label_h + margin)
        image_path = verification_dir / f"verify_{Path(row.filename).stem}.jpg"
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        thumb = cv2.resize(image, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        canvas[y : y + thumb_h, x : x + thumb_w] = thumb
        label = f"{row.case_type}:{row.filename}"
        cv2.putText(canvas, label, (x, y + thumb_h + 18), font, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(output_path), canvas)


def update_global_mitigations_comparison(metrics: dict[str, float | int | str]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "all_mitigations_comparison.csv"
    existing: list[dict[str, str]] = []
    if csv_path.is_file():
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row.get("mitigation_id") == metrics["mitigation_id"]:
                    continue
                existing.append({col: row.get(col, "") for col in GLOBAL_MITIGATION_COLUMNS})

    new_row = {
        "mitigation_id": str(metrics["mitigation_id"]),
        "source_attack_id": str(metrics["source_attack_id"]),
        "subset_total": str(metrics["subset_total"]),
        "source_attack_success_count": str(metrics["source_attack_success_count"]),
        "recovered_target_count": str(metrics["recovered_target_count"]),
        "detection_recovery_rate": f"{float(metrics['detection_recovery_rate']):.4f}",
        "remaining_failed_targets": str(metrics["remaining_failed_targets"]),
        "control_count": str(metrics["control_count"]),
        "control_preserved_count": str(metrics["control_preserved_count"]),
        "control_preservation_rate": f"{float(metrics['control_preservation_rate']):.4f}",
        "control_regression_count": str(metrics["control_regression_count"]),
        "control_regression_rate": f"{float(metrics['control_regression_rate']):.4f}",
        "person_recovery_rate": f"{float(metrics['person_recovery_rate']):.4f}",
        "car_recovery_rate": f"{float(metrics['car_recovery_rate']):.4f}",
        "mean_confidence_recovery_success": f"{float(metrics['mean_confidence_recovery_success']):.6f}",
        "mean_iou_recovery_success": f"{float(metrics['mean_iou_recovery_success']):.6f}",
        "mean_clean_to_mitigated_confidence_gap_success": (
            f"{float(metrics['mean_clean_to_mitigated_confidence_gap_success']):.6f}"
        ),
        "mean_clean_to_mitigated_iou_gap_success": (
            f"{float(metrics['mean_clean_to_mitigated_iou_gap_success']):.6f}"
        ),
        "target_match_iou_threshold": f"{float(metrics['target_match_iou_threshold']):.2f}",
    }
    all_rows = existing + [new_row]
    all_rows.sort(key=lambda row: row["mitigation_id"])
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=GLOBAL_MITIGATION_COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)

    # Global plot
    if not all_rows:
        return
    ids = [row["mitigation_id"] for row in all_rows]
    recovery = [float(row["detection_recovery_rate"]) for row in all_rows]
    preservation = [float(row["control_preservation_rate"]) for row in all_rows]
    x = np.arange(len(ids))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, recovery, width, label="Detection recovery %", color="#54A24B")
    ax.bar(x + width / 2, preservation, width, label="Control preservation %", color="#4C78A8")
    ax.set_xticks(x, ids)
    ax.set_ylabel("Percent")
    ax.set_title("All Mitigations Comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "all_mitigations_comparison.png", dpi=150)
    plt.close(fig)
