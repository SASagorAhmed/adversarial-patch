#!/usr/bin/env python3
"""Automatic clean-vs-attacked comparison reporting for the attack pipeline."""

from __future__ import annotations

import csv
import statistics
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from patch_config import patch_spec_for_filename
from pipeline_core import (
    ATTACKS_DIR,
    EXPERIMENT_ROOT,
    TARGET_MATCH_IOU_THRESHOLD,
    AttackResultRow,
    SelectedImage,
)

import cv2

GLOBAL_COMPARISON_CSV = EXPERIMENT_ROOT / "results" / "all_attacks_comparison.csv"
GLOBAL_COMPARISON_PNG = EXPERIMENT_ROOT / "results" / "all_attacks_comparison.png"

CLEAN_VS_ATTACKED_COLUMNS = [
    "attack_id",
    "image_id",
    "filename",
    "target_class",
    "clean_detected",
    "attacked_detected",
    "clean_confidence",
    "attacked_confidence",
    "confidence_drop",
    "confidence_drop_percent",
    "clean_iou",
    "attacked_iou",
    "iou_drop",
    "clean_total_detections",
    "attacked_total_detections",
    "detection_count_change",
    "eligible_for_asr",
    "attack_success",
    "patch_filename",
    "patch_size_pixels",
    "patch_scale",
]

GLOBAL_COMPARISON_COLUMNS = [
    "attack_id",
    "num_images",
    "random_seed",
    "patch_filename",
    "patch_seed",
    "target_match_iou_threshold",
    "person_count",
    "car_count",
    "clean_detected",
    "attacked_detected",
    "asr_eligible",
    "attack_successes",
    "attack_success_rate",
    "person_asr",
    "car_asr",
    "previous_attack_overlap_count",
    "previous_attack_overlap_percent",
    "mean_clean_confidence",
    "mean_attacked_confidence",
    "mean_confidence_drop",
    "mean_confidence_drop_percent",
    "mean_clean_iou",
    "mean_attacked_iou",
    "mean_iou_drop",
    "avg_clean_detections",
    "avg_attacked_detections",
    "avg_detection_count_change",
]


@dataclass
class ComparisonRow:
    attack_id: str
    image_id: str
    filename: str
    target_class: str
    clean_detected: str
    attacked_detected: str
    clean_confidence: float
    attacked_confidence: float
    confidence_drop: float
    confidence_drop_percent: float | None
    clean_iou: float
    attacked_iou: float
    iou_drop: float
    clean_total_detections: int
    attacked_total_detections: int
    detection_count_change: int
    eligible_for_asr: str
    attack_success: str
    patch_filename: str
    patch_size_pixels: int
    patch_scale: float
    clean_best_same_class_confidence: float
    clean_best_same_class_iou: float
    attacked_best_same_class_confidence: float
    attacked_best_same_class_iou: float


@dataclass
class ClassComparisonStats:
    target_class: str
    num_images: int
    clean_detected: int
    attacked_detected: int
    attack_successes: int
    asr: float
    mean_clean_confidence: float
    mean_attacked_confidence: float
    mean_confidence_drop: float
    mean_clean_iou: float
    mean_attacked_iou: float


@dataclass
class AttackComparisonStats:
    attack_id: str
    patch_filename: str
    num_images: int
    person_count: int
    car_count: int
    clean_detected: int
    clean_missed: int
    attacked_detected: int
    attacked_missed: int
    asr_eligible: int
    attack_successes: int
    attack_success_rate: float
    mean_clean_confidence: float
    mean_attacked_confidence: float
    mean_confidence_drop: float
    mean_confidence_drop_percent: float
    median_confidence_drop: float
    mean_clean_iou: float
    mean_attacked_iou: float
    mean_iou_drop: float
    avg_clean_detections: float
    avg_attacked_detections: float
    avg_detection_count_change: float
    person_stats: ClassComparisonStats
    car_stats: ClassComparisonStats


def confidence_drop_percent(clean_confidence: float, attacked_confidence: float) -> float | None:
    if clean_confidence <= 0.0:
        return None
    return ((clean_confidence - attacked_confidence) / clean_confidence) * 100.0


def build_comparison_rows(attack_rows: list[AttackResultRow]) -> list[ComparisonRow]:
    """Build comparison records from master attack result rows."""
    comparison_rows: list[ComparisonRow] = []
    for row in attack_rows:
        drop_pct = confidence_drop_percent(row.clean_confidence, row.attacked_confidence)
        comparison_rows.append(
            ComparisonRow(
                attack_id=row.attack_id,
                image_id=row.image_id,
                filename=row.filename,
                target_class=row.target_class,
                clean_detected=row.clean_detected,
                attacked_detected=row.attacked_detected,
                clean_confidence=row.clean_confidence,
                attacked_confidence=row.attacked_confidence,
                confidence_drop=row.confidence_drop,
                confidence_drop_percent=drop_pct,
                clean_iou=row.clean_iou,
                attacked_iou=row.attacked_iou,
                iou_drop=row.iou_drop,
                clean_total_detections=row.clean_total_detections,
                attacked_total_detections=row.attacked_total_detections,
                detection_count_change=row.detection_count_change,
                eligible_for_asr=row.eligible_for_asr,
                attack_success=row.attack_success,
                patch_filename=row.patch_filename,
                patch_size_pixels=row.patch_size_pixels,
                patch_scale=row.patch_scale,
                clean_best_same_class_confidence=row.clean_best_same_class_confidence,
                clean_best_same_class_iou=row.clean_best_same_class_iou,
                attacked_best_same_class_confidence=row.attacked_best_same_class_confidence,
                attacked_best_same_class_iou=row.attacked_best_same_class_iou,
            )
        )
    return comparison_rows


def _eligible_rows(rows: list[ComparisonRow]) -> list[ComparisonRow]:
    return [row for row in rows if row.eligible_for_asr == "Yes"]


def _rows_for_class(rows: list[ComparisonRow], target_class: str) -> list[ComparisonRow]:
    return [row for row in rows if row.target_class == target_class]


def _compute_class_stats(rows: list[ComparisonRow], target_class: str) -> ClassComparisonStats:
    class_rows = _rows_for_class(rows, target_class)
    eligible = [row for row in class_rows if row.eligible_for_asr == "Yes"]
    successes = [row for row in class_rows if row.attack_success == "Yes"]
    asr = (len(successes) / len(eligible) * 100.0) if eligible else 0.0
    return ClassComparisonStats(
        target_class=target_class,
        num_images=len(class_rows),
        clean_detected=sum(1 for row in class_rows if row.clean_detected == "Yes"),
        attacked_detected=sum(1 for row in class_rows if row.attacked_detected == "Yes"),
        attack_successes=len(successes),
        asr=asr,
        mean_clean_confidence=(
            statistics.fmean(row.clean_confidence for row in eligible) if eligible else 0.0
        ),
        mean_attacked_confidence=(
            statistics.fmean(row.attacked_confidence for row in eligible) if eligible else 0.0
        ),
        mean_confidence_drop=(
            statistics.fmean(row.confidence_drop for row in eligible) if eligible else 0.0
        ),
        mean_clean_iou=(statistics.fmean(row.clean_iou for row in eligible) if eligible else 0.0),
        mean_attacked_iou=(
            statistics.fmean(row.attacked_iou for row in eligible) if eligible else 0.0
        ),
    )


def compute_attack_comparison_stats(
    attack_id: str,
    patch_filename: str,
    comparison_rows: list[ComparisonRow],
) -> AttackComparisonStats:
    eligible = _eligible_rows(comparison_rows)
    successes = [row for row in comparison_rows if row.attack_success == "Yes"]
    drop_percent_values = [
        row.confidence_drop_percent
        for row in eligible
        if row.confidence_drop_percent is not None
    ]
    person_count = sum(1 for row in comparison_rows if row.target_class == "person")
    car_count = len(comparison_rows) - person_count
    clean_detected = sum(1 for row in comparison_rows if row.clean_detected == "Yes")
    attacked_detected = sum(1 for row in comparison_rows if row.attacked_detected == "Yes")

    return AttackComparisonStats(
        attack_id=attack_id,
        patch_filename=patch_filename,
        num_images=len(comparison_rows),
        person_count=person_count,
        car_count=car_count,
        clean_detected=clean_detected,
        clean_missed=len(comparison_rows) - clean_detected,
        attacked_detected=attacked_detected,
        attacked_missed=len(comparison_rows) - attacked_detected,
        asr_eligible=len(eligible),
        attack_successes=len(successes),
        attack_success_rate=(
            (len(successes) / len(eligible) * 100.0) if eligible else 0.0
        ),
        mean_clean_confidence=(
            statistics.fmean(row.clean_confidence for row in eligible) if eligible else 0.0
        ),
        mean_attacked_confidence=(
            statistics.fmean(row.attacked_confidence for row in eligible) if eligible else 0.0
        ),
        mean_confidence_drop=(
            statistics.fmean(row.confidence_drop for row in eligible) if eligible else 0.0
        ),
        mean_confidence_drop_percent=(
            statistics.fmean(drop_percent_values) if drop_percent_values else 0.0
        ),
        median_confidence_drop=(
            statistics.median([row.confidence_drop for row in eligible]) if eligible else 0.0
        ),
        mean_clean_iou=(statistics.fmean(row.clean_iou for row in eligible) if eligible else 0.0),
        mean_attacked_iou=(
            statistics.fmean(row.attacked_iou for row in eligible) if eligible else 0.0
        ),
        mean_iou_drop=(statistics.fmean(row.iou_drop for row in eligible) if eligible else 0.0),
        avg_clean_detections=(
            statistics.fmean(row.clean_total_detections for row in comparison_rows)
            if comparison_rows
            else 0.0
        ),
        avg_attacked_detections=(
            statistics.fmean(row.attacked_total_detections for row in comparison_rows)
            if comparison_rows
            else 0.0
        ),
        avg_detection_count_change=(
            statistics.fmean(row.detection_count_change for row in comparison_rows)
            if comparison_rows
            else 0.0
        ),
        person_stats=_compute_class_stats(comparison_rows, "person"),
        car_stats=_compute_class_stats(comparison_rows, "car"),
    )


def write_clean_vs_attacked_csv(path: Path, rows: list[ComparisonRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CLEAN_VS_ATTACKED_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "attack_id": row.attack_id,
                    "image_id": row.image_id,
                    "filename": row.filename,
                    "target_class": row.target_class,
                    "clean_detected": row.clean_detected,
                    "attacked_detected": row.attacked_detected,
                    "clean_confidence": f"{row.clean_confidence:.6f}",
                    "attacked_confidence": f"{row.attacked_confidence:.6f}",
                    "confidence_drop": f"{row.confidence_drop:.6f}",
                    "confidence_drop_percent": (
                        f"{row.confidence_drop_percent:.6f}"
                        if row.confidence_drop_percent is not None
                        else ""
                    ),
                    "clean_iou": f"{row.clean_iou:.6f}",
                    "attacked_iou": f"{row.attacked_iou:.6f}",
                    "iou_drop": f"{row.iou_drop:.6f}",
                    "clean_total_detections": row.clean_total_detections,
                    "attacked_total_detections": row.attacked_total_detections,
                    "detection_count_change": row.detection_count_change,
                    "eligible_for_asr": row.eligible_for_asr,
                    "attack_success": row.attack_success,
                    "patch_filename": row.patch_filename,
                    "patch_size_pixels": row.patch_size_pixels,
                    "patch_scale": row.patch_scale,
                }
            )


def _format_class_section(stats: ClassComparisonStats) -> list[str]:
    return [
        f"{stats.target_class}:",
        f"  number of images: {stats.num_images}",
        f"  clean detected: {stats.clean_detected}",
        f"  attacked detected: {stats.attacked_detected}",
        f"  attack successes: {stats.attack_successes}",
        f"  ASR: {stats.asr:.2f}%",
        f"  mean clean confidence: {stats.mean_clean_confidence:.4f}",
        f"  mean attacked confidence: {stats.mean_attacked_confidence:.4f}",
        f"  mean confidence drop: {stats.mean_confidence_drop:.4f}",
        f"  mean clean IoU: {stats.mean_clean_iou:.4f}",
        f"  mean attacked IoU: {stats.mean_attacked_iou:.4f}",
        "",
    ]


def write_comparison_summary(path: Path, stats: AttackComparisonStats) -> None:
    lines = [
        "Clean vs Attacked Comparison Summary",
        "=" * 72,
        f"Attack ID: {stats.attack_id}",
        f"Patch filename: {stats.patch_filename}",
        f"Target match IoU threshold: {TARGET_MATCH_IOU_THRESHOLD:.2f}",
        f"Number of processed images: {stats.num_images}",
        "",
        f"Total images: {stats.num_images}",
        f"Person images: {stats.person_count}",
        f"Car images: {stats.car_count}",
        "",
        f"Valid clean target detections: {stats.clean_detected}",
        f"Clean target misses: {stats.clean_missed}",
        "",
        f"Valid attacked target detections: {stats.attacked_detected}",
        f"Attacked target misses: {stats.attacked_missed}",
        "",
        f"ASR eligible targets: {stats.asr_eligible}",
        f"Successful attacks: {stats.attack_successes}",
        f"Attack Success Rate: {stats.attack_success_rate:.2f}%",
        "",
        f"Person ASR: {stats.person_stats.asr:.2f}%",
        f"Car ASR: {stats.car_stats.asr:.2f}%",
        "",
        f"Mean clean confidence: {stats.mean_clean_confidence:.4f}",
        f"Mean attacked confidence: {stats.mean_attacked_confidence:.4f}",
        f"Mean confidence drop: {stats.mean_confidence_drop:.4f}",
        f"Mean confidence drop percentage: {stats.mean_confidence_drop_percent:.4f}",
        f"Median confidence drop: {stats.median_confidence_drop:.4f}",
        "",
        f"Mean clean IoU: {stats.mean_clean_iou:.4f}",
        f"Mean attacked IoU: {stats.mean_attacked_iou:.4f}",
        f"Mean IoU drop: {stats.mean_iou_drop:.4f}",
        "",
        f"Average clean detections/image: {stats.avg_clean_detections:.4f}",
        f"Average attacked detections/image: {stats.avg_attacked_detections:.4f}",
        f"Average detection-count change: {stats.avg_detection_count_change:.4f}",
        "",
        "Person vs Car Analysis",
        "-" * 72,
        "",
        *_format_class_section(stats.person_stats),
        *_format_class_section(stats.car_stats),
        "Note: Different attacks may use different random image samples.",
        "Use identical random_seed and num_images for direct patch-to-patch comparison.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_confidence_comparison(output_path: Path, stats: AttackComparisonStats) -> None:
    labels = ["Clean", "Attacked"]
    values = [stats.mean_clean_confidence, stats.mean_attacked_confidence]
    plt.figure(figsize=(7, 5))
    bars = plt.bar(labels, values, color=["#4C78A8", "#F58518"])
    plt.ylabel("Mean Confidence (ASR-eligible targets)")
    plt.title(f"Mean Confidence Comparison — {stats.attack_id}")
    plt.ylim(0, max(1.0, max(values) * 1.15 if values else 1.0))
    for bar, value in zip(bars, values, strict=True):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3f}", ha="center", va="bottom")
    _save_figure(output_path)


def plot_confidence_drop_distribution(
    output_path: Path,
    rows: list[ComparisonRow],
    attack_id: str,
) -> None:
    drops = [row.confidence_drop for row in rows if row.eligible_for_asr == "Yes"]
    plt.figure(figsize=(8, 5))
    if drops:
        plt.hist(drops, bins=min(20, max(5, len(drops) // 2)), edgecolor="black", alpha=0.75)
    plt.xlabel("Confidence Drop")
    plt.ylabel("Number of Images")
    plt.title(f"Confidence Drop Distribution — {attack_id}")
    plt.grid(axis="y", alpha=0.3)
    _save_figure(output_path)


def plot_iou_comparison(output_path: Path, stats: AttackComparisonStats) -> None:
    labels = ["Clean", "Attacked"]
    values = [stats.mean_clean_iou, stats.mean_attacked_iou]
    plt.figure(figsize=(7, 5))
    bars = plt.bar(labels, values, color=["#54A24B", "#E45756"])
    plt.ylabel("Mean IoU (ASR-eligible targets)")
    plt.title(f"Mean IoU Comparison — {stats.attack_id}")
    plt.ylim(0, 1.0)
    for bar, value in zip(bars, values, strict=True):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3f}", ha="center", va="bottom")
    _save_figure(output_path)


def plot_detection_comparison(output_path: Path, stats: AttackComparisonStats) -> None:
    labels = [
        "Clean Targets\nDetected",
        "Attacked Targets\nDetected",
        "Avg Clean\nDetections/Image",
        "Avg Attacked\nDetections/Image",
    ]
    values = [
        stats.clean_detected,
        stats.attacked_detected,
        stats.avg_clean_detections,
        stats.avg_attacked_detections,
    ]
    plt.figure(figsize=(9, 5))
    bars = plt.bar(labels, values, color=["#72B7B2", "#EECA3B", "#B279A2", "#FF9DA6"])
    plt.ylabel("Count / Average")
    plt.title(f"Detection Comparison — {stats.attack_id}")
    for bar, value in zip(bars, values, strict=True):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom",
        )
    _save_figure(output_path)


def plot_person_vs_car_comparison(output_path: Path, stats: AttackComparisonStats) -> None:
    classes = ["person", "car"]
    class_stats = [stats.person_stats, stats.car_stats]
    metrics = ["ASR", "Mean Conf Drop", "Mean Clean IoU", "Mean Attacked IoU"]
    person_values = [
        stats.person_stats.asr,
        stats.person_stats.mean_confidence_drop,
        stats.person_stats.mean_clean_iou,
        stats.person_stats.mean_attacked_iou,
    ]
    car_values = [
        stats.car_stats.asr,
        stats.car_stats.mean_confidence_drop,
        stats.car_stats.mean_clean_iou,
        stats.car_stats.mean_attacked_iou,
    ]

    x = np.arange(len(metrics))
    width = 0.35
    plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, person_values, width, label="person", color="#4C78A8")
    plt.bar(x + width / 2, car_values, width, label="car", color="#F58518")
    plt.xticks(x, metrics)
    plt.ylabel("Value")
    plt.title(f"Person vs Car Comparison — {stats.attack_id}")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    _save_figure(output_path)


def plot_attack_success_overview(output_path: Path, stats: AttackComparisonStats) -> None:
    unsuccessful = stats.asr_eligible - stats.attack_successes
    labels = ["Successful Attacks", "Unsuccessful Attacks"]
    sizes = [stats.attack_successes, max(0, unsuccessful)]
    colors = ["#54A24B", "#E45756"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    if stats.asr_eligible > 0:
        axes[0].pie(
            sizes,
            labels=labels,
            autopct="%1.1f%%",
            colors=colors,
            startangle=90,
        )
    else:
        axes[0].text(0.5, 0.5, "No ASR-eligible targets", ha="center", va="center")
    axes[0].set_title("Attack Outcomes (ASR-eligible)")

    bar_labels = ["ASR Eligible", "Successful", "Unsuccessful"]
    bar_values = [stats.asr_eligible, stats.attack_successes, max(0, unsuccessful)]
    bars = axes[1].bar(bar_labels, bar_values, color=["#72B7B2", "#54A24B", "#E45756"])
    axes[1].set_ylabel("Number of Targets")
    axes[1].set_title(f"Attack Success Rate: {stats.attack_success_rate:.2f}%")
    for bar, value in zip(bars, bar_values, strict=True):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(value), ha="center", va="bottom")

    fig.suptitle(f"Attack Success Overview — {stats.attack_id}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_comparison_plots(
    comparison_dir: Path,
    stats: AttackComparisonStats,
    comparison_rows: list[ComparisonRow],
) -> None:
    comparison_dir.mkdir(parents=True, exist_ok=True)
    plot_confidence_comparison(comparison_dir / "confidence_comparison.png", stats)
    plot_confidence_drop_distribution(
        comparison_dir / "confidence_drop_distribution.png",
        comparison_rows,
        stats.attack_id,
    )
    plot_iou_comparison(comparison_dir / "iou_comparison.png", stats)
    plot_detection_comparison(comparison_dir / "detection_comparison.png", stats)
    plot_person_vs_car_comparison(comparison_dir / "person_vs_car_comparison.png", stats)
    plot_attack_success_overview(comparison_dir / "attack_success_overview.png", stats)


def _attack_is_completed(attack_id: str) -> bool:
    attack_dir = ATTACKS_DIR / attack_id
    if not attack_dir.is_dir():
        return False
    if (attack_dir / "FAILED.txt").is_file():
        return False
    return (attack_dir / "results" / "attack_results.csv").is_file()


def _stats_to_global_row(
    stats: AttackComparisonStats,
    random_seed: int,
    patch_seed: int | str,
    *,
    previous_attack_overlap_count: int = 0,
    previous_attack_overlap_percent: float = 0.0,
) -> dict[str, str | int | float]:
    return {
        "attack_id": stats.attack_id,
        "num_images": stats.num_images,
        "random_seed": random_seed,
        "patch_filename": stats.patch_filename,
        "patch_seed": patch_seed,
        "target_match_iou_threshold": f"{TARGET_MATCH_IOU_THRESHOLD:.2f}",
        "person_count": stats.person_count,
        "car_count": stats.car_count,
        "clean_detected": stats.clean_detected,
        "attacked_detected": stats.attacked_detected,
        "asr_eligible": stats.asr_eligible,
        "attack_successes": stats.attack_successes,
        "attack_success_rate": f"{stats.attack_success_rate:.4f}",
        "person_asr": f"{stats.person_stats.asr:.4f}",
        "car_asr": f"{stats.car_stats.asr:.4f}",
        "previous_attack_overlap_count": previous_attack_overlap_count,
        "previous_attack_overlap_percent": f"{previous_attack_overlap_percent:.4f}",
        "mean_clean_confidence": f"{stats.mean_clean_confidence:.6f}",
        "mean_attacked_confidence": f"{stats.mean_attacked_confidence:.6f}",
        "mean_confidence_drop": f"{stats.mean_confidence_drop:.6f}",
        "mean_confidence_drop_percent": f"{stats.mean_confidence_drop_percent:.6f}",
        "mean_clean_iou": f"{stats.mean_clean_iou:.6f}",
        "mean_attacked_iou": f"{stats.mean_attacked_iou:.6f}",
        "mean_iou_drop": f"{stats.mean_iou_drop:.6f}",
        "avg_clean_detections": f"{stats.avg_clean_detections:.6f}",
        "avg_attacked_detections": f"{stats.avg_attacked_detections:.6f}",
        "avg_detection_count_change": f"{stats.avg_detection_count_change:.6f}",
    }


def update_global_comparison_csv(
    stats: AttackComparisonStats,
    random_seed: int,
    patch_seed: int | str,
    *,
    previous_attack_overlap_count: int = 0,
    previous_attack_overlap_percent: float = 0.0,
) -> None:
    GLOBAL_COMPARISON_CSV.parent.mkdir(parents=True, exist_ok=True)
    existing_rows: list[dict[str, str]] = []
    if GLOBAL_COMPARISON_CSV.is_file():
        with GLOBAL_COMPARISON_CSV.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row.get("attack_id") == stats.attack_id:
                    continue
                migrated = {column: row.get(column, "") for column in GLOBAL_COMPARISON_COLUMNS}
                if not migrated.get("previous_attack_overlap_count"):
                    migrated["previous_attack_overlap_count"] = "0"
                if not migrated.get("previous_attack_overlap_percent"):
                    migrated["previous_attack_overlap_percent"] = "0.0000"
                existing_rows.append(migrated)

    new_row = _stats_to_global_row(
        stats,
        random_seed,
        patch_seed,
        previous_attack_overlap_count=previous_attack_overlap_count,
        previous_attack_overlap_percent=previous_attack_overlap_percent,
    )
    all_rows = existing_rows + [{key: str(value) for key, value in new_row.items()}]
    all_rows.sort(key=lambda row: row["attack_id"])

    with GLOBAL_COMPARISON_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=GLOBAL_COMPARISON_COLUMNS)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)


def _load_completed_global_rows() -> list[dict[str, str]]:
    if not GLOBAL_COMPARISON_CSV.is_file():
        return []
    with GLOBAL_COMPARISON_CSV.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != GLOBAL_COMPARISON_COLUMNS:
            return []
        rows = [row for row in reader if _attack_is_completed(row.get("attack_id", ""))]
    rows.sort(key=lambda row: row["attack_id"])
    return rows


def update_global_comparison_plot() -> None:
    rows = _load_completed_global_rows()
    if not rows:
        return

    attack_ids = [row["attack_id"] for row in rows]
    asr_values = [float(row["attack_success_rate"]) for row in rows]
    conf_drop_values = [float(row["mean_confidence_drop"]) for row in rows]
    iou_drop_values = [float(row["mean_iou_drop"]) for row in rows]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    x = np.arange(len(attack_ids))

    axes[0].bar(x, asr_values, color="#54A24B")
    axes[0].set_xticks(x, attack_ids, rotation=30, ha="right")
    axes[0].set_ylabel("ASR (%)")
    axes[0].set_title("Attack Success Rate")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(x, conf_drop_values, color="#4C78A8")
    axes[1].set_xticks(x, attack_ids, rotation=30, ha="right")
    axes[1].set_ylabel("Mean Confidence Drop")
    axes[1].set_title("Mean Confidence Drop")
    axes[1].grid(axis="y", alpha=0.3)

    axes[2].bar(x, iou_drop_values, color="#F58518")
    axes[2].set_xticks(x, attack_ids, rotation=30, ha="right")
    axes[2].set_ylabel("Mean IoU Drop")
    axes[2].set_title("Mean IoU Drop")
    axes[2].grid(axis="y", alpha=0.3)

    fig.suptitle("All Attacks Comparison")
    plt.tight_layout()
    GLOBAL_COMPARISON_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(GLOBAL_COMPARISON_PNG, dpi=150, bbox_inches="tight")
    plt.close()


def resolve_patch_seed(patch_filename: str) -> int | str:
    try:
        return patch_spec_for_filename(patch_filename).seed
    except ValueError:
        return "unknown"


def run_comparison_reporting(
    attack_id: str,
    random_seed: int,
    patch_filename: str,
    comparison_rows: list[ComparisonRow],
    results_dir: Path,
    *,
    previous_attack_overlap_count: int = 0,
    previous_attack_overlap_percent: float = 0.0,
) -> AttackComparisonStats:
    """Generate per-attack comparison outputs and update global comparison files."""
    comparison_dir = results_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    stats = compute_attack_comparison_stats(attack_id, patch_filename, comparison_rows)
    write_clean_vs_attacked_csv(comparison_dir / "clean_vs_attacked.csv", comparison_rows)
    write_comparison_summary(comparison_dir / "comparison_summary.txt", stats)
    generate_comparison_plots(comparison_dir, stats, comparison_rows)

    patch_seed = resolve_patch_seed(patch_filename)
    update_global_comparison_csv(
        stats,
        random_seed,
        patch_seed,
        previous_attack_overlap_count=previous_attack_overlap_count,
        previous_attack_overlap_percent=previous_attack_overlap_percent,
    )
    update_global_comparison_plot()
    return stats


def draw_verification_image(
    patched_image_path: Path,
    output_path: Path,
    item: SelectedImage,
    row: ComparisonRow,
    patch_rect: tuple[float, float, float, float],
) -> None:
    """Draw a verification image using comparison record values."""
    image = cv2.imread(str(patched_image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read patched image: {patched_image_path}")

    gx, gy, gw, gh = [int(round(v)) for v in (item.bbox_x, item.bbox_y, item.bbox_width, item.bbox_height)]
    cv2.rectangle(image, (gx, gy), (gx + gw, gy + gh), (0, 255, 0), 2)

    px, py, pw, ph = [int(round(v)) for v in patch_rect]
    cv2.rectangle(image, (px, py), (px + pw, py + ph), (0, 255, 255), 2)

    drop_pct = (
        f"{row.confidence_drop_percent:.1f}%"
        if row.confidence_drop_percent is not None
        else "N/A"
    )
    lines = [
        item.filename,
        f"Target: {item.target_class}",
        f"Clean: {row.clean_detected} conf={row.clean_confidence:.3f} IoU={row.clean_iou:.3f}",
        f"Attacked: {row.attacked_detected} conf={row.attacked_confidence:.3f} IoU={row.attacked_iou:.3f}",
        f"Conf drop: {row.confidence_drop:.3f} ({drop_pct})",
        f"Attack Success: {row.attack_success}",
    ]
    if row.eligible_for_asr == "No":
        lines.append("NOT ELIGIBLE FOR ASR")
    elif row.attack_success == "Yes":
        lines.append("ATTACK SUCCESS")
        lines.append("TARGET MISSED AFTER ATTACK")
    if row.clean_detected == "No":
        lines.append(
            f"Best same-class (clean): conf={row.clean_best_same_class_confidence:.3f} "
            f"IoU={row.clean_best_same_class_iou:.3f}"
        )
    if row.attacked_detected == "No" and row.eligible_for_asr == "Yes":
        lines.append(
            f"Best same-class (attacked): conf={row.attacked_best_same_class_confidence:.3f} "
            f"IoU={row.attacked_best_same_class_iou:.3f}"
        )

    font = cv2.FONT_HERSHEY_SIMPLEX
    y = 20
    for line in lines:
        cv2.putText(image, line, (8, y), font, 0.45, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(image, line, (8, y), font, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        y += 17
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)

