#!/usr/bin/env python3
"""Analyze Hyper-YOLO prediction labels for the first 500 COCO val2017 images."""

from __future__ import annotations

import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent
LABELS_DIR = ROOT / "runs" / "val_500" / "labels"
OUTPUT_DIR = ROOT / "runs" / "analysis_500"

COCO_NAMES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush",
}

REQUIRED_OUTPUTS = [
    "summary.csv",
    "class_statistics.csv",
    "confidence_statistics.csv",
    "baseline_metrics.json",
    "summary_report.txt",
    "detections_by_class.png",
    "confidence_histogram.png",
    "top10_classes_pie.png",
]


def percentile(sorted_vals: list[float], p: float) -> float:
    """Linear interpolation percentile for a sorted list."""
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    k = (len(sorted_vals) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(sorted_vals[int(k)])
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return float(d0 + d1)


def parse_labels(labels_dir: Path) -> tuple[list[dict], list[str], int, int, int]:
    """Parse all YOLO prediction label files."""
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    files = sorted(labels_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt label files found in {labels_dir}")

    detections: list[dict] = []
    malformed: list[str] = []
    images_with = 0
    images_without = 0

    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            images_without += 1
            continue

        file_has_valid = False
        for line_no, line in enumerate(text.splitlines(), start=1):
            raw = line.strip()
            if not raw:
                continue
            parts = raw.split()
            if len(parts) != 6:
                malformed.append(f"{path.name}:{line_no}: expected 6 fields, got {len(parts)} ({raw})")
                continue
            try:
                class_id = int(float(parts[0]))
                x_c, y_c, w, h, conf = map(float, parts[1:])
            except ValueError:
                malformed.append(f"{path.name}:{line_no}: non-numeric values ({raw})")
                continue
            if not (0 <= class_id <= 79):
                malformed.append(f"{path.name}:{line_no}: class_id out of range ({class_id})")
                continue
            if not (0.0 <= conf <= 1.0):
                malformed.append(f"{path.name}:{line_no}: confidence out of range ({conf})")
                continue
            detections.append(
                {
                    "image": path.stem,
                    "class_id": class_id,
                    "class_name": COCO_NAMES[class_id],
                    "x_center": x_c,
                    "y_center": y_c,
                    "width": w,
                    "height": h,
                    "confidence": conf,
                }
            )
            file_has_valid = True

        if file_has_valid:
            images_with += 1
        else:
            images_without += 1

    return detections, malformed, len(files), images_with, images_without


def compute_stats(
    detections: list[dict],
    processed_images: int,
    images_with: int,
    images_without: int,
) -> dict:
    """Compute all summary, confidence, and class statistics."""
    confidences = [d["confidence"] for d in detections]
    total_objects = len(detections)
    avg_per_image = total_objects / processed_images if processed_images else 0.0

    if confidences:
        conf_sorted = sorted(confidences)
        conf_stats = {
            "count": total_objects,
            "mean": statistics.fmean(confidences),
            "median": statistics.median(confidences),
            "std": statistics.pstdev(confidences) if len(confidences) > 1 else 0.0,
            "min": min(confidences),
            "max": max(confidences),
            "p25": percentile(conf_sorted, 0.25),
            "p75": percentile(conf_sorted, 0.75),
        }
    else:
        conf_stats = {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "p25": float("nan"),
            "p75": float("nan"),
        }

    class_confs: dict[int, list[float]] = defaultdict(list)
    for d in detections:
        class_confs[d["class_id"]].append(d["confidence"])

    class_rows = []
    for cid in range(80):
        vals = class_confs.get(cid, [])
        count = len(vals)
        pct = (100.0 * count / total_objects) if total_objects else 0.0
        class_rows.append(
            {
                "class_id": cid,
                "class_name": COCO_NAMES[cid],
                "detections": count,
                "percentage": pct,
                "avg_confidence": statistics.fmean(vals) if vals else float("nan"),
                "min_confidence": min(vals) if vals else float("nan"),
                "max_confidence": max(vals) if vals else float("nan"),
            }
        )

    by_count = sorted(class_rows, key=lambda r: (-r["detections"], r["class_id"]))
    top10 = [r for r in by_count if r["detections"] > 0][:10]
    least = [r for r in reversed(by_count) if r["detections"] > 0][:10]
    zero = [r for r in class_rows if r["detections"] == 0]
    unique_classes = sum(1 for r in class_rows if r["detections"] > 0)

    summary = {
        "dataset": "COCO 2017 Validation (first 500 images)",
        "model": "Hyper-YOLO-N (pretrained, weights/hyper-yolon.pt)",
        "processed_images": processed_images,
        "images_with_detections": images_with,
        "images_without_detections": images_without,
        "total_detected_objects": total_objects,
        "average_detections_per_image": avg_per_image,
        "unique_detected_classes": unique_classes,
        "mean_confidence": conf_stats["mean"],
        "median_confidence": conf_stats["median"],
        "std_confidence": conf_stats["std"],
        "min_confidence": conf_stats["min"],
        "max_confidence": conf_stats["max"],
        "p25_confidence": conf_stats["p25"],
        "p75_confidence": conf_stats["p75"],
    }

    return {
        "summary": summary,
        "confidence": conf_stats,
        "class_rows": class_rows,
        "top10": top10,
        "least": least,
        "zero": zero,
    }


def write_csv_outputs(stats: dict, output_dir: Path) -> None:
    """Write summary, class, and confidence CSV files."""
    summary_rows = [{"metric": k, "value": v} for k, v in stats["summary"].items()]
    pd.DataFrame(summary_rows).to_csv(output_dir / "summary.csv", index=False)

    class_df = pd.DataFrame(stats["class_rows"])
    class_df.to_csv(output_dir / "class_statistics.csv", index=False)

    conf_rows = [{"metric": k, "value": v} for k, v in stats["confidence"].items()]
    pd.DataFrame(conf_rows).to_csv(output_dir / "confidence_statistics.csv", index=False)


def write_json(stats: dict, output_dir: Path) -> None:
    """Write baseline_metrics.json."""
    payload = {
        "experiment": {
            "dataset": stats["summary"]["dataset"],
            "model": stats["summary"]["model"],
            "labels_dir": str(LABELS_DIR),
            "output_dir": str(output_dir),
        },
        "general": {
            "processed_images": stats["summary"]["processed_images"],
            "images_with_detections": stats["summary"]["images_with_detections"],
            "images_without_detections": stats["summary"]["images_without_detections"],
            "total_detected_objects": stats["summary"]["total_detected_objects"],
            "average_detections_per_image": stats["summary"]["average_detections_per_image"],
            "unique_detected_classes": stats["summary"]["unique_detected_classes"],
        },
        "confidence": stats["confidence"],
        "top10_classes": [
            {
                "rank": i + 1,
                "class_id": r["class_id"],
                "class_name": r["class_name"],
                "detections": r["detections"],
                "percentage": r["percentage"],
                "avg_confidence": r["avg_confidence"],
            }
            for i, r in enumerate(stats["top10"])
        ],
        "zero_detection_classes": [
            {"class_id": r["class_id"], "class_name": r["class_name"]} for r in stats["zero"]
        ],
        "least_detected_classes": [
            {
                "class_id": r["class_id"],
                "class_name": r["class_name"],
                "detections": r["detections"],
            }
            for r in stats["least"]
        ],
    }
    (output_dir / "baseline_metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def write_report(stats: dict, malformed: list[str], output_dir: Path) -> None:
    """Write publication-ready summary_report.txt."""
    s = stats["summary"]
    c = stats["confidence"]
    lines = [
        "Hyper-YOLO Detection Analysis Report",
        "=" * 72,
        "",
        "1. Dataset Information",
        "-" * 72,
        f"Dataset: {s['dataset']}",
        "Split: COCO val2017 subset (sorted filenames, first 500 images)",
        "Task: Object detection",
        "Annotation format analyzed: YOLO prediction labels with confidence",
        "",
        "2. Model Information",
        "-" * 72,
        f"Model: {s['model']}",
        "Architecture: Hyper-YOLO-N (hypergraph-enhanced YOLO detector)",
        "Training status: Pretrained weights only; no fine-tuning in this experiment",
        "",
        "3. Image-Level Results",
        "-" * 72,
        f"Processed images: {s['processed_images']}",
        f"Images with detections: {s['images_with_detections']}",
        f"Images without detections: {s['images_without_detections']}",
        f"Total detected objects: {s['total_detected_objects']}",
        f"Average detections per image: {s['average_detections_per_image']:.4f}",
        f"Unique detected classes: {s['unique_detected_classes']} / 80",
        "",
        "4. Confidence Statistics",
        "-" * 72,
        f"Mean confidence: {c['mean']:.6f}",
        f"Median confidence: {c['median']:.6f}",
        f"Standard deviation: {c['std']:.6f}",
        f"Minimum confidence: {c['min']:.6f}",
        f"Maximum confidence: {c['max']:.6f}",
        f"25th percentile: {c['p25']:.6f}",
        f"75th percentile: {c['p75']:.6f}",
        "",
        "5. Top 10 Detected Classes",
        "-" * 72,
    ]
    for i, r in enumerate(stats["top10"], start=1):
        lines.append(
            f"{i:2d}. {r['class_name']} (id={r['class_id']}): "
            f"{r['detections']} detections ({r['percentage']:.2f}%), "
            f"avg conf={r['avg_confidence']:.4f}"
        )
    if not stats["top10"]:
        lines.append("No detections found.")

    lines.extend(
        [
            "",
            "6. Important Observations",
            "-" * 72,
            (
                f"- The model produced detections on "
                f"{s['images_with_detections']}/{s['processed_images']} images."
            ),
            (
                f"- Across all predictions, mean confidence was {c['mean']:.4f} "
                f"(median {c['median']:.4f})."
            ),
            (
                f"- The most frequent class was "
                f"{stats['top10'][0]['class_name']} "
                f"({stats['top10'][0]['detections']} detections)."
                if stats["top10"]
                else "- No class dominated because no detections were found."
            ),
            f"- {len(stats['zero'])} COCO classes had zero detections in this subset.",
            (
                f"- {len(malformed)} malformed prediction lines were skipped during parsing."
                if malformed
                else "- No malformed prediction lines were encountered."
            ),
            "",
            "7. Limitations of the Experiment",
            "-" * 72,
            "- Evaluation uses a 500-image subset of COCO val2017, not the full 5,000-image set.",
            "- Statistics are computed from prediction label files and do not replace COCO mAP evaluation.",
            "- Confidence thresholds used during prediction affect detection counts and averages.",
            "- Class imbalance in the subset may bias top-class rankings relative to the full validation set.",
            "- This report summarizes detection statistics only; Precision/Recall/mAP should be taken from the official validator when available.",
            "",
            "8. Output Files",
            "-" * 72,
            f"Directory: {output_dir}",
        ]
    )
    for name in REQUIRED_OUTPUTS:
        lines.append(f"- {name}")
    lines.append("")
    (output_dir / "summary_report.txt").write_text("\n".join(lines), encoding="utf-8")


def make_charts(stats: dict, detections: list[dict], output_dir: Path) -> None:
    """Generate publication-ready charts."""
    class_rows = stats["class_rows"]
    names = [r["class_name"] for r in class_rows]
    counts = [r["detections"] for r in class_rows]

    # Bar chart: detections by class
    fig, ax = plt.subplots(figsize=(16, 6), dpi=150)
    ax.bar(range(80), counts, color="#2F6DB3")
    ax.set_title("Hyper-YOLO Detected Objects by COCO Class (500-image subset)")
    ax.set_xlabel("COCO class")
    ax.set_ylabel("Number of detections")
    ax.set_xticks(range(80))
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "detections_by_class.png")
    plt.close(fig)

    # Confidence histogram
    confidences = [d["confidence"] for d in detections]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    if confidences:
        ax.hist(confidences, bins=40, color="#3A8F6E", edgecolor="white")
        mean_c = stats["confidence"]["mean"]
        median_c = stats["confidence"]["median"]
        ax.axvline(mean_c, color="#C0392B", linestyle="--", linewidth=1.5, label=f"Mean={mean_c:.3f}")
        ax.axvline(median_c, color="#8E44AD", linestyle="-.", linewidth=1.5, label=f"Median={median_c:.3f}")
        ax.legend()
    ax.set_title("Distribution of Detection Confidence Scores")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Frequency")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "confidence_histogram.png")
    plt.close(fig)

    # Top-10 pie chart
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    top = stats["top10"]
    if top:
        labels = [f"{r['class_name']} ({r['detections']})" for r in top]
        sizes = [r["detections"] for r in top]
        ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90, textprops={"fontsize": 8})
        ax.set_title("Top 10 Most Detected Classes")
    else:
        ax.text(0.5, 0.5, "No detections", ha="center", va="center")
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_dir / "top10_classes_pie.png")
    plt.close(fig)


def print_terminal_summary(stats: dict, output_dir: Path) -> None:
    """Print the clean research summary block."""
    s = stats["summary"]
    print("=" * 41)
    print("Hyper-YOLO Detection Summary")
    print("=" * 41)
    print()
    print(f"Processed Images: {s['processed_images']}")
    print(f"Images with detections: {s['images_with_detections']}")
    print(f"Images without detections: {s['images_without_detections']}")
    print()
    print(f"Total detected objects: {s['total_detected_objects']}")
    print()
    print(f"Average detections/image: {s['average_detections_per_image']:.4f}")
    print()
    print(f"Average confidence: {s['mean_confidence']:.6f}")
    print()
    print(f"Highest confidence: {s['max_confidence']:.6f}")
    print()
    print(f"Lowest confidence: {s['min_confidence']:.6f}")
    print()
    print("Top 10 detected classes:")
    for i, r in enumerate(stats["top10"], start=1):
        print(f"  {i:2d}. {r['class_name']}: {r['detections']} ({r['percentage']:.2f}%)")
    print()
    print("Results saved to:")
    print()
    print(f"{output_dir}")
    print()
    print("=" * 41)


def verify_outputs(output_dir: Path) -> None:
    """Ensure every required output file exists and is non-empty."""
    missing = []
    empty = []
    for name in REQUIRED_OUTPUTS:
        path = output_dir / name
        if not path.is_file():
            missing.append(name)
        elif path.stat().st_size == 0:
            empty.append(name)
    if missing or empty:
        raise RuntimeError(
            "Output verification failed. "
            f"Missing={missing or 'none'}; Empty={empty or 'none'}"
        )


def main() -> int:
    if not LABELS_DIR.is_dir():
        print(f"ERROR: labels directory not found: {LABELS_DIR}", file=sys.stderr)
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    detections, malformed, processed, images_with, images_without = parse_labels(LABELS_DIR)
    if malformed:
        print(f"WARNING: skipped {len(malformed)} malformed line(s):")
        for msg in malformed[:20]:
            print(f"  - {msg}")
        if len(malformed) > 20:
            print(f"  ... and {len(malformed) - 20} more")

    stats = compute_stats(detections, processed, images_with, images_without)
    write_csv_outputs(stats, OUTPUT_DIR)
    write_json(stats, OUTPUT_DIR)
    write_report(stats, malformed, OUTPUT_DIR)
    make_charts(stats, detections, OUTPUT_DIR)
    verify_outputs(OUTPUT_DIR)
    print_terminal_summary(stats, OUTPUT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
