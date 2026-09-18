#!/usr/bin/env python3
"""Run clean Hyper-YOLO baseline on selected COCO images and save verification outputs."""

from __future__ import annotations

import csv
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
HYPER_YOLO = EXPERIMENT_ROOT.parent / "Hyper-YOLO"
CLEAN_IMAGES_DIR = EXPERIMENT_ROOT / "clean_images"
RESULTS_DIR = EXPERIMENT_ROOT / "results"
SELECTED_CSV = RESULTS_DIR / "selected_images.csv"
PREDICTIONS_DIR = RESULTS_DIR / "clean_predictions"
IMAGES_DIR = PREDICTIONS_DIR / "images"
LABELS_DIR = PREDICTIONS_DIR / "labels"
VERIFICATION_DIR = PREDICTIONS_DIR / "verification"
BASELINE_CSV = RESULTS_DIR / "clean_baseline.csv"
SUMMARY_TXT = RESULTS_DIR / "clean_baseline_summary.txt"
OVERVIEW_IMAGE = PREDICTIONS_DIR / "verification_overview.jpg"

MODEL_PATH = HYPER_YOLO / "weights" / "hyper-yolon.pt"

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.7
IMAGE_SIZE = 640
DEVICE = "cpu"

COCO_TO_TARGET = {
    1: "person",
    3: "car",
}

BASELINE_COLUMNS = [
    "image_id",
    "filename",
    "target_class",
    "class_id",
    "gt_bbox_x",
    "gt_bbox_y",
    "gt_bbox_width",
    "gt_bbox_height",
    "detected",
    "predicted_class",
    "confidence",
    "pred_bbox_x",
    "pred_bbox_y",
    "pred_bbox_width",
    "pred_bbox_height",
    "iou",
    "total_detections",
    "verification_image",
]


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox_x: float
    bbox_y: float
    bbox_width: float
    bbox_height: float


@dataclass
class BaselineRecord:
    image_id: str
    filename: str
    target_class: str
    class_id: int
    gt_bbox_x: float
    gt_bbox_y: float
    gt_bbox_width: float
    gt_bbox_height: float
    detected: str
    predicted_class: str
    confidence: float
    pred_bbox_x: float
    pred_bbox_y: float
    pred_bbox_width: float
    pred_bbox_height: float
    iou: float
    total_detections: int
    verification_image: str


def ensure_dependencies() -> None:
    if not HYPER_YOLO.is_dir():
        print(f"ERROR: Hyper-YOLO not found: {HYPER_YOLO}", file=sys.stderr)
        sys.exit(1)
    if not MODEL_PATH.is_file():
        print(f"ERROR: Model weights not found: {MODEL_PATH}", file=sys.stderr)
        sys.exit(1)
    if not SELECTED_CSV.is_file():
        print(f"ERROR: selected_images.csv not found: {SELECTED_CSV}", file=sys.stderr)
        sys.exit(1)
    if not CLEAN_IMAGES_DIR.is_dir():
        print(f"ERROR: clean_images directory not found: {CLEAN_IMAGES_DIR}", file=sys.stderr)
        sys.exit(1)


def load_yolo():
    sys.path.insert(0, str(HYPER_YOLO))
    from ultralytics import YOLO

    return YOLO(str(MODEL_PATH))


def run_inference(model) -> None:
    raw_dir = PREDICTIONS_DIR / "_yolo_raw"
    if raw_dir.exists():
        shutil.rmtree(raw_dir)

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    VERIFICATION_DIR.mkdir(parents=True, exist_ok=True)

    model.predict(
        source=str(CLEAN_IMAGES_DIR),
        imgsz=IMAGE_SIZE,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        device=DEVICE,
        half=False,
        save=True,
        save_txt=True,
        save_conf=True,
        show_boxes=True,
        show_labels=True,
        show_conf=True,
        project=str(PREDICTIONS_DIR),
        name="_yolo_raw",
        exist_ok=True,
    )

    raw_labels = raw_dir / "labels"
    for image_path in raw_dir.glob("*.jpg"):
        shutil.move(str(image_path), str(IMAGES_DIR / image_path.name))
    if raw_labels.is_dir():
        for label_path in raw_labels.glob("*.txt"):
            shutil.move(str(label_path), str(LABELS_DIR / label_path.name))

    shutil.rmtree(raw_dir, ignore_errors=True)


def read_selected_rows() -> list[dict[str, str]]:
    with SELECTED_CSV.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def yolo_index_for_target(model, target_class: str) -> int | None:
    names = model.names
    for index, name in names.items():
        if name == target_class:
            return int(index)
    return None


def parse_label_file(
    label_path: Path,
    image_width: int,
    image_height: int,
    model,
) -> list[Detection]:
    if not label_path.is_file():
        return []

    detections: list[Detection] = []
    names = model.names

    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        class_id = int(float(parts[0]))
        xc = float(parts[1])
        yc = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])
        confidence = float(parts[5]) if len(parts) > 5 else 1.0

        bbox_width = w * image_width
        bbox_height = h * image_height
        bbox_x = xc * image_width - bbox_width / 2.0
        bbox_y = yc * image_height - bbox_height / 2.0

        detections.append(
            Detection(
                class_id=class_id,
                class_name=names.get(class_id, str(class_id)),
                confidence=confidence,
                bbox_x=bbox_x,
                bbox_y=bbox_y,
                bbox_width=bbox_width,
                bbox_height=bbox_height,
            )
        )

    return detections


def box_iou_xywh(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0

    union = aw * ah + bw * bh - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def match_detection(
    detections: list[Detection],
    target_class: str,
    gt_bbox: tuple[float, float, float, float],
) -> Detection | None:
    candidates = [det for det in detections if det.class_name == target_class]
    if not candidates:
        return None

    return max(
        candidates,
        key=lambda det: box_iou_xywh(
            gt_bbox,
            (det.bbox_x, det.bbox_y, det.bbox_width, det.bbox_height),
        ),
    )


def draw_verification_image(
    source_image: Path,
    output_path: Path,
    target_class: str,
    gt_bbox: tuple[float, float, float, float],
    matched: Detection | None,
    iou: float,
) -> None:
    image = cv2.imread(str(source_image))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {source_image}")

    gx, gy, gw, gh = [int(round(v)) for v in gt_bbox]
    cv2.rectangle(image, (gx, gy), (gx + gw, gy + gh), (0, 255, 0), 2)

    if matched is not None:
        px, py, pw, ph = [
            int(round(v))
            for v in (matched.bbox_x, matched.bbox_y, matched.bbox_width, matched.bbox_height)
        ]
        cv2.rectangle(image, (px, py), (px + pw, py + ph), (0, 0, 255), 2)
        pred_text = f"Pred: {matched.class_name} {matched.confidence:.2f}"
        iou_text = f"IoU: {iou:.2f}"
    else:
        pred_text = "TARGET MISSED"
        iou_text = "IoU: 0.00"

    gt_text = f"GT: {target_class}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    line_height = 24
    x0, y0 = 10, 24

    for index, text in enumerate([gt_text, pred_text, iou_text]):
        y = y0 + index * line_height
        cv2.putText(
            image,
            text,
            (x0, y),
            font,
            scale,
            (255, 255, 255),
            thickness + 2,
            cv2.LINE_AA,
        )
        cv2.putText(image, text, (x0, y), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)

    legend_y = image.shape[0] - 50
    cv2.putText(
        image,
        "GREEN = COCO Ground Truth",
        (10, legend_y),
        font,
        0.55,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "RED = Hyper-YOLO Prediction",
        (10, legend_y + 22),
        font,
        0.55,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.imwrite(str(output_path), image)


def build_records(model) -> list[BaselineRecord]:
    records: list[BaselineRecord] = []

    for row in read_selected_rows():
        filename = row["filename"]
        target_class = row["target_class"]
        class_id = int(row["class_id"])
        image_width = int(float(row["image_width"]))
        image_height = int(float(row["image_height"]))
        gt_bbox = (
            float(row["bbox_x"]),
            float(row["bbox_y"]),
            float(row["bbox_width"]),
            float(row["bbox_height"]),
        )

        label_path = LABELS_DIR / f"{Path(filename).stem}.txt"
        detections = parse_label_file(label_path, image_width, image_height, model)
        matched = match_detection(detections, target_class, gt_bbox)

        if matched is None:
            detected = "No"
            predicted_class = ""
            confidence = 0.0
            pred_bbox = (0.0, 0.0, 0.0, 0.0)
            iou = 0.0
        else:
            detected = "Yes"
            predicted_class = matched.class_name
            confidence = matched.confidence
            pred_bbox = (
                matched.bbox_x,
                matched.bbox_y,
                matched.bbox_width,
                matched.bbox_height,
            )
            iou = box_iou_xywh(gt_bbox, pred_bbox)

        verification_name = f"verify_{Path(filename).stem}.jpg"
        verification_path = VERIFICATION_DIR / verification_name
        draw_verification_image(
            CLEAN_IMAGES_DIR / filename,
            verification_path,
            target_class,
            gt_bbox,
            matched,
            iou,
        )

        records.append(
            BaselineRecord(
                image_id=row["image_id"],
                filename=filename,
                target_class=target_class,
                class_id=class_id,
                gt_bbox_x=gt_bbox[0],
                gt_bbox_y=gt_bbox[1],
                gt_bbox_width=gt_bbox[2],
                gt_bbox_height=gt_bbox[3],
                detected=detected,
                predicted_class=predicted_class,
                confidence=confidence,
                pred_bbox_x=pred_bbox[0],
                pred_bbox_y=pred_bbox[1],
                pred_bbox_width=pred_bbox[2],
                pred_bbox_height=pred_bbox[3],
                iou=iou,
                total_detections=len(detections),
                verification_image=str(verification_path.relative_to(EXPERIMENT_ROOT)).replace("\\", "/"),
            )
        )

    return records


def write_baseline_csv(records: list[BaselineRecord]) -> None:
    with BASELINE_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=BASELINE_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "image_id": record.image_id,
                    "filename": record.filename,
                    "target_class": record.target_class,
                    "class_id": record.class_id,
                    "gt_bbox_x": record.gt_bbox_x,
                    "gt_bbox_y": record.gt_bbox_y,
                    "gt_bbox_width": record.gt_bbox_width,
                    "gt_bbox_height": record.gt_bbox_height,
                    "detected": record.detected,
                    "predicted_class": record.predicted_class,
                    "confidence": record.confidence,
                    "pred_bbox_x": record.pred_bbox_x,
                    "pred_bbox_y": record.pred_bbox_y,
                    "pred_bbox_width": record.pred_bbox_width,
                    "pred_bbox_height": record.pred_bbox_height,
                    "iou": record.iou,
                    "total_detections": record.total_detections,
                    "verification_image": record.verification_image,
                }
            )


def write_summary(records: list[BaselineRecord]) -> None:
    person_count = sum(1 for record in records if record.target_class == "person")
    car_count = sum(1 for record in records if record.target_class == "car")
    detected_records = [record for record in records if record.detected == "Yes"]
    missed_records = [record for record in records if record.detected == "No"]

    mean_conf = (
        sum(record.confidence for record in detected_records) / len(detected_records)
        if detected_records
        else 0.0
    )
    mean_iou = (
        sum(record.iou for record in detected_records) / len(detected_records)
        if detected_records
        else 0.0
    )
    avg_total_detections = (
        sum(record.total_detections for record in records) / len(records)
        if records
        else 0.0
    )

    lines = [
        "Clean Hyper-YOLO Baseline Summary",
        "=" * 72,
        f"total images: {len(records)}",
        f"person images: {person_count}",
        f"car images: {car_count}",
        f"successfully detected target objects: {len(detected_records)}",
        f"missed target objects: {len(missed_records)}",
        f"mean target confidence: {mean_conf:.4f}",
        f"mean IoU: {mean_iou:.4f}",
        f"average total detections per image: {avg_total_detections:.4f}",
        f"model path: {MODEL_PATH}",
        f"confidence threshold: {CONF_THRESHOLD}",
        f"IoU threshold: {IOU_THRESHOLD}",
        f"image size: {IMAGE_SIZE}",
        f"device: {DEVICE}",
        f"verification image folder: {VERIFICATION_DIR}",
    ]
    SUMMARY_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def create_overview(records: list[BaselineRecord]) -> None:
    cols = 5
    rows = math.ceil(len(records) / cols)
    thumb_w, thumb_h = 320, 240
    label_h = 28
    margin = 10

    canvas_w = cols * thumb_w + (cols + 1) * margin
    canvas_h = rows * (thumb_h + label_h) + (rows + 1) * margin
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX

    for index, record in enumerate(records):
        row = index // cols
        col = index % cols
        x = margin + col * (thumb_w + margin)
        y = margin + row * (thumb_h + label_h + margin)

        image_path = EXPERIMENT_ROOT / record.verification_image
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        thumb = cv2.resize(image, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        canvas[y : y + thumb_h, x : x + thumb_w] = thumb

        text_y = y + thumb_h + 20
        cv2.putText(
            canvas,
            record.filename,
            (x, text_y),
            font,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(OVERVIEW_IMAGE), canvas)


def verify_outputs(records: list[BaselineRecord]) -> None:
    clean_count = len(list(CLEAN_IMAGES_DIR.glob("*.jpg")))
    if clean_count != 20:
        raise RuntimeError(f"Expected 20 clean images, found {clean_count}")
    if len(records) != 20:
        raise RuntimeError(f"Expected 20 baseline records, found {len(records)}")
    if len(list(VERIFICATION_DIR.glob("*.jpg"))) != 20:
        raise RuntimeError("Expected 20 verification images")
    if not OVERVIEW_IMAGE.is_file():
        raise RuntimeError(f"Missing overview image: {OVERVIEW_IMAGE}")
    if not BASELINE_CSV.is_file():
        raise RuntimeError(f"Missing baseline CSV: {BASELINE_CSV}")

    for record in records:
        if not (CLEAN_IMAGES_DIR / record.filename).is_file():
            raise RuntimeError(f"Missing clean image: {record.filename}")
        if not (EXPERIMENT_ROOT / record.verification_image).is_file():
            raise RuntimeError(f"Missing verification image for {record.filename}")


def print_terminal_summary(records: list[BaselineRecord]) -> None:
    detected = sum(1 for record in records if record.detected == "Yes")
    missed = len(records) - detected

    print("Clean Hyper-YOLO baseline complete")
    print(f"- model path: {MODEL_PATH}")
    print(f"- confidence threshold: {CONF_THRESHOLD}")
    print(f"- IoU threshold: {IOU_THRESHOLD}")
    print(f"- image size: {IMAGE_SIZE}")
    print(f"- device: {DEVICE}")
    print(f"- images processed: {len(records)}")
    print(f"- targets detected: {detected}")
    print(f"- targets missed: {missed}")
    print(f"- baseline CSV: {BASELINE_CSV}")
    print(f"- summary file: {SUMMARY_TXT}")
    print(f"- verification folder: {VERIFICATION_DIR}")
    print(f"- overview image: {OVERVIEW_IMAGE}")
    print("Sample verification images:")
    for record in records[:5]:
        print(f"  - {EXPERIMENT_ROOT / record.verification_image}")


def main() -> None:
    ensure_dependencies()
    model = load_yolo()

    for class_id, target_class in COCO_TO_TARGET.items():
        if yolo_index_for_target(model, target_class) is None:
            print(
                f"ERROR: Could not map COCO class_id {class_id} ({target_class}) "
                "to YOLO class names.",
                file=sys.stderr,
            )
            sys.exit(1)

    run_inference(model)
    records = build_records(model)
    write_baseline_csv(records)
    write_summary(records)
    create_overview(records)
    verify_outputs(records)
    print_terminal_summary(records)


if __name__ == "__main__":
    main()
