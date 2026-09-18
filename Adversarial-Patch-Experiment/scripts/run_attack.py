#!/usr/bin/env python3
"""Run a diffusion-generated candidate patch attack experiment."""

from __future__ import annotations

import csv
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from attack_utils import get_next_attack_id  # noqa: E402

EXPERIMENT_ROOT = SCRIPT_DIR.parent
HYPER_YOLO = EXPERIMENT_ROOT.parent / "Hyper-YOLO"
CLEAN_IMAGES_DIR = EXPERIMENT_ROOT / "clean_images"
DIFFUSION_PATCHES_DIR = EXPERIMENT_ROOT / "diffusion_patches"
RESULTS_DIR = EXPERIMENT_ROOT / "results"
ATTACKS_DIR = HYPER_YOLO / "attacks"
SELECTED_CSV = RESULTS_DIR / "selected_images.csv"
CLEAN_BASELINE_CSV = RESULTS_DIR / "clean_baseline.csv"
MODEL_PATH = HYPER_YOLO / "weights" / "hyper-yolon.pt"

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.7
IMAGE_SIZE = 640
DEVICE = "cpu"
PATCH_SCALE = 0.30
EXPECTED_IMAGE_COUNT = 20
ASR_ELIGIBLE_COUNT = 19

DEFAULT_PATCH_NAME = "diffusion_patch_01.png"

ATTACK_RESULTS_COLUMNS = [
    "image_id",
    "filename",
    "target_class",
    "class_id",
    "clean_detected",
    "clean_confidence",
    "clean_iou",
    "clean_total_detections",
    "attacked_detected",
    "attacked_confidence",
    "attacked_iou",
    "attacked_total_detections",
    "confidence_drop",
    "iou_drop",
    "detection_count_change",
    "patch_filename",
    "patch_size_pixels",
    "patch_scale",
    "patch_center_x",
    "patch_center_y",
    "eligible_for_asr",
    "target_missed_after_attack",
    "attack_success",
]


@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox_x: float
    bbox_y: float
    bbox_width: float
    bbox_height: float


@dataclass
class AttackRecord:
    image_id: str
    filename: str
    target_class: str
    class_id: int
    clean_detected: str
    clean_confidence: float
    clean_iou: float
    clean_total_detections: int
    attacked_detected: str
    attacked_confidence: float
    attacked_iou: float
    attacked_total_detections: int
    confidence_drop: float
    iou_drop: float
    detection_count_change: int
    patch_filename: str
    patch_size_pixels: int
    patch_scale: float
    patch_center_x: float
    patch_center_y: float
    eligible_for_asr: str
    target_missed_after_attack: str
    attack_success: str
    verification_image: str


def validate_inputs(patch_path: Path) -> None:
    errors: list[str] = []

    clean_images = sorted(CLEAN_IMAGES_DIR.glob("*.jpg"))
    if len(clean_images) != EXPECTED_IMAGE_COUNT:
        errors.append(
            f"Expected {EXPECTED_IMAGE_COUNT} clean images, found {len(clean_images)} in {CLEAN_IMAGES_DIR}"
        )

    if not SELECTED_CSV.is_file():
        errors.append(f"Missing selected_images.csv: {SELECTED_CSV}")
    elif len(list(csv.DictReader(SELECTED_CSV.open(encoding="utf-8")))) != EXPECTED_IMAGE_COUNT:
        errors.append(f"selected_images.csv must contain {EXPECTED_IMAGE_COUNT} rows")

    if not CLEAN_BASELINE_CSV.is_file():
        errors.append(f"Missing clean_baseline.csv: {CLEAN_BASELINE_CSV}")
    elif len(list(csv.DictReader(CLEAN_BASELINE_CSV.open(encoding="utf-8")))) != EXPECTED_IMAGE_COUNT:
        errors.append(f"clean_baseline.csv must contain {EXPECTED_IMAGE_COUNT} rows")

    if not patch_path.is_file():
        errors.append(f"Missing candidate patch: {patch_path}")

    if not MODEL_PATH.is_file():
        errors.append(f"Missing Hyper-YOLO weights: {MODEL_PATH}")

    if not HYPER_YOLO.is_dir():
        errors.append(f"Missing Hyper-YOLO project: {HYPER_YOLO}")

    if errors:
        print("INPUT VALIDATION FAILED", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        sys.exit(1)


def load_yolo():
    sys.path.insert(0, str(HYPER_YOLO))
    from ultralytics import YOLO

    return YOLO(str(MODEL_PATH))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def create_attack_folder(attack_id: str) -> Path:
    attack_dir = ATTACKS_DIR / attack_id
    if attack_dir.exists():
        raise FileExistsError(f"Attack folder already exists: {attack_dir}")

    ATTACKS_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("patch", "patched_images", "prediction_images", "labels", "verification"):
        (attack_dir / name).mkdir(parents=True, exist_ok=False)
    return attack_dir


def copy_patch(source: Path, attack_dir: Path, patch_name: str) -> Path:
    destination = attack_dir / "patch" / patch_name
    shutil.copy2(source, destination)
    return destination


def compute_patch_geometry(
    gt_bbox: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> tuple[int, float, float, float, float]:
    gt_x, gt_y, gt_w, gt_h = gt_bbox
    patch_size = max(1, int(round(PATCH_SCALE * min(gt_w, gt_h))))
    center_x = gt_x + gt_w / 2.0
    center_y = gt_y + gt_h / 2.0
    patch_x = center_x - patch_size / 2.0
    patch_y = center_y - patch_size / 2.0
    patch_x = max(0.0, min(patch_x, image_width - patch_size))
    patch_y = max(0.0, min(patch_y, image_height - patch_size))
    return patch_size, center_x, center_y, patch_x, patch_y


def apply_patch_to_image(
    image_bgr: np.ndarray,
    patch_bgr: np.ndarray,
    patch_x: float,
    patch_y: float,
    patch_size: int,
) -> np.ndarray:
    result = image_bgr.copy()
    patch_resized = cv2.resize(patch_bgr, (patch_size, patch_size), interpolation=cv2.INTER_AREA)
    x1 = int(round(patch_x))
    y1 = int(round(patch_y))
    x2 = x1 + patch_size
    y2 = y1 + patch_size
    h, w = result.shape[:2]
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return result
    crop_w = x2 - x1
    crop_h = y2 - y1
    result[y1:y2, x1:x2] = patch_resized[:crop_h, :crop_w]
    return result


def apply_patches(
    attack_dir: Path,
    selected_rows: list[dict[str, str]],
    patch_bgr: np.ndarray,
    patch_name: str,
) -> dict[str, dict[str, float | int]]:
    patched_dir = attack_dir / "patched_images"
    geometry: dict[str, dict[str, float | int]] = {}

    for row in selected_rows:
        filename = row["filename"]
        source_path = CLEAN_IMAGES_DIR / filename
        image = cv2.imread(str(source_path))
        if image is None:
            raise FileNotFoundError(f"Could not read clean image: {source_path}")

        image_height, image_width = image.shape[:2]
        gt_bbox = (
            float(row["bbox_x"]),
            float(row["bbox_y"]),
            float(row["bbox_width"]),
            float(row["bbox_height"]),
        )
        patch_size, center_x, center_y, patch_x, patch_y = compute_patch_geometry(
            gt_bbox, image_width, image_height
        )
        patched = apply_patch_to_image(image, patch_bgr, patch_x, patch_y, patch_size)
        cv2.imwrite(str(patched_dir / filename), patched)
        geometry[filename] = {
            "patch_size_pixels": patch_size,
            "patch_center_x": center_x,
            "patch_center_y": center_y,
            "patch_x": patch_x,
            "patch_y": patch_y,
        }

    return geometry


def run_yolo_inference(model, attack_dir: Path) -> None:
    raw_dir = attack_dir / "_yolo_raw"
    if raw_dir.exists():
        shutil.rmtree(raw_dir)

    pred_images = attack_dir / "prediction_images"
    labels_dir = attack_dir / "labels"
    pred_images.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    model.predict(
        source=str(attack_dir / "patched_images"),
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
        project=str(attack_dir),
        name="_yolo_raw",
        exist_ok=True,
    )

    raw_labels = raw_dir / "labels"
    for image_path in raw_dir.glob("*.jpg"):
        shutil.move(str(image_path), str(pred_images / image_path.name))
    if raw_labels.is_dir():
        for label_path in raw_labels.glob("*.txt"):
            shutil.move(str(label_path), str(labels_dir / label_path.name))
    shutil.rmtree(raw_dir, ignore_errors=True)


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
        xc, yc, w, h = map(float, parts[1:5])
        confidence = float(parts[5]) if len(parts) > 5 else 1.0
        bbox_width = w * image_width
        bbox_height = h * image_height
        bbox_x = xc * image_width - bbox_width / 2.0
        bbox_y = yc * image_height - bbox_height / 2.0
        detections.append(
            Detection(
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
    return inter_area / union if union > 0.0 else 0.0


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


def draw_verification(
    patched_image_path: Path,
    output_path: Path,
    filename: str,
    target_class: str,
    gt_bbox: tuple[float, float, float, float],
    patch_rect: tuple[float, float, float, float],
    record: AttackRecord,
) -> None:
    image = cv2.imread(str(patched_image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read patched image: {patched_image_path}")

    gx, gy, gw, gh = [int(round(v)) for v in gt_bbox]
    cv2.rectangle(image, (gx, gy), (gx + gw, gy + gh), (0, 255, 0), 2)

    px, py, pw, ph = [int(round(v)) for v in patch_rect]
    cv2.rectangle(image, (px, py), (px + pw, py + ph), (0, 255, 255), 2)

    lines = [
        filename,
        f"Target: {target_class}",
        f"Clean: {record.clean_detected} conf={record.clean_confidence:.3f} IoU={record.clean_iou:.3f}",
        f"Attacked: {record.attacked_detected} conf={record.attacked_confidence:.3f} IoU={record.attacked_iou:.3f}",
        f"Conf drop: {record.confidence_drop:.3f}",
        f"Attack success: {record.attack_success}",
    ]
    if record.target_missed_after_attack == "Yes":
        lines.append("TARGET MISSED AFTER ATTACK")
    if record.eligible_for_asr == "No":
        lines.append("NOT ELIGIBLE FOR ASR")
        lines.append("Clean target was already missed")

    font = cv2.FONT_HERSHEY_SIMPLEX
    y = 22
    for line in lines:
        cv2.putText(image, line, (8, y), font, 0.48, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(image, line, (8, y), font, 0.48, (0, 0, 0), 1, cv2.LINE_AA)
        y += 18

    cv2.imwrite(str(output_path), image)


def build_records(
    model,
    attack_dir: Path,
    selected_rows: list[dict[str, str]],
    baseline_by_filename: dict[str, dict[str, str]],
    geometry: dict[str, dict[str, float | int]],
    patch_name: str,
) -> list[AttackRecord]:
    records: list[AttackRecord] = []

    for row in selected_rows:
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
        baseline = baseline_by_filename[filename]
        clean_detected = baseline["detected"]
        clean_confidence = float(baseline["confidence"] or 0.0)
        clean_iou = float(baseline["iou"] or 0.0)
        clean_total = int(float(baseline["total_detections"]))

        label_path = attack_dir / "labels" / f"{Path(filename).stem}.txt"
        detections = parse_label_file(label_path, image_width, image_height, model)
        matched = match_detection(detections, target_class, gt_bbox)

        if matched is None:
            attacked_detected = "No"
            attacked_confidence = 0.0
            attacked_iou = 0.0
        else:
            attacked_detected = "Yes"
            attacked_confidence = matched.confidence
            attacked_iou = box_iou_xywh(
                gt_bbox,
                (matched.bbox_x, matched.bbox_y, matched.bbox_width, matched.bbox_height),
            )

        attacked_total = len(detections)
        confidence_drop = clean_confidence - attacked_confidence
        iou_drop = clean_iou - attacked_iou
        detection_count_change = attacked_total - clean_total

        eligible = "Yes" if clean_detected == "Yes" else "No"
        missed_after = "Yes" if attacked_detected == "No" else "No"
        success = "Yes" if eligible == "Yes" and missed_after == "Yes" else "No"

        geo = geometry[filename]
        patch_size = int(geo["patch_size_pixels"])
        patch_rect = (
            float(geo["patch_x"]),
            float(geo["patch_y"]),
            float(patch_size),
            float(patch_size),
        )

        verification_name = f"verify_{Path(filename).stem}.jpg"
        verification_path = attack_dir / "verification" / verification_name

        record = AttackRecord(
            image_id=row["image_id"],
            filename=filename,
            target_class=target_class,
            class_id=class_id,
            clean_detected=clean_detected,
            clean_confidence=clean_confidence,
            clean_iou=clean_iou,
            clean_total_detections=clean_total,
            attacked_detected=attacked_detected,
            attacked_confidence=attacked_confidence,
            attacked_iou=attacked_iou,
            attacked_total_detections=attacked_total,
            confidence_drop=confidence_drop,
            iou_drop=iou_drop,
            detection_count_change=detection_count_change,
            patch_filename=patch_name,
            patch_size_pixels=patch_size,
            patch_scale=PATCH_SCALE,
            patch_center_x=float(geo["patch_center_x"]),
            patch_center_y=float(geo["patch_center_y"]),
            eligible_for_asr=eligible,
            target_missed_after_attack=missed_after,
            attack_success=success,
            verification_image=str(verification_path.relative_to(EXPERIMENT_ROOT)).replace("\\", "/"),
        )
        draw_verification(
            attack_dir / "patched_images" / filename,
            verification_path,
            filename,
            target_class,
            gt_bbox,
            patch_rect,
            record,
        )
        records.append(record)

    return records


def write_attack_results(attack_dir: Path, records: list[AttackRecord]) -> None:
    path = attack_dir / "attack_results.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ATTACK_RESULTS_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "image_id": record.image_id,
                    "filename": record.filename,
                    "target_class": record.target_class,
                    "class_id": record.class_id,
                    "clean_detected": record.clean_detected,
                    "clean_confidence": record.clean_confidence,
                    "clean_iou": record.clean_iou,
                    "clean_total_detections": record.clean_total_detections,
                    "attacked_detected": record.attacked_detected,
                    "attacked_confidence": record.attacked_confidence,
                    "attacked_iou": record.attacked_iou,
                    "attacked_total_detections": record.attacked_total_detections,
                    "confidence_drop": record.confidence_drop,
                    "iou_drop": record.iou_drop,
                    "detection_count_change": record.detection_count_change,
                    "patch_filename": record.patch_filename,
                    "patch_size_pixels": record.patch_size_pixels,
                    "patch_scale": record.patch_scale,
                    "patch_center_x": record.patch_center_x,
                    "patch_center_y": record.patch_center_y,
                    "eligible_for_asr": record.eligible_for_asr,
                    "target_missed_after_attack": record.target_missed_after_attack,
                    "attack_success": record.attack_success,
                }
            )


def create_overview(attack_dir: Path, records: list[AttackRecord]) -> Path:
    overview_path = attack_dir / "attack_overview.jpg"
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
        image = cv2.imread(str(attack_dir / "verification" / f"verify_{Path(record.filename).stem}.jpg"))
        if image is None:
            continue
        thumb = cv2.resize(image, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        canvas[y : y + thumb_h, x : x + thumb_w] = thumb
        cv2.putText(canvas, record.filename, (x, y + thumb_h + 20), font, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.imwrite(str(overview_path), canvas)
    return overview_path


def write_attack_summary(attack_dir: Path, attack_id: str, patch_name: str, records: list[AttackRecord]) -> None:
    eligible = [r for r in records if r.eligible_for_asr == "Yes"]
    successes = [r for r in records if r.attack_success == "Yes"]
    clean_detected = sum(1 for r in records if r.clean_detected == "Yes")
    clean_missed = len(records) - clean_detected
    attacked_detected = sum(1 for r in records if r.attacked_detected == "Yes")
    new_misses = sum(1 for r in records if r.target_missed_after_attack == "Yes" and r.eligible_for_asr == "Yes")

    mean_clean_conf = sum(r.clean_confidence for r in eligible) / len(eligible) if eligible else 0.0
    mean_attacked_conf = sum(r.attacked_confidence for r in eligible) / len(eligible) if eligible else 0.0
    mean_conf_drop = sum(r.confidence_drop for r in eligible) / len(eligible) if eligible else 0.0
    mean_clean_iou = sum(r.clean_iou for r in eligible) / len(eligible) if eligible else 0.0
    mean_attacked_iou = sum(r.attacked_iou for r in eligible) / len(eligible) if eligible else 0.0
    avg_clean_det = sum(r.clean_total_detections for r in records) / len(records)
    avg_attacked_det = sum(r.attacked_total_detections for r in records) / len(records)
    asr = (len(successes) / ASR_ELIGIBLE_COUNT) * 100.0

    lines = [
        f"Attack ID: {attack_id}",
        "",
        "Attack type:",
        "Diffusion-generated candidate patch evaluation",
        "",
        f"Patch:",
        patch_name,
        "",
        "Images processed:",
        "20",
        "",
        "Patch scale:",
        "30% of shorter GT bounding-box side",
        "",
        "Patch position:",
        "Center of GT bounding box",
        "",
        "Rotation:",
        "0 degrees",
        "",
        "Brightness:",
        "unchanged",
        "",
        "Opacity:",
        "100%",
        "",
        f"Clean targets detected:",
        str(clean_detected),
        "",
        f"Clean targets already missed:",
        str(clean_missed),
        "",
        f"ASR eligible images:",
        str(len(eligible)),
        "",
        f"Attacked targets detected:",
        str(attacked_detected),
        "",
        f"New target misses caused after patch:",
        str(new_misses),
        "",
        f"Attack successes:",
        str(len(successes)),
        "",
        f"Attack Success Rate:",
        f"{len(successes)} / {ASR_ELIGIBLE_COUNT} = {asr:.2f}%",
        "",
        f"Mean clean confidence among eligible targets:",
        f"{mean_clean_conf:.4f}",
        "",
        f"Mean attacked confidence among eligible targets:",
        f"{mean_attacked_conf:.4f}",
        "",
        f"Mean confidence drop:",
        f"{mean_conf_drop:.4f}",
        "",
        f"Mean clean IoU:",
        f"{mean_clean_iou:.4f}",
        "",
        f"Mean attacked IoU:",
        f"{mean_attacked_iou:.4f}",
        "",
        f"Average clean detections/image:",
        f"{avg_clean_det:.4f}",
        "",
        f"Average attacked detections/image:",
        f"{avg_attacked_det:.4f}",
    ]
    (attack_dir / "attack_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify_outputs(attack_id: str, attack_dir: Path, records: list[AttackRecord]) -> None:
    if attack_id != "attack_01":
        raise RuntimeError(f"Unexpected attack id: {attack_id}")
    if (ATTACKS_DIR / "attack_02").exists():
        raise RuntimeError("attack_02 must not be created")
    if len(list((attack_dir / "patched_images").glob("*.jpg"))) != EXPECTED_IMAGE_COUNT:
        raise RuntimeError("Expected 20 patched images")
    if len(list((attack_dir / "prediction_images").glob("*.jpg"))) != EXPECTED_IMAGE_COUNT:
        raise RuntimeError("Expected 20 prediction images")
    if len(list((attack_dir / "verification").glob("*.jpg"))) != EXPECTED_IMAGE_COUNT:
        raise RuntimeError("Expected 20 verification images")
    if len(records) != EXPECTED_IMAGE_COUNT:
        raise RuntimeError("Expected 20 attack result rows")
    if not (attack_dir / "attack_summary.txt").is_file():
        raise RuntimeError("Missing attack_summary.txt")
    if not (attack_dir / "attack_overview.jpg").is_file():
        raise RuntimeError("Missing attack_overview.jpg")


def main() -> None:
    patch_name = DEFAULT_PATCH_NAME
    patch_path = DIFFUSION_PATCHES_DIR / patch_name
    validate_inputs(patch_path)

    attack_id = get_next_attack_id()
    print(f"Starting Attack 01 — Diffusion-generated candidate patch evaluation")
    print(f"- next attack folder: {attack_id}")

    attack_dir = create_attack_folder(attack_id)
    copy_patch(patch_path, attack_dir, patch_name)

    patch_bgr = cv2.imread(str(attack_dir / "patch" / patch_name), cv2.IMREAD_COLOR)
    if patch_bgr is None:
        print(f"ERROR: Could not read patch image: {patch_path}", file=sys.stderr)
        sys.exit(1)

    selected_rows = read_csv(SELECTED_CSV)
    baseline_rows = read_csv(CLEAN_BASELINE_CSV)
    baseline_by_filename = {row["filename"]: row for row in baseline_rows}

    geometry = apply_patches(attack_dir, selected_rows, patch_bgr, patch_name)

    model = load_yolo()
    run_yolo_inference(model, attack_dir)

    records = build_records(
        model, attack_dir, selected_rows, baseline_by_filename, geometry, patch_name
    )
    write_attack_results(attack_dir, records)
    overview_path = create_overview(attack_dir, records)
    write_attack_summary(attack_dir, attack_id, patch_name, records)
    verify_outputs(attack_id, attack_dir, records)

    successes = sum(1 for r in records if r.attack_success == "Yes")
    asr = successes / ASR_ELIGIBLE_COUNT * 100.0

    print("")
    print("Attack 01 — Diffusion-generated candidate patch evaluation complete")
    print(f"- attack folder: {attack_dir}")
    print(f"- patched images: {attack_dir / 'patched_images'}")
    print(f"- prediction images: {attack_dir / 'prediction_images'}")
    print(f"- labels: {attack_dir / 'labels'}")
    print(f"- verification: {attack_dir / 'verification'}")
    print(f"- attack_results.csv: {attack_dir / 'attack_results.csv'}")
    print(f"- attack_summary.txt: {attack_dir / 'attack_summary.txt'}")
    print(f"- attack_overview.jpg: {overview_path}")
    print(f"- attack successes: {successes} / {ASR_ELIGIBLE_COUNT}")
    print(f"- Attack Success Rate: {asr:.2f}%")
    print("")
    print("attack_01 dynamically created: YES")
    print("attack_02 created: NO")
    print("Clean images modified: NO")
    print("Clean baseline modified: NO")
    print("Original diffusion patch modified: NO")


if __name__ == "__main__":
    main()
