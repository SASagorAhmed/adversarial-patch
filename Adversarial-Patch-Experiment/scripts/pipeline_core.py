#!/usr/bin/env python3
"""Shared helpers for the automatic adversarial patch attack pipeline."""

from __future__ import annotations

import csv
import json
import math
import random
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from patch_config import PatchSpec, patch_spec_for_filename, patch_spec_for_number

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
HYPER_YOLO = EXPERIMENT_ROOT.parent / "Hyper-YOLO"
STABLE_DIFFUSION = EXPERIMENT_ROOT.parent / "Stable-Diffusion-Patch"
DIFFUSION_PATCHES_DIR = EXPERIMENT_ROOT / "diffusion_patches"
ATTACKS_DIR = HYPER_YOLO / "attacks"
MODEL_PATH = HYPER_YOLO / "weights" / "hyper-yolon.pt"
HYPER_YOLO_PYTHON = HYPER_YOLO / ".venv" / "Scripts" / "python.exe"
SD_PYTHON = STABLE_DIFFUSION / ".venv" / "Scripts" / "python.exe"
SD_MODEL_PATH = STABLE_DIFFUSION / "model"
PATCH_GENERATOR_SCRIPT = EXPERIMENT_ROOT / "scripts" / "generate_candidate_patch.py"
PATCH_WIDTH = 512
PATCH_HEIGHT = 512

ANNOTATION_CANDIDATES = [
    HYPER_YOLO / "annotations" / "instances_val2017.json",
    HYPER_YOLO / "coco" / "annotations" / "instances_val2017.json",
]

TARGET_CLASSES = {"person": 1, "car": 3}
ATTACK_DIR_PATTERN = re.compile(r"^attack_(\d+)$")

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.7
TARGET_MATCH_IOU_THRESHOLD = 0.50
IMAGE_SIZE = 640
DEVICE = "cpu"
PATCH_SCALE = 0.30
MIN_DIM_RATIO = 0.05
OVERVIEW_MAX_THUMBS = 20
BASE_NUM_IMAGES = 200
NUM_IMAGES_INCREMENT = 50


@dataclass(frozen=True)
class SelectedImage:
    image_id: int
    filename: str
    target_class: str
    class_id: int
    bbox_x: float
    bbox_y: float
    bbox_width: float
    bbox_height: float
    image_width: int
    image_height: int
    random_seed: int


@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox_x: float
    bbox_y: float
    bbox_width: float
    bbox_height: float


@dataclass
class AttackResultRow:
    attack_id: str
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
    attack_success: str
    clean_best_same_class_confidence: float
    clean_best_same_class_iou: float
    attacked_best_same_class_confidence: float
    attacked_best_same_class_iou: float


ATTACK_RESULTS_COLUMNS = [
    "attack_id",
    "image_id",
    "filename",
    "target_class",
    "class_id",
    "clean_detected",
    "clean_confidence",
    "clean_iou",
    "clean_total_detections",
    "clean_best_same_class_confidence",
    "clean_best_same_class_iou",
    "attacked_detected",
    "attacked_confidence",
    "attacked_iou",
    "attacked_total_detections",
    "attacked_best_same_class_confidence",
    "attacked_best_same_class_iou",
    "confidence_drop",
    "iou_drop",
    "detection_count_change",
    "patch_filename",
    "patch_size_pixels",
    "patch_scale",
    "patch_center_x",
    "patch_center_y",
    "eligible_for_asr",
    "attack_success",
]

SELECTED_COLUMNS = [
    "image_id",
    "filename",
    "target_class",
    "class_id",
    "bbox_x",
    "bbox_y",
    "bbox_width",
    "bbox_height",
    "image_width",
    "image_height",
    "random_seed",
]


def attack_id_from_number(number: int) -> str:
    return f"attack_{number:02d}"


def list_existing_attack_numbers() -> list[int]:
    if not ATTACKS_DIR.is_dir():
        return []
    numbers: list[int] = []
    for path in ATTACKS_DIR.iterdir():
        if path.is_dir() and (match := ATTACK_DIR_PATTERN.match(path.name)):
            numbers.append(int(match.group(1)))
    return sorted(numbers)


def get_next_attack_number() -> int:
    existing = list_existing_attack_numbers()
    return 1 if not existing else max(existing) + 1


def get_next_attack_id() -> str:
    return attack_id_from_number(get_next_attack_number())


def default_patch_name_for_attack_number(number: int) -> str:
    return f"diffusion_patch_{number:02d}.png"


def default_num_images_for_attack_number(number: int) -> int:
    """Return the default sample size for an attack number (200, 250, 300, ...)."""
    return BASE_NUM_IMAGES + ((number - 1) * NUM_IMAGES_INCREMENT)


def resolve_num_images(explicit_num_images: int | None, attack_number: int | None = None) -> int:
    """Resolve image count from explicit override or automatic schedule."""
    if explicit_num_images is not None:
        return explicit_num_images
    number = attack_number if attack_number is not None else get_next_attack_number()
    return default_num_images_for_attack_number(number)


def collect_previous_attack_filenames() -> set[str]:
    """Collect filenames used in completed previous attacks."""
    filenames: set[str] = set()
    if not ATTACKS_DIR.is_dir():
        return filenames

    for attack_dir in sorted(ATTACKS_DIR.iterdir()):
        if not attack_dir.is_dir():
            continue
        if (attack_dir / "FAILED.txt").is_file():
            continue
        selected_csv = attack_dir / "results" / "selected_images.csv"
        if not selected_csv.is_file():
            continue
        with selected_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                filename = row.get("filename")
                if filename:
                    filenames.add(filename)
    return filenames


def compute_previous_attack_overlap(
    selected: list[SelectedImage],
    previous_filenames: set[str],
) -> tuple[int, float]:
    """Return overlap count and percent with previous attack image sets."""
    if not selected:
        return 0, 0.0
    overlap_count = sum(1 for item in selected if item.filename in previous_filenames)
    overlap_percent = overlap_count / len(selected) * 100.0
    return overlap_count, overlap_percent


def resolve_patch_spec(patch_name: str) -> PatchSpec:
    """Return patch generation settings for a candidate patch filename."""
    return patch_spec_for_filename(patch_name)


def resolve_patch_spec_for_attack(patch_name: str | None) -> tuple[str, PatchSpec]:
    """Resolve patch filename and spec for the next or explicit attack."""
    if patch_name:
        return patch_name, resolve_patch_spec(patch_name)
    number = get_next_attack_number()
    spec = patch_spec_for_number(number)
    return spec.filename, spec


def validate_patch_file(path: Path) -> tuple[bool, str]:
    """Validate a candidate patch PNG (exists, format, 512x512, non-empty)."""
    if not path.is_file():
        return False, "file does not exist"
    if path.stat().st_size <= 0:
        return False, "file is empty"
    try:
        with Image.open(path) as image:
            if image.format != "PNG":
                return False, f"expected PNG, got {image.format}"
            width, height = image.size
            if width != PATCH_WIDTH or height != PATCH_HEIGHT:
                return False, f"expected {PATCH_WIDTH}x{PATCH_HEIGHT}, got {width}x{height}"
    except OSError as exc:
        return False, f"invalid image: {exc}"
    return True, "valid"


def generate_candidate_patch(spec: PatchSpec, output_path: Path | None = None) -> Path:
    """Generate a candidate patch via the Stable Diffusion subprocess."""
    if not SD_PYTHON.is_file():
        raise FileNotFoundError(f"Stable Diffusion Python not found: {SD_PYTHON}")
    if not SD_MODEL_PATH.is_dir():
        raise FileNotFoundError(f"Stable Diffusion model not found: {SD_MODEL_PATH}")
    if not PATCH_GENERATOR_SCRIPT.is_file():
        raise FileNotFoundError(f"Patch generator script not found: {PATCH_GENERATOR_SCRIPT}")

    target = output_path or (DIFFUSION_PATCHES_DIR / spec.filename)
    DIFFUSION_PATCHES_DIR.mkdir(parents=True, exist_ok=True)

    command = [
        str(SD_PYTHON),
        str(PATCH_GENERATOR_SCRIPT),
        "--output",
        str(target),
        "--seed",
        str(spec.seed),
        "--prompt",
        spec.prompt,
        "--model-path",
        str(SD_MODEL_PATH),
    ]
    print(f"Generating candidate patch: {spec.filename}")
    print(f"  seed: {spec.seed}")
    print(f"  output: {target}")
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.stdout:
        print(result.stdout.rstrip())
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown error"
        raise RuntimeError(f"Stable Diffusion patch generation failed:\n{detail}")

    valid, reason = validate_patch_file(target)
    if not valid:
        raise RuntimeError(f"Generated patch failed validation: {reason}")
    return target


def ensure_candidate_patch(patch_name: str, *, generate: bool = True) -> Path:
    """Ensure the candidate patch exists; optionally generate it if missing."""
    patch_path = DIFFUSION_PATCHES_DIR / patch_name
    valid, reason = validate_patch_file(patch_path)
    if valid:
        return patch_path
    if not generate:
        raise FileNotFoundError(f"Candidate patch not found or invalid ({reason}): {patch_path}")

    spec = resolve_patch_spec(patch_name)
    return generate_candidate_patch(spec, patch_path)


def check_python_environment(python_path: Path, label: str) -> tuple[bool, str]:
    if not python_path.is_file():
        return False, f"{label} Python not found: {python_path}"
    result = subprocess.run(
        [str(python_path), "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown error"
        return False, f"{label} Python failed: {detail}"
    version = (result.stdout or result.stderr).strip()
    return True, version


def check_sd_imports() -> tuple[bool, str]:
    """Verify diffusers/torch are importable in the SD environment."""
    code = "import torch; from diffusers import StableDiffusionPipeline; print('ok')"
    result = subprocess.run(
        [str(SD_PYTHON), "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "import failed"
        return False, detail
    return True, "diffusers and torch available"


def find_annotation_path() -> Path:
    for candidate in ANNOTATION_CANDIDATES:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "COCO annotations not found. Inspected: "
        + ", ".join(str(path) for path in ANNOTATION_CANDIDATES)
    )


def find_image_dir() -> Path:
    coco_root = HYPER_YOLO / "coco"
    if not coco_root.is_dir():
        raise FileNotFoundError(f"COCO root not found: {coco_root}")

    best_dir: Path | None = None
    best_count = 0
    for directory in coco_root.rglob("val2017"):
        if not directory.is_dir():
            continue
        count = sum(1 for _ in directory.glob("*.jpg"))
        if count > best_count:
            best_count = count
            best_dir = directory
    if best_dir is None or best_count == 0:
        raise FileNotFoundError("COCO val2017 image directory not found under Hyper-YOLO/coco")
    return best_dir


def bbox_is_large_enough(
    bbox_width: float,
    bbox_height: float,
    image_width: int,
    image_height: int,
) -> bool:
    return (
        bbox_width >= MIN_DIM_RATIO * image_width
        and bbox_height >= MIN_DIM_RATIO * image_height
    )


def build_class_candidates(coco: dict) -> dict[str, list[SelectedImage]]:
    images_by_id = {image["id"]: image for image in coco["images"]}
    pools: dict[str, list[SelectedImage]] = {name: [] for name in TARGET_CLASSES}
    per_image_best: dict[tuple[str, int], SelectedImage] = {}

    for annotation in coco["annotations"]:
        category_id = annotation["category_id"]
        target_class = next(
            (name for name, cid in TARGET_CLASSES.items() if cid == category_id),
            None,
        )
        if target_class is None:
            continue

        image = images_by_id.get(annotation["image_id"])
        if image is None:
            continue

        bbox_x, bbox_y, bbox_width, bbox_height = annotation["bbox"]
        image_width = image["width"]
        image_height = image["height"]
        if not bbox_is_large_enough(bbox_width, bbox_height, image_width, image_height):
            continue

        candidate = SelectedImage(
            image_id=annotation["image_id"],
            filename=image["file_name"],
            target_class=target_class,
            class_id=category_id,
            bbox_x=bbox_x,
            bbox_y=bbox_y,
            bbox_width=bbox_width,
            bbox_height=bbox_height,
            image_width=image_width,
            image_height=image_height,
            random_seed=0,
        )
        key = (target_class, annotation["image_id"])
        existing = per_image_best.get(key)
        if existing is None:
            per_image_best[key] = candidate
        else:
            old_area = existing.bbox_width * existing.bbox_height
            new_area = bbox_width * bbox_height
            if new_area > old_area:
                per_image_best[key] = candidate

    for (target_class, _), candidate in per_image_best.items():
        pools[target_class].append(candidate)
    return pools


def random_select_balanced(
    pools: dict[str, list[SelectedImage]],
    num_images: int,
    seed: int,
    exclude_filenames: set[str] | None = None,
) -> list[SelectedImage]:
    if num_images < 2:
        raise ValueError("--num-images must be at least 2")

    person_count = num_images // 2
    car_count = num_images - person_count
    rng = random.Random(seed)
    used_filenames: set[str] = set()
    selected: list[SelectedImage] = []
    previous_filenames = exclude_filenames or set()

    for target_class, count in (("person", person_count), ("car", car_count)):
        available = [item for item in pools[target_class] if item.filename not in used_filenames]
        preferred = [item for item in available if item.filename not in previous_filenames]
        pool = preferred if len(preferred) >= count else available
        if len(pool) < count:
            raise RuntimeError(
                f"Not enough eligible {target_class} images "
                f"(need {count}, found {len(pool)})"
            )
        picks = rng.sample(pool, count)
        for item in picks:
            selected.append(
                SelectedImage(
                    image_id=item.image_id,
                    filename=item.filename,
                    target_class=item.target_class,
                    class_id=item.class_id,
                    bbox_x=item.bbox_x,
                    bbox_y=item.bbox_y,
                    bbox_width=item.bbox_width,
                    bbox_height=item.bbox_height,
                    image_width=item.image_width,
                    image_height=item.image_height,
                    random_seed=seed,
                )
            )
            used_filenames.add(item.filename)

    selected.sort(key=lambda item: item.filename)
    return selected


def create_attack_structure(attack_id: str) -> Path:
    attack_dir = ATTACKS_DIR / attack_id
    if attack_dir.exists():
        raise FileExistsError(f"Attack folder already exists: {attack_dir}")

    paths = [
        attack_dir / "clean_images",
        attack_dir / "patch",
        attack_dir / "patched_images",
        attack_dir / "results" / "clean_predictions" / "images",
        attack_dir / "results" / "clean_predictions" / "labels",
        attack_dir / "results" / "attacked_predictions" / "images",
        attack_dir / "results" / "attacked_predictions" / "labels",
        attack_dir / "results" / "verification",
    ]
    ATTACKS_DIR.mkdir(parents=True, exist_ok=True)
    for path in paths:
        path.mkdir(parents=True, exist_ok=False)
    return attack_dir


def write_failed_marker(attack_dir: Path, stage: str, error: Exception) -> None:
    content = f"stage: {stage}\nerror: {error}\n"
    (attack_dir / "FAILED.txt").write_text(content, encoding="utf-8")


def copy_selected_images(
    selected: list[SelectedImage],
    source_dir: Path,
    destination_dir: Path,
) -> None:
    for item in selected:
        source = source_dir / item.filename
        if not source.is_file():
            raise FileNotFoundError(f"Missing COCO image: {source}")
        shutil.copy2(source, destination_dir / item.filename)


def write_selected_csv(path: Path, selected: list[SelectedImage]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SELECTED_COLUMNS)
        writer.writeheader()
        for item in selected:
            writer.writerow(
                {
                    "image_id": item.image_id,
                    "filename": item.filename,
                    "target_class": item.target_class,
                    "class_id": item.class_id,
                    "bbox_x": item.bbox_x,
                    "bbox_y": item.bbox_y,
                    "bbox_width": item.bbox_width,
                    "bbox_height": item.bbox_height,
                    "image_width": item.image_width,
                    "image_height": item.image_height,
                    "random_seed": item.random_seed,
                }
            )


def load_yolo():
    sys.path.insert(0, str(HYPER_YOLO))
    from ultralytics import YOLO

    return YOLO(str(MODEL_PATH))


def run_yolo_inference(
    model,
    source_dir: Path,
    images_dir: Path,
    labels_dir: Path,
    raw_parent: Path,
) -> None:
    raw_dir = raw_parent / "_yolo_raw"
    if raw_dir.exists():
        shutil.rmtree(raw_dir)

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    model.predict(
        source=str(source_dir),
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
        project=str(raw_parent),
        name="_yolo_raw",
        exist_ok=True,
    )

    raw_labels = raw_dir / "labels"
    for image_path in raw_dir.glob("*.jpg"):
        shutil.move(str(image_path), str(images_dir / image_path.name))
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
) -> tuple[Detection | None, float]:
    """Return the best same-class prediction and its IoU with the GT box."""
    candidates = [det for det in detections if det.class_name == target_class]
    if not candidates:
        return None, 0.0
    best = max(
        candidates,
        key=lambda det: box_iou_xywh(
            gt_bbox,
            (det.bbox_x, det.bbox_y, det.bbox_width, det.bbox_height),
        ),
    )
    best_iou = box_iou_xywh(
        gt_bbox,
        (best.bbox_x, best.bbox_y, best.bbox_width, best.bbox_height),
    )
    return best, best_iou


def evaluate_target_match(
    detections: list[Detection],
    target_class: str,
    gt_bbox: tuple[float, float, float, float],
) -> dict[str, float | int | str]:
    """Evaluate whether the selected GT target is validly detected."""
    best, best_iou = match_detection(detections, target_class, gt_bbox)
    best_confidence = best.confidence if best is not None else 0.0
    if best is None or best_iou < TARGET_MATCH_IOU_THRESHOLD:
        return {
            "detected": "No",
            "confidence": 0.0,
            "iou": 0.0,
            "best_same_class_confidence": best_confidence,
            "best_same_class_iou": best_iou,
            "total_detections": len(detections),
        }
    return {
        "detected": "Yes",
        "confidence": best_confidence,
        "iou": best_iou,
        "best_same_class_confidence": best_confidence,
        "best_same_class_iou": best_iou,
        "total_detections": len(detections),
    }


def evaluate_predictions(
    selected: list[SelectedImage],
    labels_dir: Path,
    model,
) -> dict[str, dict[str, float | int | str]]:
    results: dict[str, dict[str, float | int | str]] = {}
    for item in selected:
        gt_bbox = (item.bbox_x, item.bbox_y, item.bbox_width, item.bbox_height)
        label_path = labels_dir / f"{Path(item.filename).stem}.txt"
        detections = parse_label_file(label_path, item.image_width, item.image_height, model)
        results[item.filename] = evaluate_target_match(detections, item.target_class, gt_bbox)
    return results


def write_clean_baseline_csv(
    path: Path,
    selected: list[SelectedImage],
    clean_results: dict[str, dict[str, float | int | str]],
) -> None:
    columns = [
        "image_id",
        "filename",
        "target_class",
        "class_id",
        "gt_bbox_x",
        "gt_bbox_y",
        "gt_bbox_width",
        "gt_bbox_height",
        "detected",
        "confidence",
        "iou",
        "best_same_class_confidence",
        "best_same_class_iou",
        "total_detections",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for item in selected:
            clean = clean_results[item.filename]
            writer.writerow(
                {
                    "image_id": item.image_id,
                    "filename": item.filename,
                    "target_class": item.target_class,
                    "class_id": item.class_id,
                    "gt_bbox_x": item.bbox_x,
                    "gt_bbox_y": item.bbox_y,
                    "gt_bbox_width": item.bbox_width,
                    "gt_bbox_height": item.bbox_height,
                    "detected": clean["detected"],
                    "confidence": clean["confidence"],
                    "iou": clean["iou"],
                    "best_same_class_confidence": clean["best_same_class_confidence"],
                    "best_same_class_iou": clean["best_same_class_iou"],
                    "total_detections": clean["total_detections"],
                }
            )


def write_clean_baseline_summary(
    path: Path,
    selected: list[SelectedImage],
    clean_results: dict[str, dict[str, float | int | str]],
) -> None:
    detected = [item for item in selected if clean_results[item.filename]["detected"] == "Yes"]
    person_count = sum(1 for item in selected if item.target_class == "person")
    car_count = len(selected) - person_count
    mean_conf = (
        sum(float(clean_results[item.filename]["confidence"]) for item in detected) / len(detected)
        if detected
        else 0.0
    )
    mean_iou = (
        sum(float(clean_results[item.filename]["iou"]) for item in detected) / len(detected)
        if detected
        else 0.0
    )
    lines = [
        "Clean Hyper-YOLO Baseline Summary (attack-local)",
        "=" * 72,
        f"Target match IoU threshold: {TARGET_MATCH_IOU_THRESHOLD:.2f}",
        f"total images: {len(selected)}",
        f"person images: {person_count}",
        f"car images: {car_count}",
        f"valid clean target detections: {len(detected)}",
        f"clean target misses: {len(selected) - len(detected)}",
        f"mean target confidence: {mean_conf:.4f}",
        f"mean IoU: {mean_iou:.4f}",
        f"model path: {MODEL_PATH}",
        f"confidence threshold: {CONF_THRESHOLD}",
        f"IoU threshold: {IOU_THRESHOLD}",
        f"image size: {IMAGE_SIZE}",
        f"device: {DEVICE}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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


def apply_patches_to_selected(
    selected: list[SelectedImage],
    clean_images_dir: Path,
    patched_images_dir: Path,
    patch_bgr: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    geometry: dict[str, dict[str, float | int]] = {}
    for item in selected:
        image = cv2.imread(str(clean_images_dir / item.filename))
        if image is None:
            raise FileNotFoundError(f"Could not read clean image: {item.filename}")
        gt_bbox = (item.bbox_x, item.bbox_y, item.bbox_width, item.bbox_height)
        patch_size, center_x, center_y, patch_x, patch_y = compute_patch_geometry(
            gt_bbox, item.image_width, item.image_height
        )
        patched = apply_patch_to_image(image, patch_bgr, patch_x, patch_y, patch_size)
        cv2.imwrite(str(patched_images_dir / item.filename), patched)
        geometry[item.filename] = {
            "patch_size_pixels": patch_size,
            "patch_center_x": center_x,
            "patch_center_y": center_y,
            "patch_x": patch_x,
            "patch_y": patch_y,
        }
    return geometry


def build_attack_rows(
    attack_id: str,
    selected: list[SelectedImage],
    clean_results: dict[str, dict[str, float | int | str]],
    attacked_results: dict[str, dict[str, float | int | str]],
    geometry: dict[str, dict[str, float | int]],
    patch_filename: str,
) -> list[AttackResultRow]:
    rows: list[AttackResultRow] = []
    for item in selected:
        clean = clean_results[item.filename]
        attacked = attacked_results[item.filename]
        geo = geometry[item.filename]
        clean_detected = str(clean["detected"])
        attacked_detected = str(attacked["detected"])
        clean_confidence = float(clean["confidence"])
        clean_iou = float(clean["iou"])
        attacked_confidence = float(attacked["confidence"])
        attacked_iou = float(attacked["iou"])
        eligible = "Yes" if clean_detected == "Yes" else "No"
        attack_success = "Yes" if eligible == "Yes" and attacked_detected == "No" else "No"
        rows.append(
            AttackResultRow(
                attack_id=attack_id,
                image_id=str(item.image_id),
                filename=item.filename,
                target_class=item.target_class,
                class_id=item.class_id,
                clean_detected=clean_detected,
                clean_confidence=clean_confidence,
                clean_iou=clean_iou,
                clean_total_detections=int(clean["total_detections"]),
                attacked_detected=attacked_detected,
                attacked_confidence=attacked_confidence,
                attacked_iou=attacked_iou,
                attacked_total_detections=int(attacked["total_detections"]),
                confidence_drop=clean_confidence - attacked_confidence,
                iou_drop=clean_iou - attacked_iou,
                detection_count_change=int(attacked["total_detections"])
                - int(clean["total_detections"]),
                patch_filename=patch_filename,
                patch_size_pixels=int(geo["patch_size_pixels"]),
                patch_scale=PATCH_SCALE,
                patch_center_x=float(geo["patch_center_x"]),
                patch_center_y=float(geo["patch_center_y"]),
                eligible_for_asr=eligible,
                attack_success=attack_success,
                clean_best_same_class_confidence=float(clean["best_same_class_confidence"]),
                clean_best_same_class_iou=float(clean["best_same_class_iou"]),
                attacked_best_same_class_confidence=float(attacked["best_same_class_confidence"]),
                attacked_best_same_class_iou=float(attacked["best_same_class_iou"]),
            )
        )
    return rows


def write_attack_results_csv(path: Path, rows: list[AttackResultRow]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ATTACK_RESULTS_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def draw_verification_image(
    patched_image_path: Path,
    output_path: Path,
    item: SelectedImage,
    row: AttackResultRow,
    patch_rect: tuple[float, float, float, float],
) -> None:
    image = cv2.imread(str(patched_image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read patched image: {patched_image_path}")

    gx, gy, gw, gh = [int(round(v)) for v in (item.bbox_x, item.bbox_y, item.bbox_width, item.bbox_height)]
    cv2.rectangle(image, (gx, gy), (gx + gw, gy + gh), (0, 255, 0), 2)

    px, py, pw, ph = [int(round(v)) for v in patch_rect]
    cv2.rectangle(image, (px, py), (px + pw, py + ph), (0, 255, 255), 2)

    lines = [
        item.filename,
        f"Target: {item.target_class}",
        f"Clean: {row.clean_detected} conf={row.clean_confidence:.3f} IoU={row.clean_iou:.3f}",
        f"Attacked: {row.attacked_detected} conf={row.attacked_confidence:.3f} IoU={row.attacked_iou:.3f}",
        f"Conf drop: {row.confidence_drop:.3f}",
        f"Attack Success: {row.attack_success}",
    ]
    if row.eligible_for_asr == "No":
        lines.append("NOT ELIGIBLE FOR ASR")
    elif row.attack_success == "Yes":
        lines.append("ATTACK SUCCESS")
        lines.append("TARGET MISSED AFTER ATTACK")

    font = cv2.FONT_HERSHEY_SIMPLEX
    y = 20
    for line in lines:
        cv2.putText(image, line, (8, y), font, 0.45, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(image, line, (8, y), font, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        y += 17
    cv2.imwrite(str(output_path), image)


def create_overview_image(
    verification_dir: Path,
    rows: list[AttackResultRow],
    output_path: Path,
) -> None:
    total = len(rows)
    sample_count = min(total, OVERVIEW_MAX_THUMBS)
    if total <= OVERVIEW_MAX_THUMBS:
        indices = list(range(total))
    else:
        step = total / sample_count
        indices = [int(round(i * step)) for i in range(sample_count)]
        indices = sorted(set(min(index, total - 1) for index in indices))

    cols = 5
    rows_count = math.ceil(len(indices) / cols)
    thumb_w, thumb_h = 320, 240
    label_h = 28
    margin = 10
    canvas_w = cols * thumb_w + (cols + 1) * margin
    canvas_h = rows_count * (thumb_h + label_h) + (rows_count + 1) * margin
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for plot_index, row_index in enumerate(indices):
        row = rows[row_index]
        grid_row = plot_index // cols
        grid_col = plot_index % cols
        x = margin + grid_col * (thumb_w + margin)
        y = margin + grid_row * (thumb_h + label_h + margin)
        image_path = verification_dir / f"verify_{Path(row.filename).stem}.jpg"
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        thumb = cv2.resize(image, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        canvas[y : y + thumb_h, x : x + thumb_w] = thumb
        cv2.putText(canvas, row.filename, (x, y + thumb_h + 20), font, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.imwrite(str(output_path), canvas)


def write_attack_summary(
    path: Path,
    attack_id: str,
    random_seed: int,
    num_images: int,
    patch_filename: str,
    rows: list[AttackResultRow],
    *,
    previous_attack_overlap_count: int = 0,
    previous_attack_overlap_percent: float = 0.0,
) -> None:
    person_count = sum(1 for row in rows if row.target_class == "person")
    car_count = len(rows) - person_count
    clean_detected = sum(1 for row in rows if row.clean_detected == "Yes")
    clean_missed = len(rows) - clean_detected
    attacked_detected = sum(1 for row in rows if row.attacked_detected == "Yes")
    attacked_missed = len(rows) - attacked_detected
    eligible = [row for row in rows if row.eligible_for_asr == "Yes"]
    successes = [row for row in rows if row.attack_success == "Yes"]
    asr_denominator = len(eligible)
    asr = (len(successes) / asr_denominator * 100.0) if asr_denominator else 0.0

    mean_clean_conf = sum(row.clean_confidence for row in eligible) / len(eligible) if eligible else 0.0
    mean_attacked_conf = (
        sum(row.attacked_confidence for row in eligible) / len(eligible) if eligible else 0.0
    )
    mean_conf_drop = sum(row.confidence_drop for row in eligible) / len(eligible) if eligible else 0.0
    mean_clean_iou = sum(row.clean_iou for row in eligible) / len(eligible) if eligible else 0.0
    mean_attacked_iou = sum(row.attacked_iou for row in eligible) / len(eligible) if eligible else 0.0
    mean_iou_drop = sum(row.iou_drop for row in eligible) / len(eligible) if eligible else 0.0
    avg_clean_det = sum(row.clean_total_detections for row in rows) / len(rows)
    avg_attacked_det = sum(row.attacked_total_detections for row in rows) / len(rows)

    def class_asr(target_class: str) -> float:
        class_eligible = [row for row in eligible if row.target_class == target_class]
        class_successes = [row for row in successes if row.target_class == target_class]
        return (len(class_successes) / len(class_eligible) * 100.0) if class_eligible else 0.0

    person_asr = class_asr("person")
    car_asr = class_asr("car")

    lines = [
        f"Attack ID: {attack_id}",
        f"Random seed: {random_seed}",
        f"Number of requested images: {num_images}",
        f"Number of processed images: {len(rows)}",
        "",
        f"Target match IoU threshold: {TARGET_MATCH_IOU_THRESHOLD:.2f}",
        "",
        f"Total images: {len(rows)}",
        f"Person images: {person_count}",
        f"Car images: {car_count}",
        "",
        f"Patch filename: {patch_filename}",
        "Patch scale: 30% of shorter GT bounding-box side",
        "Patch position: Center of GT bounding box",
        "Rotation: 0 degrees",
        "Brightness: unchanged",
        "Opacity: 100%",
        "",
        f"Valid clean target detections: {clean_detected}",
        f"Clean target misses: {clean_missed}",
        "",
        f"Valid attacked target detections: {attacked_detected}",
        f"Attacked target misses: {attacked_missed}",
        "",
        f"ASR eligible targets: {asr_denominator}",
        f"Successful attacks: {len(successes)}",
        f"Attack Success Rate: {len(successes)} / {asr_denominator} = {asr:.2f}%",
        "",
        f"Person ASR: {person_asr:.2f}%",
        f"Car ASR: {car_asr:.2f}%",
        "",
        f"Previous-attack image overlap count: {previous_attack_overlap_count}",
        f"Previous-attack image overlap percent: {previous_attack_overlap_percent:.2f}%",
        "",
        f"Mean clean confidence: {mean_clean_conf:.4f}",
        f"Mean attacked confidence: {mean_attacked_conf:.4f}",
        f"Mean confidence drop: {mean_conf_drop:.4f}",
        "",
        f"Mean clean IoU: {mean_clean_iou:.4f}",
        f"Mean attacked IoU: {mean_attacked_iou:.4f}",
        f"Mean IoU drop: {mean_iou_drop:.4f}",
        "",
        f"Average clean detections/image: {avg_clean_det:.4f}",
        f"Average attacked detections/image: {avg_attacked_det:.4f}",
        "",
        f"Model path: {MODEL_PATH}",
        f"conf: {CONF_THRESHOLD}",
        f"iou: {IOU_THRESHOLD}",
        f"target_match_iou: {TARGET_MATCH_IOU_THRESHOLD}",
        f"imgsz: {IMAGE_SIZE}",
        f"device: {DEVICE}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_selected_csv(path: Path) -> list[SelectedImage]:
    selected: list[SelectedImage] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            selected.append(
                SelectedImage(
                    image_id=int(row["image_id"]),
                    filename=row["filename"],
                    target_class=row["target_class"],
                    class_id=int(row["class_id"]),
                    bbox_x=float(row["bbox_x"]),
                    bbox_y=float(row["bbox_y"]),
                    bbox_width=float(row["bbox_width"]),
                    bbox_height=float(row["bbox_height"]),
                    image_width=int(row["image_width"]),
                    image_height=int(row["image_height"]),
                    random_seed=int(row["random_seed"]),
                )
            )
    selected.sort(key=lambda item: item.filename)
    return selected


def recalculate_attack_evaluation(
    attack_dir: Path,
    patch_filename: str | None = None,
) -> tuple[
    list[AttackResultRow],
    list[SelectedImage],
    dict[str, dict[str, float | int]],
    dict[str, dict[str, float | int | str]],
]:
    """Recalculate evaluation outputs from saved predictions without rerunning YOLO."""
    attack_id = attack_dir.name
    results_dir = attack_dir / "results"
    selected = load_selected_csv(results_dir / "selected_images.csv")

    if patch_filename is None:
        with (results_dir / "attack_results.csv").open("r", encoding="utf-8", newline="") as handle:
            first_row = next(csv.DictReader(handle))
            patch_filename = first_row["patch_filename"]

    model = load_yolo()
    clean_results = evaluate_predictions(
        selected,
        results_dir / "clean_predictions" / "labels",
        model,
    )
    attacked_results = evaluate_predictions(
        selected,
        results_dir / "attacked_predictions" / "labels",
        model,
    )

    geometry: dict[str, dict[str, float | int]] = {}
    for item in selected:
        gt_bbox = (item.bbox_x, item.bbox_y, item.bbox_width, item.bbox_height)
        patch_size, center_x, center_y, patch_x, patch_y = compute_patch_geometry(
            gt_bbox, item.image_width, item.image_height
        )
        geometry[item.filename] = {
            "patch_size_pixels": patch_size,
            "patch_center_x": center_x,
            "patch_center_y": center_y,
            "patch_x": patch_x,
            "patch_y": patch_y,
        }

    attack_rows = build_attack_rows(
        attack_id,
        selected,
        clean_results,
        attacked_results,
        geometry,
        patch_filename,
    )
    return attack_rows, selected, geometry, clean_results
