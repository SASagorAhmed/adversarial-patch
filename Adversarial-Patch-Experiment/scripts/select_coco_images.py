#!/usr/bin/env python3
"""Select 20 COCO val2017 images (10 person, 10 car) for adversarial patch experiments."""

from __future__ import annotations

import csv
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
HYPER_YOLO = EXPERIMENT_ROOT.parent / "Hyper-YOLO"
CLEAN_IMAGES_DIR = EXPERIMENT_ROOT / "clean_images"
RESULTS_DIR = EXPERIMENT_ROOT / "results"
CSV_PATH = RESULTS_DIR / "selected_images.csv"

ANNOTATION_CANDIDATES = [
    HYPER_YOLO / "annotations" / "instances_val2017.json",
    HYPER_YOLO / "coco" / "annotations" / "instances_val2017.json",
]

TARGET_CLASSES = {
    "person": 1,
    "car": 3,
}

PERSON_COUNT = 10
CAR_COUNT = 10
MIN_DIM_RATIO = 0.05


@dataclass(frozen=True)
class Candidate:
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
    score: float


def find_annotation_path() -> Path:
    inspected = []
    for candidate in ANNOTATION_CANDIDATES:
        inspected.append(str(candidate))
        if candidate.is_file():
            return candidate

    print("ERROR: Could not find COCO annotations.", file=sys.stderr)
    print("Inspected paths:", file=sys.stderr)
    for path in inspected:
        print(f"  - {path}", file=sys.stderr)
    sys.exit(1)


def find_image_dir() -> Path:
    coco_root = HYPER_YOLO / "coco"
    inspected_dirs: list[Path] = []

    if not coco_root.is_dir():
        print("ERROR: Hyper-YOLO COCO root not found.", file=sys.stderr)
        print(f"Inspected: {coco_root}", file=sys.stderr)
        sys.exit(1)

    best_dir: Path | None = None
    best_count = 0

    for directory in coco_root.rglob("val2017"):
        if not directory.is_dir():
            continue
        inspected_dirs.append(directory)
        jpg_count = sum(1 for _ in directory.glob("*.jpg"))
        if jpg_count > best_count:
            best_count = jpg_count
            best_dir = directory

    if best_dir is None or best_count == 0:
        print("ERROR: Could not find COCO val2017 image directory.", file=sys.stderr)
        print("Inspected folders:", file=sys.stderr)
        for directory in inspected_dirs:
            print(f"  - {directory}", file=sys.stderr)
        if not inspected_dirs:
            print(f"  - {coco_root} (no val2017 subdirectories found)", file=sys.stderr)
        sys.exit(1)

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


def build_candidates(coco: dict) -> dict[str, list[Candidate]]:
    images_by_id = {image["id"]: image for image in coco["images"]}
    candidates: dict[str, list[Candidate]] = {name: [] for name in TARGET_CLASSES}

    per_image_best: dict[tuple[str, int], Candidate] = {}

    for annotation in coco["annotations"]:
        category_id = annotation["category_id"]
        target_class = None
        for class_name, class_id in TARGET_CLASSES.items():
            if category_id == class_id:
                target_class = class_name
                break
        if target_class is None:
            continue

        image_id = annotation["image_id"]
        image = images_by_id.get(image_id)
        if image is None:
            continue

        bbox_x, bbox_y, bbox_width, bbox_height = annotation["bbox"]
        image_width = image["width"]
        image_height = image["height"]

        if not bbox_is_large_enough(bbox_width, bbox_height, image_width, image_height):
            continue

        area_ratio = (bbox_width * bbox_height) / (image_width * image_height)
        candidate = Candidate(
            image_id=image_id,
            filename=image["file_name"],
            target_class=target_class,
            class_id=category_id,
            bbox_x=bbox_x,
            bbox_y=bbox_y,
            bbox_width=bbox_width,
            bbox_height=bbox_height,
            image_width=image_width,
            image_height=image_height,
            score=area_ratio,
        )

        key = (target_class, image_id)
        existing = per_image_best.get(key)
        if existing is None or candidate.score > existing.score:
            per_image_best[key] = candidate

    for (target_class, _), candidate in per_image_best.items():
        candidates[target_class].append(candidate)

    for class_name in candidates:
        candidates[class_name].sort(key=lambda item: item.score, reverse=True)

    return candidates


def select_unique_candidates(
    candidates: dict[str, list[Candidate]],
) -> list[Candidate]:
    selected: list[Candidate] = []
    used_filenames: set[str] = set()

    for target_class, count in (("person", PERSON_COUNT), ("car", CAR_COUNT)):
        picked = 0
        for candidate in candidates[target_class]:
            if candidate.filename in used_filenames:
                continue
            selected.append(candidate)
            used_filenames.add(candidate.filename)
            picked += 1
            if picked >= count:
                break

        if picked < count:
            print(
                f"ERROR: Only found {picked} usable {target_class} images "
                f"(needed {count}).",
                file=sys.stderr,
            )
            sys.exit(1)

    return selected


def copy_images(selected: list[Candidate], image_dir: Path) -> None:
    CLEAN_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    for candidate in selected:
        source = image_dir / candidate.filename
        if not source.is_file():
            print(f"ERROR: Missing source image: {source}", file=sys.stderr)
            sys.exit(1)
        shutil.copy2(source, CLEAN_IMAGES_DIR / candidate.filename)


def write_csv(selected: list[Candidate]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
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
    ]

    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for candidate in selected:
            writer.writerow(
                {
                    "image_id": candidate.image_id,
                    "filename": candidate.filename,
                    "target_class": candidate.target_class,
                    "class_id": candidate.class_id,
                    "bbox_x": candidate.bbox_x,
                    "bbox_y": candidate.bbox_y,
                    "bbox_width": candidate.bbox_width,
                    "bbox_height": candidate.bbox_height,
                    "image_width": candidate.image_width,
                    "image_height": candidate.image_height,
                }
            )


def verify_outputs(selected: list[Candidate]) -> None:
    copied = list(CLEAN_IMAGES_DIR.glob("*.jpg"))
    if len(copied) != len(selected):
        print(
            f"ERROR: Expected {len(selected)} copied images, found {len(copied)}.",
            file=sys.stderr,
        )
        sys.exit(1)

    for candidate in selected:
        destination = CLEAN_IMAGES_DIR / candidate.filename
        if not destination.is_file():
            print(f"ERROR: Missing copied image: {destination}", file=sys.stderr)
            sys.exit(1)

    if not CSV_PATH.is_file():
        print(f"ERROR: CSV not created: {CSV_PATH}", file=sys.stderr)
        sys.exit(1)


def print_summary(
    image_dir: Path,
    annotation_path: Path,
    selected: list[Candidate],
) -> None:
    person_count = sum(1 for item in selected if item.target_class == "person")
    car_count = sum(1 for item in selected if item.target_class == "car")

    print("COCO selection summary")
    print(f"- detected COCO image path: {image_dir}")
    print(f"- detected annotation path: {annotation_path}")
    print(f"- person images selected: {person_count}")
    print(f"- car images selected: {car_count}")
    print(f"- total unique images copied: {len(selected)}")
    print(f"- CSV output path: {CSV_PATH}")


def main() -> None:
    if not HYPER_YOLO.is_dir():
        print(f"ERROR: Hyper-YOLO project not found: {HYPER_YOLO}", file=sys.stderr)
        sys.exit(1)

    annotation_path = find_annotation_path()
    image_dir = find_image_dir()

    with annotation_path.open("r", encoding="utf-8") as handle:
        coco = json.load(handle)

    candidates = build_candidates(coco)
    selected = select_unique_candidates(candidates)
    copy_images(selected, image_dir)
    write_csv(selected)
    verify_outputs(selected)
    print_summary(image_dir, annotation_path, selected)


if __name__ == "__main__":
    main()
