#!/usr/bin/env python3
"""Select deterministic success + control subsets for mitigation experiments."""

from __future__ import annotations

import csv
import json
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from mitigation_config import (
    MITIGATION_SPECS,
    PATCH_SCALE,
    MitigationSpec,
    attack_dir,
    get_mitigation_spec,
)


@dataclass
class PatchGeometry:
    patch_size: int
    center_x: float
    center_y: float
    patch_x: float
    patch_y: float
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass
class SubsetRow:
    mitigation_id: str
    source_attack_id: str
    subset_role: str  # success | control
    image_id: str
    filename: str
    target_class: str
    class_id: int
    bbox_x: float
    bbox_y: float
    bbox_width: float
    bbox_height: float
    image_width: int
    image_height: int
    attack_random_seed: int
    clean_detected: str
    attacked_detected: str
    eligible_for_asr: str
    attack_success: str
    clean_confidence: float
    attacked_confidence: float
    clean_iou: float
    attacked_iou: float
    patch_filename: str
    patch_size_pixels: int
    patch_scale: float
    patch_center_x: float
    patch_center_y: float
    reconstructed_patch_size: int
    reconstructed_patch_x: float
    reconstructed_patch_y: float
    reconstructed_x1: int
    reconstructed_y1: int
    reconstructed_x2: int
    reconstructed_y2: int
    control_selection_seed: int


SUBSET_COLUMNS = [
    "mitigation_id",
    "source_attack_id",
    "subset_role",
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
    "attack_random_seed",
    "clean_detected",
    "attacked_detected",
    "eligible_for_asr",
    "attack_success",
    "clean_confidence",
    "attacked_confidence",
    "clean_iou",
    "attacked_iou",
    "patch_filename",
    "patch_size_pixels",
    "patch_scale",
    "patch_center_x",
    "patch_center_y",
    "reconstructed_patch_size",
    "reconstructed_patch_x",
    "reconstructed_patch_y",
    "reconstructed_x1",
    "reconstructed_y1",
    "reconstructed_x2",
    "reconstructed_y2",
    "control_selection_seed",
]


def reconstruct_patch_geometry(
    bbox_x: float,
    bbox_y: float,
    bbox_width: float,
    bbox_height: float,
    image_width: int,
    image_height: int,
) -> PatchGeometry:
    """Reproduce the frozen attack patch placement formula exactly."""
    patch_size = max(1, int(round(PATCH_SCALE * min(bbox_width, bbox_height))))
    center_x = bbox_x + bbox_width / 2.0
    center_y = bbox_y + bbox_height / 2.0
    patch_x = center_x - patch_size / 2.0
    patch_y = center_y - patch_size / 2.0
    patch_x = max(0.0, min(patch_x, image_width - patch_size))
    patch_y = max(0.0, min(patch_y, image_height - patch_size))
    x1 = int(round(patch_x))
    y1 = int(round(patch_y))
    x2 = x1 + patch_size
    y2 = y1 + patch_size
    return PatchGeometry(
        patch_size=patch_size,
        center_x=center_x,
        center_y=center_y,
        patch_x=patch_x,
        patch_y=patch_y,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
    )


def load_attack_tables(attack_id: str) -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    root = attack_dir(attack_id)
    attack_csv = root / "results" / "attack_results.csv"
    selected_csv = root / "results" / "selected_images.csv"
    if not attack_csv.is_file():
        raise FileNotFoundError(f"Missing attack results CSV: {attack_csv}")
    if not selected_csv.is_file():
        raise FileNotFoundError(f"Missing selected images CSV: {selected_csv}")

    with attack_csv.open("r", encoding="utf-8", newline="") as handle:
        attack_rows = list(csv.DictReader(handle))
    with selected_csv.open("r", encoding="utf-8", newline="") as handle:
        selected_by_filename = {row["filename"]: row for row in csv.DictReader(handle)}
    return attack_rows, selected_by_filename


def is_success_row(row: dict[str, str]) -> bool:
    return (
        row.get("eligible_for_asr") == "Yes"
        and row.get("clean_detected") == "Yes"
        and row.get("attacked_detected") == "No"
        and row.get("attack_success") == "Yes"
    )


def is_control_row(row: dict[str, str]) -> bool:
    return (
        row.get("eligible_for_asr") == "Yes"
        and row.get("attack_success") == "No"
        and row.get("attacked_detected") == "Yes"
    )


def build_subset_row(
    spec: MitigationSpec,
    role: str,
    attack_row: dict[str, str],
    selected: dict[str, str],
) -> SubsetRow:
    filename = attack_row["filename"]
    bbox_x = float(selected["bbox_x"])
    bbox_y = float(selected["bbox_y"])
    bbox_width = float(selected["bbox_width"])
    bbox_height = float(selected["bbox_height"])
    image_width = int(selected["image_width"])
    image_height = int(selected["image_height"])
    geometry = reconstruct_patch_geometry(
        bbox_x, bbox_y, bbox_width, bbox_height, image_width, image_height
    )
    recorded_size = int(float(attack_row["patch_size_pixels"]))
    if geometry.patch_size != recorded_size:
        raise ValueError(
            f"Patch size mismatch for {filename}: "
            f"reconstructed={geometry.patch_size}, recorded={recorded_size}"
        )

    return SubsetRow(
        mitigation_id=spec.mitigation_id,
        source_attack_id=spec.source_attack_id,
        subset_role=role,
        image_id=str(attack_row["image_id"]),
        filename=filename,
        target_class=attack_row["target_class"],
        class_id=int(float(attack_row["class_id"])),
        bbox_x=bbox_x,
        bbox_y=bbox_y,
        bbox_width=bbox_width,
        bbox_height=bbox_height,
        image_width=image_width,
        image_height=image_height,
        attack_random_seed=int(selected["random_seed"]),
        clean_detected=attack_row["clean_detected"],
        attacked_detected=attack_row["attacked_detected"],
        eligible_for_asr=attack_row["eligible_for_asr"],
        attack_success=attack_row["attack_success"],
        clean_confidence=float(attack_row["clean_confidence"]),
        attacked_confidence=float(attack_row["attacked_confidence"]),
        clean_iou=float(attack_row["clean_iou"]),
        attacked_iou=float(attack_row["attacked_iou"]),
        patch_filename=str(attack_row.get("patch_filename", "")),
        patch_size_pixels=recorded_size,
        patch_scale=float(attack_row.get("patch_scale", PATCH_SCALE)),
        patch_center_x=float(attack_row["patch_center_x"]),
        patch_center_y=float(attack_row["patch_center_y"]),
        reconstructed_patch_size=geometry.patch_size,
        reconstructed_patch_x=geometry.patch_x,
        reconstructed_patch_y=geometry.patch_y,
        reconstructed_x1=geometry.x1,
        reconstructed_y1=geometry.y1,
        reconstructed_x2=geometry.x2,
        reconstructed_y2=geometry.y2,
        control_selection_seed=spec.control_selection_seed,
    )


def _balanced_control_sample(
    controls: list[dict[str, str]],
    count: int,
    seed: int,
) -> list[dict[str, str]]:
    if len(controls) < count:
        raise RuntimeError(
            f"Not enough control candidates (need {count}, found {len(controls)})"
        )
    rng = random.Random(seed)
    person = [row for row in controls if row["target_class"] == "person"]
    car = [row for row in controls if row["target_class"] == "car"]
    person_n = count // 2
    car_n = count - person_n

    selected: list[dict[str, str]] = []
    if len(person) >= person_n and len(car) >= car_n:
        selected.extend(rng.sample(person, person_n))
        selected.extend(rng.sample(car, car_n))
    else:
        selected = rng.sample(controls, count)

    selected.sort(key=lambda row: row["filename"])
    return selected


def select_mitigation_subset(spec: MitigationSpec) -> list[SubsetRow]:
    attack_rows, selected_by_filename = load_attack_tables(spec.source_attack_id)

    successes = [row for row in attack_rows if is_success_row(row)]
    controls_pool = [row for row in attack_rows if is_control_row(row)]
    successes.sort(key=lambda row: row["filename"])
    controls = _balanced_control_sample(
        controls_pool, spec.control_count, spec.control_selection_seed
    )

    subset: list[SubsetRow] = []
    for row in successes:
        selected = selected_by_filename.get(row["filename"])
        if selected is None:
            raise KeyError(f"Missing GT row for success image: {row['filename']}")
        subset.append(build_subset_row(spec, "success", row, selected))

    for row in controls:
        selected = selected_by_filename.get(row["filename"])
        if selected is None:
            raise KeyError(f"Missing GT row for control image: {row['filename']}")
        subset.append(build_subset_row(spec, "control", row, selected))

    subset.sort(key=lambda item: (item.subset_role, item.filename))
    return subset


def write_subset_csv(path: Path, rows: list[SubsetRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUBSET_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


SELECTED_ATTACK_RESULTS_COLUMNS = [
    "attack_id",
    "image_id",
    "filename",
    "target_class",
    "clean_detected",
    "attacked_detected",
    "clean_confidence",
    "attacked_confidence",
    "clean_iou",
    "attacked_iou",
    "eligible_for_asr",
    "attack_success",
    "patch_filename",
    "patch_size_pixels",
    "patch_scale",
    "patch_center_x",
    "patch_center_y",
]

SELECTED_IMAGES_METADATA_COLUMNS = [
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
    "patch_x1",
    "patch_y1",
    "patch_x2",
    "patch_y2",
    "patch_size_pixels",
    "subset_role",
]


def write_selected_attack_results_csv(path: Path, rows: list[SubsetRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SELECTED_ATTACK_RESULTS_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "attack_id": row.source_attack_id,
                    "image_id": row.image_id,
                    "filename": row.filename,
                    "target_class": row.target_class,
                    "clean_detected": row.clean_detected,
                    "attacked_detected": row.attacked_detected,
                    "clean_confidence": row.clean_confidence,
                    "attacked_confidence": row.attacked_confidence,
                    "clean_iou": row.clean_iou,
                    "attacked_iou": row.attacked_iou,
                    "eligible_for_asr": row.eligible_for_asr,
                    "attack_success": row.attack_success,
                    "patch_filename": row.patch_filename,
                    "patch_size_pixels": row.patch_size_pixels,
                    "patch_scale": row.patch_scale,
                    "patch_center_x": row.patch_center_x,
                    "patch_center_y": row.patch_center_y,
                }
            )


def write_selected_images_metadata_csv(path: Path, rows: list[SubsetRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SELECTED_IMAGES_METADATA_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "image_id": row.image_id,
                    "filename": row.filename,
                    "target_class": row.target_class,
                    "class_id": row.class_id,
                    "bbox_x": row.bbox_x,
                    "bbox_y": row.bbox_y,
                    "bbox_width": row.bbox_width,
                    "bbox_height": row.bbox_height,
                    "image_width": row.image_width,
                    "image_height": row.image_height,
                    "random_seed": row.attack_random_seed,
                    "patch_x1": row.reconstructed_x1,
                    "patch_y1": row.reconstructed_y1,
                    "patch_x2": row.reconstructed_x2,
                    "patch_y2": row.reconstructed_y2,
                    "patch_size_pixels": row.reconstructed_patch_size,
                    "subset_role": row.subset_role,
                }
            )


def build_source_info(
    spec: MitigationSpec,
    rows: list[SubsetRow],
) -> dict:
    summary = summarize_subset(rows)
    patch_name = rows[0].patch_filename if rows else ""
    attack_seed = rows[0].attack_random_seed if rows else None
    return {
        "mitigation_id": spec.mitigation_id,
        "source_attack_id": spec.source_attack_id,
        "source_attack_folder": str(attack_dir(spec.source_attack_id)),
        "source_patch_filename": patch_name,
        "source_attack_random_seed": attack_seed,
        "target_match_iou_threshold": 0.50,
        "number_success_cases": summary["success_count"],
        "number_control_cases": summary["control_count"],
        "person_success_count": summary["success_person"],
        "car_success_count": summary["success_car"],
        "person_control_count": summary["control_person"],
        "car_control_count": summary["control_car"],
        "control_selection_seed": spec.control_selection_seed,
        "restoration_seed": spec.restoration_seed,
        "source_attack_data_mode": "READ_ONLY_COPY",
        "clean_images_usage": "evaluation_and_verification_only_not_used_for_restoration",
    }


def save_source_attack_package(
    mitigation_root: Path,
    spec: MitigationSpec,
    rows: list[SubsetRow],
) -> None:
    """Write selected_subset + copied attack metadata under source_attack_data/."""
    data_dir = mitigation_root / "source_attack_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    write_subset_csv(data_dir / "selected_subset.csv", rows)
    write_selected_attack_results_csv(data_dir / "selected_attack_results.csv", rows)
    write_selected_images_metadata_csv(data_dir / "selected_images_metadata.csv", rows)
    info = build_source_info(spec, rows)
    (data_dir / "source_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


def copy_source_images(mitigation_root: Path, rows: list[SubsetRow]) -> None:
    """Copy clean and attacked images into mitigation folders (never move originals)."""
    clean_dest = mitigation_root / "source_clean_images"
    attacked_dest = mitigation_root / "source_attacked_images"
    clean_dest.mkdir(parents=True, exist_ok=True)
    attacked_dest.mkdir(parents=True, exist_ok=True)
    for row in rows:
        attack_root = attack_dir(row.source_attack_id)
        clean_src = attack_root / "clean_images" / row.filename
        attacked_src = attack_root / "patched_images" / row.filename
        if not clean_src.is_file():
            raise FileNotFoundError(f"Missing clean source image: {clean_src}")
        if not attacked_src.is_file():
            raise FileNotFoundError(f"Missing attacked source image: {attacked_src}")
        shutil.copy2(clean_src, clean_dest / row.filename)
        shutil.copy2(attacked_src, attacked_dest / row.filename)


def summarize_subset(rows: list[SubsetRow]) -> dict[str, int]:
    success = [row for row in rows if row.subset_role == "success"]
    control = [row for row in rows if row.subset_role == "control"]
    return {
        "total": len(rows),
        "success_count": len(success),
        "control_count": len(control),
        "person_count": sum(1 for row in rows if row.target_class == "person"),
        "car_count": sum(1 for row in rows if row.target_class == "car"),
        "success_person": sum(1 for row in success if row.target_class == "person"),
        "success_car": sum(1 for row in success if row.target_class == "car"),
        "control_person": sum(1 for row in control if row.target_class == "person"),
        "control_car": sum(1 for row in control if row.target_class == "car"),
    }


def validate_geometry_for_attack(attack_id: str) -> tuple[int, int, list[str]]:
    """Return (checked, mismatches, error_messages) for all attack rows."""
    attack_rows, selected_by_filename = load_attack_tables(attack_id)
    errors: list[str] = []
    checked = 0
    for row in attack_rows:
        selected = selected_by_filename.get(row["filename"])
        if selected is None:
            errors.append(f"Missing GT for {row['filename']}")
            continue
        geometry = reconstruct_patch_geometry(
            float(selected["bbox_x"]),
            float(selected["bbox_y"]),
            float(selected["bbox_width"]),
            float(selected["bbox_height"]),
            int(selected["image_width"]),
            int(selected["image_height"]),
        )
        recorded = int(float(row["patch_size_pixels"]))
        checked += 1
        if geometry.patch_size != recorded:
            errors.append(
                f"{row['filename']}: reconstructed={geometry.patch_size} recorded={recorded}"
            )
    return checked, len(errors), errors


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Select mitigation subset (dry write optional).")
    parser.add_argument(
        "--mitigation-id",
        required=True,
        choices=sorted(MITIGATION_SPECS.keys()),
    )
    parser.add_argument("--output", default=None, help="Optional CSV output path.")
    args = parser.parse_args()
    spec = get_mitigation_spec(args.mitigation_id)
    rows = select_mitigation_subset(spec)
    summary = summarize_subset(rows)
    print(json.dumps({"spec": asdict(spec), "summary": summary}, indent=2))
    if args.output:
        write_subset_csv(Path(args.output), rows)
        print(f"Wrote: {args.output}")
