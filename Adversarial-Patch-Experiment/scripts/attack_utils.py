#!/usr/bin/env python3
"""Utilities for dynamic attack folder creation and numbering."""

from __future__ import annotations

import csv
import re
import shutil
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
HYPER_YOLO = EXPERIMENT_ROOT.parent / "Hyper-YOLO"
ATTACKS_DIR = HYPER_YOLO / "attacks"
DIFFUSION_PATCHES_DIR = EXPERIMENT_ROOT / "diffusion_patches"
RESULTS_DIR = EXPERIMENT_ROOT / "results"
COMPARISON_CSV = RESULTS_DIR / "all_attacks_comparison.csv"

ATTACK_DIR_PATTERN = re.compile(r"^attack_(\d+)$")

SUBFOLDERS = (
    "patch",
    "patched_images",
    "prediction_images",
    "labels",
    "verification",
)

ATTACK_RESULTS_COLUMNS = [
    "attack_id",
    "patch_file",
    "image_id",
    "filename",
    "target_class",
    "class_id",
    "gt_bbox_x",
    "gt_bbox_y",
    "gt_bbox_width",
    "gt_bbox_height",
    "clean_detected",
    "clean_confidence",
    "clean_iou",
    "attack_detected",
    "attack_confidence",
    "attack_iou",
    "total_detections",
    "verification_image",
]

COMPARISON_COLUMNS = [
    "attack_id",
    "patch_file",
    "filename",
    "target_class",
    "clean_detected",
    "clean_confidence",
    "clean_iou",
    "attack_detected",
    "attack_confidence",
    "attack_iou",
    "detection_suppressed",
    "verification_image",
]

PATCH_USED_NAME = "patch_used.png"


def attack_id_from_number(number: int) -> str:
    return f"attack_{number:02d}"


def list_existing_attack_numbers() -> list[int]:
    if not ATTACKS_DIR.is_dir():
        return []

    numbers: list[int] = []
    for path in ATTACKS_DIR.iterdir():
        if not path.is_dir():
            continue
        match = ATTACK_DIR_PATTERN.match(path.name)
        if match:
            numbers.append(int(match.group(1)))
    return sorted(numbers)


def get_next_attack_number() -> int:
    existing = list_existing_attack_numbers()
    if not existing:
        return 1
    return max(existing) + 1


def get_next_attack_id() -> str:
    return attack_id_from_number(get_next_attack_number())


def attack_dir_for_id(attack_id: str) -> Path:
    return ATTACKS_DIR / attack_id


def ensure_attacks_root() -> None:
    ATTACKS_DIR.mkdir(parents=True, exist_ok=True)


def ensure_comparison_template() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not COMPARISON_CSV.is_file():
        with COMPARISON_CSV.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=COMPARISON_COLUMNS)
            writer.writeheader()


def create_attack_folder(attack_id: str | None = None) -> Path:
    """Create the folder structure for a new attack when it actually starts."""
    ensure_attacks_root()
    ensure_comparison_template()

    if attack_id is None:
        attack_id = get_next_attack_id()

    attack_dir = attack_dir_for_id(attack_id)
    if attack_dir.exists():
        raise FileExistsError(f"Attack folder already exists: {attack_dir}")

    for subfolder in SUBFOLDERS:
        (attack_dir / subfolder).mkdir(parents=True, exist_ok=False)

    results_csv = attack_dir / "attack_results.csv"
    with results_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ATTACK_RESULTS_COLUMNS)
        writer.writeheader()

    summary_path = attack_dir / "attack_summary.txt"
    summary_path.write_text(
        "\n".join(
            [
                f"{attack_id} summary",
                "=" * 72,
                "status: started",
                f"patch file: {attack_dir / 'patch' / PATCH_USED_NAME}",
                "",
                "Output folders:",
                f"- patched_images: {attack_dir / 'patched_images'}",
                f"- prediction_images: {attack_dir / 'prediction_images'}",
                f"- labels: {attack_dir / 'labels'}",
                f"- verification: {attack_dir / 'verification'}",
                f"- attack_results.csv: {results_csv}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return attack_dir


def copy_candidate_patch_to_attack(
    attack_dir: Path,
    candidate_patch: Path,
    destination_name: str = PATCH_USED_NAME,
) -> Path:
    """Copy a candidate patch from diffusion_patches into the attack folder."""
    if not candidate_patch.is_file():
        raise FileNotFoundError(f"Candidate patch not found: {candidate_patch}")

    destination = attack_dir / "patch" / destination_name
    shutil.copy2(candidate_patch, destination)
    return destination


def attacks_folder_is_empty() -> bool:
    if not ATTACKS_DIR.is_dir():
        return True
    return not any(ATTACKS_DIR.iterdir())
