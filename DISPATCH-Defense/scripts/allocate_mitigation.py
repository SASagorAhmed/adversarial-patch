#!/usr/bin/env python
"""Allocate next mitigation_XX folder under DISPATCH-Defense/mitigations/."""
from __future__ import annotations

import re
from pathlib import Path

from config.paths_config import MITIGATIONS_DIR, assert_under_project_root, ensure_dir

_MIT_RE = re.compile(r"^mitigation_(\d+)$")


def list_mitigation_ids(mitigations_dir: Path | None = None) -> list[int]:
    root = assert_under_project_root(mitigations_dir or MITIGATIONS_DIR)
    if not root.is_dir():
        return []
    ids = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        m = _MIT_RE.match(p.name)
        if m:
            ids.append(int(m.group(1)))
    return sorted(ids)


def next_mitigation_id(mitigations_dir: Path | None = None) -> str:
    ids = list_mitigation_ids(mitigations_dir)
    n = (max(ids) + 1) if ids else 1
    return f"mitigation_{n:02d}"


def create_mitigation_folder(mitigation_id: str | None = None) -> Path:
    """Create next (or specified) mitigation folder with required subdirs. Never overwrite."""
    ensure_dir(MITIGATIONS_DIR)
    mid = mitigation_id or next_mitigation_id()
    root = assert_under_project_root(MITIGATIONS_DIR / mid)
    if root.exists():
        raise RuntimeError(f"REFUSING overwrite: {root} already exists")
    subdirs = [
        "source_attack_data",
        "source_attacked_images",
        "source_clean_images",
        "masks/checkerboard_m0",
        "masks/checkerboard_m1",
        "masks/adversarial_masks",
        "masks/evaluation_true_patch_masks",
        "neutralized_inputs/masked_input_m0",
        "neutralized_inputs/masked_input_m1",
        "restored_images",
        "results/regenerated/pass_0",
        "results/regenerated/pass_1",
        "results/regenerated/full",
        "results/difference_maps/raw_numeric",
        "results/difference_maps/raw_visualization",
        "results/difference_maps/smoothed",
        "results/clean_predictions/images",
        "results/clean_predictions/labels",
        "results/attacked_predictions/images",
        "results/attacked_predictions/labels",
        "results/restored_predictions/images",
        "results/restored_predictions/labels",
        "results/verification",
        "results/plots",
        "results/logs",
    ]
    for rel in subdirs:
        ensure_dir(root / rel)
    return root
