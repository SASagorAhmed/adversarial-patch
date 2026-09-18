#!/usr/bin/env python
"""Copy 5-image sources and write hashes_before. Teacher-Notebook writes only."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(r"D:\project CS\DISPATCH-Teacher-Notebook").resolve()
sys.path.insert(0, str(ROOT))
from helpers.safety import assert_writable, ensure_dir, load_config

FILENAMES = [
    "000000127263.jpg",
    "000000393093.jpg",
    "000000026926.jpg",
    "000000407083.jpg",
    "000000331604.jpg",
]
EXPECTED = {
    "000000127263.jpg": dict(target_class="car", case_type="attack_success", patch_size="20", patch_x1="561", patch_y1="131", patch_x2="581", patch_y2="151"),
    "000000393093.jpg": dict(target_class="car", case_type="attack_success", patch_size="14", patch_x1="313", patch_y1="371", patch_x2="327", patch_y2="385"),
    "000000026926.jpg": dict(target_class="car", case_type="attack_success", patch_size="22", patch_x1="160", patch_y1="192", patch_x2="182", patch_y2="214"),
    "000000407083.jpg": dict(target_class="car", case_type="attack_success", patch_size="144", patch_x1="168", patch_y1="241", patch_x2="312", patch_y2="385"),
    "000000331604.jpg": dict(target_class="person", case_type="control", patch_size="21", patch_x1="320", patch_y1="210", patch_x2="341", patch_y2="231"),
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    cfg = load_config()
    src_csv = Path(cfg["dispatch_root"]) / "testing_mitigations" / "mitigation_02" / "automatic" / "source_attack_data" / "selected_images.csv"
    att_src = Path(cfg["paper_faithful_root"]) / "diagnostic" / "source_attacked_images"
    cln_src = Path(cfg["paper_faithful_root"]) / "diagnostic" / "source_clean_images"
    with src_csv.open(encoding="utf-8", newline="") as f:
        by = {r["filename"]: r for r in csv.DictReader(f)}
    mismatches = []
    for fn, exp in EXPECTED.items():
        if fn not in by:
            mismatches.append(f"MISSING {fn}")
            continue
        r = by[fn]
        for k, v in exp.items():
            if str(r[k]).strip() != str(v).strip():
                mismatches.append(f"{fn} {k}: frozen={r[k]!r} expected={v!r}")
    if mismatches:
        print("STOP metadata mismatch:")
        print("\n".join(mismatches))
        return 2

    att_dir = ensure_dir(ROOT / "source_data" / "attacked_images")
    cln_dir = ensure_dir(ROOT / "source_data" / "clean_images")
    meta_dir = ensure_dir(ROOT / "source_data" / "source_attack_data")
    fields = [
        "image_id", "filename", "target_class", "case_type",
        "image_width", "image_height", "gt_x", "gt_y", "gt_width", "gt_height",
        "patch_x1", "patch_y1", "patch_x2", "patch_y2", "patch_size",
        "clean_detected", "clean_confidence", "clean_iou",
        "attacked_detected", "attacked_confidence", "attacked_iou",
        "attack_success", "source_attack_id",
    ]
    out_rows = []
    for fn in FILENAMES:
        r = by[fn]
        sa, sc = att_src / fn, cln_src / fn
        if not sa.is_file() or not sc.is_file():
            print("STOP missing image", fn, sa.is_file(), sc.is_file())
            return 2
        shutil.copy2(sa, assert_writable(att_dir / fn))
        shutil.copy2(sc, assert_writable(cln_dir / fn))
        out_rows.append({
            "image_id": r["image_id"],
            "filename": fn,
            "target_class": r["target_class"],
            "case_type": r["case_type"],
            "image_width": r["image_width"],
            "image_height": r["image_height"],
            "gt_x": r["gt_x"],
            "gt_y": r["gt_y"],
            "gt_width": r["gt_width"],
            "gt_height": r["gt_height"],
            "patch_x1": r["patch_x1"],
            "patch_y1": r["patch_y1"],
            "patch_x2": r["patch_x2"],
            "patch_y2": r["patch_y2"],
            "patch_size": r["patch_size"],
            "clean_detected": r["source_clean_detected"],
            "clean_confidence": r["source_clean_confidence"],
            "clean_iou": r["source_clean_iou"],
            "attacked_detected": r["source_attacked_detected"],
            "attacked_confidence": r["source_attacked_confidence"],
            "attacked_iou": r["source_attacked_iou"],
            "attack_success": r["attack_success"],
            "source_attack_id": r["source_attack_id"],
        })
    dest = assert_writable(meta_dir / "selected_images.csv")
    with dest.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)
    shutil.copy2(src_csv, assert_writable(meta_dir / "frozen_testing_mit02_selected_images.csv"))

    hash_targets = [
        Path(cfg["dispatch_root"]) / "config" / "dispatch_config.py",
        Path(cfg["paper_faithful_root"]) / "implementation" / "ldm_adapter.py",
        Path(cfg["paper_faithful_root"]) / "implementation" / "localization.py",
        Path(cfg["hyper_yolo_weights"]),
        src_csv,
    ]
    hashes = {}
    for p in hash_targets:
        hashes[str(p)] = {"sha256": sha256(p), "size_bytes": p.stat().st_size}
    ensure_dir(ROOT / "safety")
    (assert_writable(ROOT / "safety" / "hashes_before.json")).write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    print("Copied 5 clean + 5 attacked. Metadata MATCH. hashes_before written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
