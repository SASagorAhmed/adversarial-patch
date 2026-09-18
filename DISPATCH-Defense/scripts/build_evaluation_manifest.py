#!/usr/bin/env python
"""Build independent evaluation manifests from frozen attack CSVs (READ-ONLY).

Does NOT use mitigation_01/02 selections.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.paths_config import EXTERNAL_ATTACK_ROOT, PROJECT_ROOT, assert_under_project_root


ATTACK_IDS = [f"attack_{i:02d}" for i in range(1, 6)]


def _find_attack_csv(attack_id: str) -> Path:
    """Locate a per-image results CSV under a frozen attack folder (read-only)."""
    attack_dir = EXTERNAL_ATTACK_ROOT / "attacks" / attack_id
    candidates = [
        attack_dir / "results" / "per_image_results.csv",
        attack_dir / "results" / "per_image.csv",
        attack_dir / "per_image_results.csv",
        attack_dir / "results.csv",
    ]
    # Also search shallowly
    for c in candidates:
        if c.is_file():
            return c
    if attack_dir.is_dir():
        for p in attack_dir.rglob("*per_image*.csv"):
            return p
    raise FileNotFoundError(
        f"No per-image CSV found for {attack_id} under {attack_dir}. "
        "Manifest builder cannot invent labels."
    )


def _row_bool(row: dict, *keys: str) -> bool | None:
    for k in keys:
        if k in row and row[k] not in (None, ""):
            v = str(row[k]).strip().lower()
            if v in {"1", "true", "yes", "y"}:
                return True
            if v in {"0", "false", "no", "n"}:
                return False
    return None


def classify_subset(row: dict) -> str | None:
    """Return attack_success | control | None based on clean/attacked detection."""
    clean = _row_bool(row, "clean_detected", "clean_target_detected", "detected_clean")
    attacked = _row_bool(row, "attacked_detected", "attack_detected", "detected_attacked")
    if clean is None or attacked is None:
        return None
    if clean and (not attacked):
        return "attack_success"
    if clean and attacked:
        return "control"
    return None


def build_manifest(attack: str, out_csv: Path) -> int:
    attacks = ATTACK_IDS if attack == "all" else [attack]
    rows_out = []
    for aid in attacks:
        csv_path = _find_attack_csv(aid)
        with csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                subset = classify_subset(row)
                if subset is None:
                    continue
                image_id = (
                    row.get("image_id")
                    or row.get("coco_id")
                    or row.get("id")
                    or ""
                )
                rows_out.append(
                    {
                        "source_attack": aid,
                        "image_id": str(image_id).zfill(12) if str(image_id).isdigit() else image_id,
                        "filename": row.get("filename") or row.get("image_filename") or "",
                        "target_class": row.get("target_class") or row.get("class") or "",
                        "subset_type": subset,
                        "clean_detected": _row_bool(row, "clean_detected", "clean_target_detected"),
                        "attacked_detected": _row_bool(row, "attacked_detected", "attack_detected"),
                        "source_csv": str(csv_path),
                        "source_attack_dir": str(EXTERNAL_ATTACK_ROOT / "attacks" / aid),
                    }
                )

    out_csv = assert_under_project_root(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "source_attack",
        "image_id",
        "filename",
        "target_class",
        "subset_type",
        "clean_detected",
        "attacked_detected",
        "source_csv",
        "source_attack_dir",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows_out)
    return len(rows_out)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--attack", required=True, help="attack_01..attack_05 or all")
    p.add_argument(
        "--out",
        default=str(PROJECT_ROOT / "dataset" / "manifests" / "evaluation_manifest.csv"),
    )
    args = p.parse_args()
    if args.attack not in ATTACK_IDS + ["all"]:
        print(f"Invalid --attack {args.attack}")
        return 2
    n = build_manifest(args.attack, Path(args.out))
    print(f"Wrote {n} rows -> {args.out}")
    print("NOTE: Do not start a full experiment until explicitly approved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
