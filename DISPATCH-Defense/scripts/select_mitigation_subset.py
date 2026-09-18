#!/usr/bin/env python
"""Independently select DISPATCH mitigation subsets from frozen attack CSVs.

Does NOT read old mitigation_01/02 selections.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.dispatch_config import DISPATCH_SEED
from config.paths_config import EXTERNAL_ATTACK_ROOT, assert_under_project_root


def _yn(v) -> bool:
    return str(v).strip().lower() in {"yes", "true", "1", "y"}


def load_attack_rows(attack_id: str) -> tuple[list[dict], dict[str, dict]]:
    attack_dir = EXTERNAL_ATTACK_ROOT / "attacks" / attack_id
    results_csv = attack_dir / "results" / "attack_results.csv"
    selected_csv = attack_dir / "results" / "selected_images.csv"
    if not results_csv.is_file():
        raise FileNotFoundError(results_csv)
    if not selected_csv.is_file():
        raise FileNotFoundError(selected_csv)
    with results_csv.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    meta = {}
    with selected_csv.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            meta[str(r["image_id"])] = r
    return rows, meta


def select_subset(
    attack_id: str,
    n_success: int,
    n_control: int,
    seed: int = DISPATCH_SEED,
) -> list[dict]:
    rows, meta = load_attack_rows(attack_id)
    success = [
        r
        for r in rows
        if _yn(r.get("clean_detected")) and (not _yn(r.get("attacked_detected")))
    ]
    control = [
        r
        for r in rows
        if _yn(r.get("clean_detected")) and _yn(r.get("attacked_detected"))
    ]
    if len(success) < n_success:
        raise RuntimeError(f"Need {n_success} success cases, found {len(success)}")
    if len(control) < n_control:
        raise RuntimeError(f"Need {n_control} controls, found {len(control)}")

    rng = random.Random(seed)
    # Prefer deterministic order then sample: all success if n_success == len, else sample
    success_sorted = sorted(success, key=lambda r: int(r["image_id"]))
    if n_success == len(success_sorted):
        chosen_success = success_sorted
    else:
        chosen_success = sorted(rng.sample(success_sorted, n_success), key=lambda r: int(r["image_id"]))

    # Stratified controls by class toward balance
    by_cls: dict[str, list] = {}
    for r in control:
        by_cls.setdefault(r["target_class"], []).append(r)
    for k in by_cls:
        by_cls[k] = sorted(by_cls[k], key=lambda r: int(r["image_id"]))
        rng.shuffle(by_cls[k])

    classes = sorted(by_cls.keys())
    chosen_control: list[dict] = []
    # Round-robin from classes for balance
    idxs = {c: 0 for c in classes}
    while len(chosen_control) < n_control:
        progressed = False
        for c in classes:
            if len(chosen_control) >= n_control:
                break
            i = idxs[c]
            if i < len(by_cls[c]):
                chosen_control.append(by_cls[c][i])
                idxs[c] = i + 1
                progressed = True
        if not progressed:
            break
    if len(chosen_control) < n_control:
        raise RuntimeError("Could not fill control quota")

    out = []
    for r in chosen_success + chosen_control:
        iid = str(r["image_id"])
        m = meta.get(iid, {})
        subset = "attack_success" if r in chosen_success or (
            _yn(r.get("clean_detected")) and not _yn(r.get("attacked_detected"))
        ) else "control"
        # safer subset label
        subset = (
            "attack_success"
            if _yn(r.get("clean_detected")) and not _yn(r.get("attacked_detected"))
            else "control"
        )
        cx = float(r.get("patch_center_x") or 0)
        cy = float(r.get("patch_center_y") or 0)
        psz = float(r.get("patch_size_pixels") or 0)
        half = psz / 2.0
        out.append(
            {
                "source_attack": attack_id,
                "image_id": iid.zfill(12) if iid.isdigit() else iid,
                "image_id_raw": iid,
                "filename": r["filename"],
                "target_class": r["target_class"],
                "subset_type": subset,
                "clean_detected": _yn(r["clean_detected"]),
                "attacked_detected": _yn(r["attacked_detected"]),
                "clean_confidence": float(r.get("clean_confidence") or 0),
                "attacked_confidence": float(r.get("attacked_confidence") or 0)
                if r.get("attacked_confidence") not in (None, "")
                else None,
                "clean_iou": float(r.get("clean_iou") or 0),
                "attacked_iou": float(r.get("attacked_iou") or 0)
                if r.get("attacked_iou") not in (None, "")
                else None,
                "bbox_x": float(m.get("bbox_x") or 0),
                "bbox_y": float(m.get("bbox_y") or 0),
                "bbox_width": float(m.get("bbox_width") or 0),
                "bbox_height": float(m.get("bbox_height") or 0),
                "image_width": int(float(m.get("image_width") or 0)),
                "image_height": int(float(m.get("image_height") or 0)),
                "patch_size_pixels": psz,
                "patch_center_x": cx,
                "patch_center_y": cy,
                "true_patch_x1": cx - half,
                "true_patch_y1": cy - half,
                "true_patch_x2": cx + half,
                "true_patch_y2": cy + half,
                "source_attacked_path": str(
                    EXTERNAL_ATTACK_ROOT / "attacks" / attack_id / "patched_images" / r["filename"]
                ),
                "source_clean_path": str(
                    EXTERNAL_ATTACK_ROOT / "attacks" / attack_id / "clean_images" / r["filename"]
                ),
                "selection_seed": seed,
            }
        )
    # Mark subset correctly using membership sets
    succ_ids = {str(r["image_id"]) for r in chosen_success}
    for row in out:
        row["subset_type"] = "attack_success" if row["image_id_raw"] in succ_ids else "control"
    out.sort(key=lambda r: (0 if r["subset_type"] == "attack_success" else 1, int(r["image_id_raw"])))
    return out


def write_selected_csv(rows: list[dict], path: Path) -> None:
    path = assert_under_project_root(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--attack", default="attack_02")
    p.add_argument("--n-success", type=int, default=15)
    p.add_argument("--n-control", type=int, default=15)
    p.add_argument("--seed", type=int, default=DISPATCH_SEED)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    rows = select_subset(args.attack, args.n_success, args.n_control, args.seed)
    write_selected_csv(rows, Path(args.out))
    print(f"Wrote {len(rows)} rows -> {args.out}")
    print(
        "success",
        sum(1 for r in rows if r["subset_type"] == "attack_success"),
        "control",
        sum(1 for r in rows if r["subset_type"] == "control"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
