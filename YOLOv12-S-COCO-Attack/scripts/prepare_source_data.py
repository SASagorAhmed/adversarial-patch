"""Physically copy Hyper-YOLO attack clean/patched images + metadata (no symlinks)."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

from PIL import Image

PROJECT = Path(r"D:\project CS\YOLOv12-S-COCO-Attack").resolve()
SOURCE_ROOT = Path(r"D:\project CS\Adversarial-Patch-Experiment\attacks").resolve()
ATTACK_IDS = [f"attack_0{i}" for i in range(1, 6)]
EXPECTED_COUNTS = {
    "attack_01": 200,
    "attack_02": 250,
    "attack_03": 300,
    "attack_04": 350,
    "attack_05": 400,
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    dest_root = PROJECT / "source_data"
    ver_dir = PROJECT / "results" / "source_verification"
    ver_dir.mkdir(parents=True, exist_ok=True)
    before: dict[str, dict] = {}
    mismatches: list[dict] = []
    count_report: dict[str, dict] = {}

    for aid in ATTACK_IDS:
        src_base = SOURCE_ROOT / aid
        clean_src = src_base / "clean_images"
        patch_src = src_base / "patched_images"
        meta_src = src_base / "results" / "selected_images.csv"
        results_src = src_base / "results" / "attack_results.csv"
        summary_src = src_base / "results" / "attack_summary.txt"

        with meta_src.open(encoding="utf-8", newline="") as f:
            selected = list(csv.DictReader(f))

        dest_clean = dest_root / aid / "clean_images"
        dest_patch = dest_root / aid / "patched_images"
        dest_meta = dest_root / aid / "metadata"
        for d in (dest_clean, dest_patch, dest_meta):
            d.mkdir(parents=True, exist_ok=True)

        pair_rows = []
        for row in selected:
            fn = row["filename"]
            cpath = clean_src / fn
            ppath = patch_src / fn
            status = "verified"
            reason = ""
            clean_wh = ""
            patch_wh = ""

            if not cpath.is_file() or not ppath.is_file():
                status = "unresolved"
                reason = "missing_file"
                mismatches.append({"attack_id": aid, "filename": fn, "reason": reason})
            else:
                with Image.open(cpath) as cim, Image.open(ppath) as pim:
                    clean_wh = f"{cim.size[0]}x{cim.size[1]}"
                    patch_wh = f"{pim.size[0]}x{pim.size[1]}"
                    if cim.size != pim.size:
                        status = "unresolved"
                        reason = f"dimension_mismatch clean={clean_wh} patched={patch_wh}"
                        mismatches.append({"attack_id": aid, "filename": fn, "reason": reason})
                if status == "verified":
                    before[str(cpath)] = {
                        "sha256": sha256_file(cpath),
                        "size_bytes": cpath.stat().st_size,
                        "role": "clean",
                        "attack_id": aid,
                    }
                    before[str(ppath)] = {
                        "sha256": sha256_file(ppath),
                        "size_bytes": ppath.stat().st_size,
                        "role": "patched",
                        "attack_id": aid,
                    }
                    shutil.copy2(cpath, dest_clean / fn)
                    shutil.copy2(ppath, dest_patch / fn)
                    if (dest_clean / fn).is_symlink() or (dest_patch / fn).is_symlink():
                        raise RuntimeError(f"Symlink detected: {aid}/{fn}")

            pair_rows.append(
                {
                    "attack_id": aid,
                    "image_id": row["image_id"],
                    "clean_filename": fn,
                    "patched_filename": fn if status == "verified" else "",
                    "pair_verified": "yes" if status == "verified" else "no",
                    "target_class": row["target_class"],
                    "clean_wh": clean_wh,
                    "patched_wh": patch_wh,
                    "status": status,
                    "reason": reason,
                }
            )

        shutil.copy2(meta_src, dest_meta / "selected_images.csv")
        if results_src.is_file():
            shutil.copy2(results_src, dest_meta / "hyper_yolo_attack_results.csv")
            before[str(results_src)] = {
                "sha256": sha256_file(results_src),
                "size_bytes": results_src.stat().st_size,
                "role": "metadata",
                "attack_id": aid,
            }
        if summary_src.is_file():
            shutil.copy2(summary_src, dest_meta / "hyper_yolo_attack_summary.txt")
            before[str(summary_src)] = {
                "sha256": sha256_file(summary_src),
                "size_bytes": summary_src.stat().st_size,
                "role": "metadata",
                "attack_id": aid,
            }
        before[str(meta_src)] = {
            "sha256": sha256_file(meta_src),
            "size_bytes": meta_src.stat().st_size,
            "role": "metadata",
            "attack_id": aid,
        }

        out_csv = ver_dir / f"{aid}_pair_verification.csv"
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "attack_id",
                    "image_id",
                    "clean_filename",
                    "patched_filename",
                    "pair_verified",
                    "target_class",
                    "clean_wh",
                    "patched_wh",
                    "status",
                    "reason",
                ],
            )
            w.writeheader()
            w.writerows(pair_rows)

        n_clean = len(list(dest_clean.glob("*.jpg"))) + len(list(dest_clean.glob("*.png")))
        n_patch = len(list(dest_patch.glob("*.jpg"))) + len(list(dest_patch.glob("*.png")))
        n_ok = sum(1 for r in pair_rows if r["pair_verified"] == "yes")
        count_report[aid] = {
            "selected_rows": len(selected),
            "copied_pairs": n_ok,
            "clean_images_on_disk": n_clean,
            "patched_images_on_disk": n_patch,
            "expected": EXPECTED_COUNTS[aid],
            "unresolved": len(pair_rows) - n_ok,
        }
        print(
            f"{aid}: selected={len(selected)} copied={n_ok} "
            f"clean_disk={n_clean} patched_disk={n_patch} expected={EXPECTED_COUNTS[aid]} "
            f"unresolved={len(pair_rows)-n_ok}"
        )

    (PROJECT / "results" / "source_integrity_hashes_before.json").write_text(
        json.dumps(before, indent=2), encoding="utf-8"
    )
    (PROJECT / "results" / "source_count_report.json").write_text(
        json.dumps(count_report, indent=2), encoding="utf-8"
    )
    (ver_dir / "mismatch_report.json").write_text(json.dumps(mismatches, indent=2), encoding="utf-8")
    print(f"Hashed {len(before)} source files")
    if mismatches:
        print(f"MISMATCHES: {len(mismatches)} — see results/source_verification/mismatch_report.json")
        print("STOP before inference.")
        return 2
    print("All pairs verified. Safe to run inference.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
