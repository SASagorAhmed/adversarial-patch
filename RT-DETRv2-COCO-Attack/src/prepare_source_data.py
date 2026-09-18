"""Physically copy attack clean/patched images + metadata into this project."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

PROJECT = Path(r"D:\project CS\RT-DETRv2-COCO-Attack").resolve()
SOURCE_ROOT = Path(r"D:\project CS\Adversarial-Patch-Experiment\attacks").resolve()
ATTACK_IDS = [f"attack_0{i}" for i in range(1, 6)]


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
            ok = cpath.is_file() and ppath.is_file()
            if ok:
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
                    "patched_filename": fn if ok else "",
                    "pair_verified": "yes" if ok else "no",
                    "target_class": row["target_class"],
                    "status": "verified" if ok else "unresolved",
                }
            )

        shutil.copy2(meta_src, dest_meta / "selected_images.csv")
        shutil.copy2(results_src, dest_meta / "hyper_yolo_attack_results.csv")
        shutil.copy2(summary_src, dest_meta / "hyper_yolo_attack_summary.txt")
        for mp in (meta_src, results_src, summary_src):
            before[str(mp)] = {
                "sha256": sha256_file(mp),
                "size_bytes": mp.stat().st_size,
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
                    "status",
                ],
            )
            w.writeheader()
            w.writerows(pair_rows)
        n_ok = sum(1 for r in pair_rows if r["pair_verified"] == "yes")
        print(f"{aid}: copied pairs={n_ok} unresolved={len(pair_rows)-n_ok}")

    (PROJECT / "results" / "source_integrity_hashes_before.json").write_text(
        json.dumps(before, indent=2), encoding="utf-8"
    )
    print(f"Hashed {len(before)} source files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
