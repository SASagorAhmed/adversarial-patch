#!/usr/bin/env python
"""Write duplicated Automatic-vs-Known comparison into THIS branch only."""
from __future__ import annotations

import sys
from pathlib import Path

BRANCH = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
PROJECT = Path(r"D:\project CS\DISPATCH-Defense")
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from common_lib import TESTING_ROOT, load_csv, sha256_file, write_csv, yn  # noqa: E402


def interpret(auto_row, known_row) -> tuple[str, str]:
    if auto_row["case_type"] == "control":
        return "control", "control"
    auto_rec = yn(auto_row["automatic_recovered"])
    kex_rec = yn(known_row["known_exact_recovered"])
    if (not auto_rec) and kex_rec:
        return "known_exact", "Case A: automatic fail / known succeed — localization likely bottleneck"
    if auto_rec and kex_rec:
        return "both", "Case B: both succeed"
    if (not auto_rec) and (not kex_rec):
        return "neither", "Case C: both fail — localization alone does not explain failure"
    return "automatic", "Automatic recovered but known-location did not — mixed / incidental automatic regeneration"


def main() -> int:
    auto_csv = TESTING_ROOT / "automatic" / "results" / "per_image_results.csv"
    known_csv = TESTING_ROOT / "known_location" / "results" / "per_image_results.csv"
    if not auto_csv.is_file() or not known_csv.is_file():
        print("STOP: both branch per_image_results.csv must exist before comparison")
        return 2
    auto_rows = {r["image_id"]: r for r in load_csv(auto_csv)}
    known_rows = {r["image_id"]: r for r in load_csv(known_csv)}
    if set(auto_rows) != set(known_rows):
        print("STOP: branch image IDs differ")
        return 3

    hashes = []
    compare = []
    for iid in auto_rows:
        a, k = auto_rows[iid], known_rows[iid]
        fn = a["filename"]
        best, interp = interpret(a, k)
        compare.append({
            "image_id": iid,
            "filename": fn,
            "target_class": a["target_class"],
            "case_type": a["case_type"],
            "patch_size": a["patch_size"],
            "automatic_mask_iou": a["automatic_mask_iou"],
            "automatic_detected": a["restored_detected"],
            "automatic_confidence": a["restored_confidence"],
            "automatic_iou": a["restored_iou"],
            "automatic_recovered": a["automatic_recovered"],
            "known_exact_detected": k["known_exact_detected"],
            "known_exact_confidence": k["known_exact_confidence"],
            "known_exact_iou": k["known_exact_iou"],
            "known_exact_recovered": k["known_exact_recovered"],
            "known_expand4_detected": k["known_expand4_detected"],
            "known_expand4_confidence": k["known_expand4_confidence"],
            "known_expand4_iou": k["known_expand4_iou"],
            "known_expand4_recovered": k["known_expand4_recovered"],
            "best_detection_branch": best,
            "interpretation": interp,
        })
        ac = TESTING_ROOT / "automatic" / "source_clean_images" / fn
        kc = TESTING_ROOT / "known_location" / "source_clean_images" / fn
        aa = TESTING_ROOT / "automatic" / "source_attacked_images" / fn
        ka = TESTING_ROOT / "known_location" / "source_attacked_images" / fn
        hashes.append({
            "filename": fn,
            "clean_sha256_automatic": sha256_file(ac),
            "clean_sha256_known": sha256_file(kc),
            "clean_match": sha256_file(ac) == sha256_file(kc),
            "attacked_sha256_automatic": sha256_file(aa),
            "attacked_sha256_known": sha256_file(ka),
            "attacked_match": sha256_file(aa) == sha256_file(ka),
        })

    out_dir = BRANCH / "results" / "comparison"
    write_csv(out_dir / "automatic_vs_known.csv", compare)
    write_csv(out_dir / "source_hash_validation.csv", hashes)

    succ = [r for r in compare if r["case_type"] == "attack_success"]
    auto_n = sum(1 for r in succ if yn(r["automatic_recovered"]))
    kex_n = sum(1 for r in succ if yn(r["known_exact_recovered"]))
    k4_n = sum(1 for r in succ if yn(r["known_expand4_recovered"]))
    hash_ok = all(h["clean_match"] and h["attacked_match"] for h in hashes)

    if kex_n > auto_n:
        improved, bottleneck = "YES", "MIXED (known recovered additional cases; see per-image interpretations)"
    elif kex_n < auto_n:
        improved, bottleneck = "NO", "MIXED (automatic recovered cases exact-mask LDM missed)"
    else:
        improved, bottleneck = "MIXED", "MIXED — same count, different images possible"

    txt = f"""TESTING MITIGATION 01 — AUTOMATIC vs KNOWN (copied into this branch)
====================================================================

This file is duplicated inside BOTH branches. There is no shared results folder.

Automatic recovery: {auto_n} / {len(succ)}   (diagnostic subset — NOT DISPATCH DRR)
Known Exact recovery: {kex_n} / {len(succ)}
Known +4px recovery: {k4_n} / {len(succ)}

Known location improved over Automatic: {improved}
Evidence-supported primary bottleneck: {bottleneck}

Source copies byte-equivalent across branches: {'YES' if hash_ok else 'NO'}

Per image:
""" + "\n".join(
        f"  {r['filename']}  auto_rec={r['automatic_recovered']}  exact_rec={r['known_exact_recovered']}  "
        f"+4_rec={r['known_expand4_recovered']}  mask_iou={r['automatic_mask_iou']}  {r['interpretation']}"
        for r in compare
    ) + """

manual_visual_review_required = true
Original filenames preserved.
Shared active source/result dependency: NO
"""
    (out_dir / "summary.txt").write_text(txt, encoding="utf-8")
    print(txt)
    print(f"Wrote comparison into {out_dir}")
    return 0 if hash_ok else 4


if __name__ == "__main__":
    raise SystemExit(main())
