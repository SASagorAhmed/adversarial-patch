#!/usr/bin/env python
"""Compare current Automatic + Known Location (read-only) vs isolated reference."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(r"D:\project CS\DISPATCH-Paper-Faithful").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.reference_config import CURRENT_AUTO_RESULTS, CURRENT_KNOWN_RESULTS, SELECTED_IDS, assert_isolated


def load(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return {r["image_id"]: r for r in csv.DictReader(f)}


def fnum(v):
    if v is None or str(v).strip() in {"", "None", "none"}:
        return None
    try:
        return float(v)
    except Exception:
        return None


def fbool(v) -> bool:
    return str(v).strip().lower() in {"true", "1", "yes"}


def fmt_det(det, conf, iou) -> str:
    if not fbool(det):
        return "detected=False"
    return f"detected=True conf={fnum(conf)} iou={fnum(iou)}"


def main() -> int:
    cur = load(CURRENT_AUTO_RESULTS)
    kn = load(CURRENT_KNOWN_RESULTS)
    ref_csv = ROOT / "diagnostic" / "results" / "per_image_results.csv"
    ref = load(ref_csv)
    rows = []
    lines = [
        "CURRENT AUTOMATIC vs PAPER-FAITHFUL REFERENCE vs KNOWN LOCATION",
        "Current/known rows READ-ONLY from testing_mitigations/mitigation_02.",
        "This is a diagnostic subset — NOT paper mAP and NOT DISPATCH DRR.",
        "",
    ]
    for iid in SELECTED_IDS:
        c, k, r = cur[iid], kn[iid], ref[iid]
        rec = {
            "image_id": iid,
            "filename": r["filename"],
            "target_class": r["target_class"],
            "case_type": r["case_type"],
            "true_patch_area": r["true_patch_area"],
            "current_auto_mask_iou": c.get("automatic_mask_iou"),
            "reference_auto_mask_iou": r.get("reference_mask_iou"),
            "current_precision": c.get("automatic_mask_precision"),
            "reference_precision": r.get("reference_precision"),
            "current_recall": c.get("automatic_mask_recall"),
            "reference_recall": r.get("reference_recall"),
            "current_predicted_area": c.get("predicted_mask_area"),
            "reference_predicted_area": r.get("reference_predicted_area"),
            "area_ratio_current": (
                float(c["predicted_mask_area"]) / float(c["true_patch_area"])
                if float(c.get("true_patch_area") or 0)
                else None
            ),
            "area_ratio_reference": r.get("area_ratio_reference"),
            "attacked_detected": c.get("attacked_detected"),
            "current_auto_detected": c.get("restored_detected"),
            "current_auto_conf": c.get("restored_confidence"),
            "current_auto_iou": c.get("restored_iou"),
            "reference_auto_detected": r.get("reference_detected"),
            "reference_auto_conf": r.get("reference_confidence"),
            "reference_auto_iou": r.get("reference_iou"),
            "known_location_detected": k.get("known_exact_detected"),
            "known_location_conf": k.get("known_exact_confidence"),
            "known_location_iou": k.get("known_exact_iou"),
            "known_expand4_detected": k.get("known_expand4_detected"),
            "known_expand4_conf": k.get("known_expand4_confidence"),
            "known_expand4_iou": k.get("known_expand4_iou"),
        }
        rows.append(rec)
        lines.append(f"=== {r['filename']}  {r['target_class']}  {r['case_type']} ===")
        lines.append(
            f"CURRENT AUTOMATIC: mask IoU={c.get('automatic_mask_iou')} P={c.get('automatic_mask_precision')} "
            f"R={c.get('automatic_mask_recall')} area={c.get('predicted_mask_area')}/{c.get('true_patch_area')} "
            f"{fmt_det(c.get('restored_detected'), c.get('restored_confidence'), c.get('restored_iou'))}"
        )
        lines.append(
            f"PAPER-FAITHFUL REFERENCE: mask IoU={r.get('reference_mask_iou')} P={r.get('reference_precision')} "
            f"R={r.get('reference_recall')} area={r.get('reference_predicted_area')}/{r.get('true_patch_area')} "
            f"{fmt_det(r.get('reference_detected'), r.get('reference_confidence'), r.get('reference_iou'))}"
        )
        lines.append(
            f"KNOWN LOCATION exact: {fmt_det(k.get('known_exact_detected'), k.get('known_exact_confidence'), k.get('known_exact_iou'))}  "
            f"+4px: {fmt_det(k.get('known_expand4_detected'), k.get('known_expand4_confidence'), k.get('known_expand4_iou'))}"
        )
        lines.append("")

    out_csv = assert_isolated(ROOT / "diagnostic" / "results" / "comparison_current_vs_reference.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    txt = assert_isolated(ROOT / "diagnostic" / "results" / "comparison_summary.txt")
    txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
