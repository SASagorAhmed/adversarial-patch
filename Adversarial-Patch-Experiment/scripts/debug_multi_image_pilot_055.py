#!/usr/bin/env python3
"""6-image mitigation pilot: neutralization ON + strength 0.55.

Outputs ONLY under mitigation_debug/multi_image_pilot_055/.
Does not modify attacks or create final mitigation_01 results.
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from mitigation_config import (  # noqa: E402
    CONTEXT_CROP_FACTOR,
    DIFFUSION_RESOLUTION,
    FEATHER_PIXELS,
    GUIDANCE_SCALE,
    MINIMUM_CONTEXT_SIDE,
    NEGATIVE_PROMPT,
    NUM_INFERENCE_STEPS,
    POSITIVE_PROMPT,
    RESTORE_WORKER_SCRIPT,
    RESTORATION_SEED,
    SD_MODEL_PATH,
    SD_PYTHON,
    STRENGTH,
    attack_dir,
    get_mitigation_spec,
)
from pipeline_core import (  # noqa: E402
    evaluate_target_match,
    load_yolo,
    parse_label_file,
    run_yolo_inference,
)
from select_mitigation_subset import select_mitigation_subset  # noqa: E402

PILOT_ROOT = SCRIPT_DIR.parent / "mitigation_debug" / "multi_image_pilot_055"
STRENGTH_PILOT = 0.55
assert abs(STRENGTH_PILOT - STRENGTH) < 1e-9


def select_pilot_rows(subset_rows: list) -> list[tuple[object, str]]:
    """Deterministic 3 success + 3 control with size/class diversity."""
    successes = sorted(
        [r for r in subset_rows if r.subset_role == "success"],
        key=lambda r: (r.reconstructed_patch_size, r.filename),
    )
    controls = sorted(
        [r for r in subset_rows if r.subset_role == "control"],
        key=lambda r: (r.reconstructed_patch_size, r.filename),
    )
    if len(successes) < 3 or len(controls) < 3:
        raise RuntimeError("Need at least 3 success and 3 control rows")

    # Small / median / large by sorted size
    s_picks = [
        (successes[0], "success_small_patch_first_by_size_then_filename"),
        (
            successes[len(successes) // 2],
            "success_median_patch_by_sorted_size_then_filename",
        ),
        (successes[-1], "success_large_patch_last_by_size_then_filename"),
    ]
    c_picks = [
        (controls[0], "control_small_patch_first_by_size_then_filename"),
        (
            controls[len(controls) // 2],
            "control_median_patch_by_sorted_size_then_filename",
        ),
        (controls[-1], "control_large_patch_last_by_size_then_filename"),
    ]

    chosen = s_picks + c_picks
    classes = {row.target_class for row, _ in chosen}
    # If person or car missing, swap the median of the denser role with first
    # opposite-class unused row of same role (deterministic).
    if classes != {"person", "car"}:
        missing = ({"person", "car"} - classes).pop()
        for role, pool, picks, mid_idx in (
            ("success", successes, s_picks, 1),
            ("control", controls, c_picks, 1),
        ):
            used = {r.filename for r, _ in picks}
            replacement = next(
                (r for r in pool if r.target_class == missing and r.filename not in used),
                None,
            )
            if replacement is not None and picks[mid_idx][0].target_class != missing:
                picks[mid_idx] = (
                    replacement,
                    f"{role}_median_swapped_for_{missing}_class_coverage",
                )
        chosen = s_picks + c_picks

    return chosen


def restore_one(row, pilot_root: Path) -> None:
    attacked_src = attack_dir(row.source_attack_id) / "patched_images" / row.filename
    if not attacked_src.is_file():
        raise FileNotFoundError(attacked_src)

    stem = Path(row.filename).stem
    mask_path = pilot_root / "masks" / f"{stem}_patch_mask.png"
    restored_path = pilot_root / "restored_images" / row.filename
    neut_path = pilot_root / "neutralized_inputs" / f"{stem}_512.png"
    raw_path = pilot_root / "raw_img2img_outputs" / f"{stem}_raw512.png"
    meta_path = pilot_root / "restore_meta" / f"{stem}.json"

    command = [
        str(SD_PYTHON),
        str(RESTORE_WORKER_SCRIPT),
        "--attacked-image",
        str(attacked_src),
        "--output-restored",
        str(restored_path),
        "--output-mask",
        str(mask_path),
        "--model-path",
        str(SD_MODEL_PATH),
        "--x1",
        str(row.reconstructed_x1),
        "--y1",
        str(row.reconstructed_y1),
        "--x2",
        str(row.reconstructed_x2),
        "--y2",
        str(row.reconstructed_y2),
        "--prompt",
        POSITIVE_PROMPT,
        "--negative-prompt",
        NEGATIVE_PROMPT,
        "--seed",
        str(RESTORATION_SEED),
        "--steps",
        str(NUM_INFERENCE_STEPS),
        "--guidance-scale",
        str(GUIDANCE_SCALE),
        "--strength",
        str(STRENGTH_PILOT),
        "--resolution",
        str(DIFFUSION_RESOLUTION),
        "--context-factor",
        str(CONTEXT_CROP_FACTOR),
        "--min-context-side",
        str(MINIMUM_CONTEXT_SIDE),
        "--feather-pixels",
        str(FEATHER_PIXELS),
        "--neutralize",
        "--output-neutralized-512",
        str(neut_path),
        "--output-raw-img2img",
        str(raw_path),
        "--metadata-json",
        str(meta_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "unknown").strip()
        raise RuntimeError(f"Restore failed for {row.filename}:\n{detail}")
    if not restored_path.is_file():
        raise RuntimeError(f"Missing restored image: {restored_path}")


def draw_verification(
    clean_path: Path,
    attacked_path: Path,
    neut_path: Path,
    restored_path: Path,
    out_path: Path,
    row,
    mitigated_detected: str,
    mitigated_confidence: float,
    mitigated_iou: float,
    outcome: str,
) -> None:
    clean = cv2.imread(str(clean_path))
    attacked = cv2.imread(str(attacked_path))
    neut = cv2.imread(str(neut_path))
    restored = cv2.imread(str(restored_path))
    if any(img is None for img in (clean, attacked, neut, restored)):
        raise RuntimeError(f"Failed to read verification inputs for {row.filename}")

    h = max(clean.shape[0], attacked.shape[0], restored.shape[0], 360)
    w = max(clean.shape[1], attacked.shape[1], restored.shape[1], 360)

    def prep(img: np.ndarray, title: str, draw_patch: bool) -> np.ndarray:
        panel = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA).copy()
        if draw_patch:
            # Map patch from original coords if sizes match restored/attacked
            if panel.shape[0] == restored.shape[0] and panel.shape[1] == restored.shape[1]:
                sx = sy = 1.0
                # after resize above, use scaled coords
            sx = w / float(row.image_width)
            sy = h / float(row.image_height)
            cv2.rectangle(
                panel,
                (int(row.reconstructed_x1 * sx), int(row.reconstructed_y1 * sy)),
                (int(row.reconstructed_x2 * sx), int(row.reconstructed_y2 * sy)),
                (0, 255, 255),
                2,
            )
            cv2.rectangle(
                panel,
                (int(row.bbox_x * sx), int(row.bbox_y * sy)),
                (int((row.bbox_x + row.bbox_width) * sx), int((row.bbox_y + row.bbox_height) * sy)),
                (0, 255, 0),
                2,
            )
        cv2.putText(panel, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3)
        cv2.putText(panel, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        return panel

    # Neutralized input is 512; still show with label (patch approx center)
    neut_panel = cv2.resize(neut, (w, h), interpolation=cv2.INTER_AREA).copy()
    cv2.putText(neut_panel, "NEUTRALIZED INPUT", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3)
    cv2.putText(neut_panel, "NEUTRALIZED INPUT", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

    panels = [
        prep(clean, "CLEAN", True),
        prep(attacked, "ATTACKED", True),
        neut_panel,
        prep(restored, "MITIGATED", True),
    ]
    gap = 8
    header_h = 70
    canvas = np.full((h + header_h, w * 4 + gap * 3, 3), 255, dtype=np.uint8)
    for i, panel in enumerate(panels):
        x0 = i * (w + gap)
        canvas[header_h : header_h + h, x0 : x0 + w] = panel

    header = (
        f"{row.filename} | {row.subset_role.upper()} | {row.target_class} | "
        f"patch={row.reconstructed_patch_size}px | {outcome}"
    )
    sub = (
        f"clean={row.clean_detected}/{row.clean_confidence:.3f}/{row.clean_iou:.3f}  "
        f"attacked={row.attacked_detected}/{row.attacked_confidence:.3f}/{row.attacked_iou:.3f}  "
        f"mitigated={mitigated_detected}/{mitigated_confidence:.3f}/{mitigated_iou:.3f}"
    )
    cv2.putText(canvas, header, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
    cv2.putText(canvas, sub, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


def visual_review_notes(attacked_path: Path, restored_path: Path, x1: int, y1: int, x2: int, y2: int) -> tuple[str, str]:
    """Heuristic sticker-removal / hallucination flags from pixel change."""
    attacked = cv2.imread(str(attacked_path))
    restored = cv2.imread(str(restored_path))
    if attacked is None or restored is None:
        return "UNKNOWN", "UNKNOWN"
    a = attacked[y1:y2, x1:x2].astype(np.float32)
    r = restored[y1:y2, x1:x2].astype(np.float32)
    mae = float(np.mean(np.abs(a - r)))
    # High contrast colorful stickers change a lot when removed
    sticker_removed = "YES" if mae >= 18.0 else "LIKELY_PARTIAL" if mae >= 8.0 else "NO"
    # Strong structured edges in restored patch vs smooth blur → possible hallucination
    gray = cv2.cvtColor(restored[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    edge_density = float(np.mean(edges > 0))
    hallucinated = "YES_POSSIBLE" if edge_density > 0.18 and mae >= 25 else "NO_OBVIOUS"
    return sticker_removed, hallucinated


def main() -> None:
    for relative in (
        "neutralized_inputs",
        "raw_img2img_outputs",
        "restored_images",
        "masks",
        "restore_meta",
        "yolo_predictions/images",
        "yolo_predictions/labels",
        "verification",
    ):
        (PILOT_ROOT / relative).mkdir(parents=True, exist_ok=True)

    spec = get_mitigation_spec("mitigation_01")
    subset_rows = select_mitigation_subset(spec)
    pilot = select_pilot_rows(subset_rows)

    # Save selection
    selected_path = PILOT_ROOT / "selected_pilot.csv"
    with selected_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "filename",
                "subset_role",
                "target_class",
                "patch_size_pixels",
                "patch_x1",
                "patch_y1",
                "patch_x2",
                "patch_y2",
                "selection_reason",
            ],
        )
        writer.writeheader()
        for row, reason in pilot:
            writer.writerow(
                {
                    "filename": row.filename,
                    "subset_role": row.subset_role,
                    "target_class": row.target_class,
                    "patch_size_pixels": row.reconstructed_patch_size,
                    "patch_x1": row.reconstructed_x1,
                    "patch_y1": row.reconstructed_y1,
                    "patch_x2": row.reconstructed_x2,
                    "patch_y2": row.reconstructed_y2,
                    "selection_reason": reason,
                }
            )

    print("Pilot selection:")
    for row, reason in pilot:
        print(
            f"  {row.filename} | {row.subset_role} | {row.target_class} | "
            f"patch={row.reconstructed_patch_size} | {reason}"
        )

    # Restore each image
    for index, (row, _reason) in enumerate(pilot, start=1):
        print(f"[{index}/6] Restoring {row.filename} ...")
        restore_one(row, PILOT_ROOT)

    # Hyper-YOLO on restored images
    print("Running Hyper-YOLO on pilot restored images...")
    model = load_yolo()
    restored_dir = PILOT_ROOT / "restored_images"
    pred_images = PILOT_ROOT / "yolo_predictions" / "images"
    pred_labels = PILOT_ROOT / "yolo_predictions" / "labels"
    run_yolo_inference(
        model,
        restored_dir,
        pred_images,
        pred_labels,
        PILOT_ROOT / "yolo_predictions",
    )

    result_rows = []
    for row, reason in pilot:
        label_path = pred_labels / f"{Path(row.filename).stem}.txt"
        detections = parse_label_file(label_path, row.image_width, row.image_height, model)
        match = evaluate_target_match(
            detections,
            row.target_class,
            (row.bbox_x, row.bbox_y, row.bbox_width, row.bbox_height),
        )
        mitigated_detected = str(match["detected"])
        mitigated_confidence = float(match["confidence"])
        mitigated_iou = float(match["iou"])

        if row.subset_role == "success":
            outcome = "RECOVERED" if mitigated_detected == "Yes" else "NOT RECOVERED"
        else:
            outcome = "PRESERVED" if mitigated_detected == "Yes" else "REGRESSED"

        attacked_src = attack_dir(row.source_attack_id) / "patched_images" / row.filename
        clean_src = attack_dir(row.source_attack_id) / "clean_images" / row.filename
        stem = Path(row.filename).stem
        sticker_removed, hallucinated = visual_review_notes(
            attacked_src,
            restored_dir / row.filename,
            row.reconstructed_x1,
            row.reconstructed_y1,
            row.reconstructed_x2,
            row.reconstructed_y2,
        )

        draw_verification(
            clean_src,
            attacked_src,
            PILOT_ROOT / "neutralized_inputs" / f"{stem}_512.png",
            restored_dir / row.filename,
            PILOT_ROOT / "verification" / f"verify_{stem}.jpg",
            row,
            mitigated_detected,
            mitigated_confidence,
            mitigated_iou,
            outcome,
        )

        result_rows.append(
            {
                "filename": row.filename,
                "subset_role": row.subset_role,
                "target_class": row.target_class,
                "patch_size_pixels": row.reconstructed_patch_size,
                "patch_x1": row.reconstructed_x1,
                "patch_y1": row.reconstructed_y1,
                "patch_x2": row.reconstructed_x2,
                "patch_y2": row.reconstructed_y2,
                "selection_reason": reason,
                "clean_detected": row.clean_detected,
                "attacked_detected": row.attacked_detected,
                "mitigated_detected": mitigated_detected,
                "clean_confidence": row.clean_confidence,
                "attacked_confidence": row.attacked_confidence,
                "mitigated_confidence": mitigated_confidence,
                "clean_iou": row.clean_iou,
                "attacked_iou": row.attacked_iou,
                "mitigated_iou": mitigated_iou,
                "outcome": outcome,
                "heuristic_sticker_removed": sticker_removed,
                "heuristic_new_structure": hallucinated,
                "strength": STRENGTH_PILOT,
                "neutralization": "ON",
                "restoration_seed": RESTORATION_SEED,
            }
        )

    results_path = PILOT_ROOT / "pilot_results.csv"
    with results_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result_rows[0].keys()))
        writer.writeheader()
        writer.writerows(result_rows)

    successes = [r for r in result_rows if r["subset_role"] == "success"]
    controls = [r for r in result_rows if r["subset_role"] == "control"]
    recovered = sum(1 for r in successes if r["outcome"] == "RECOVERED")
    preserved = sum(1 for r in controls if r["outcome"] == "PRESERVED")
    person = [r for r in result_rows if r["target_class"] == "person"]
    car = [r for r in result_rows if r["target_class"] == "car"]

    lines = [
        "MULTI-IMAGE 0.55 PILOT SUMMARY",
        "=" * 72,
        "Method: Simplified known-location diffusion-based restoration baseline",
        "Config: neutralization=ON, strength=0.55, steps=30, guidance=7.5, res=512",
        f"context_crop_factor={CONTEXT_CROP_FACTOR}, min_context={MINIMUM_CONTEXT_SIDE}, feather={FEATHER_PIXELS}",
        f"restoration_seed={RESTORATION_SEED}",
        "NOTE: 6-image configuration sanity check only. Not statistical defense performance.",
        "",
        f"success recovered: {recovered} / 3",
        f"controls preserved: {preserved} / 3",
        "",
        "Per-image:",
    ]
    for r in result_rows:
        lines.append(
            f"- {r['filename']} | {r['subset_role']} | {r['target_class']} | "
            f"patch={r['patch_size_pixels']} | {r['outcome']} | "
            f"sticker_removed={r['heuristic_sticker_removed']} | "
            f"new_structure={r['heuristic_new_structure']} | "
            f"mitigated={r['mitigated_detected']} "
            f"({r['mitigated_confidence']:.3f}/{r['mitigated_iou']:.3f})"
        )
    lines.extend(
        [
            "",
            f"person cases: {len(person)} | outcomes: "
            + ", ".join(f"{r['filename']}={r['outcome']}" for r in person),
            f"car cases: {len(car)} | outcomes: "
            + ", ".join(f"{r['filename']}={r['outcome']}" for r in car),
            "",
            f"output_root: {PILOT_ROOT}",
        ]
    )
    (PILOT_ROOT / "pilot_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
