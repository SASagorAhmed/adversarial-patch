#!/usr/bin/env python3
"""Mitigation-only pipeline for known-location diffusion restoration.

Attack folders remain READ-ONLY. Do not modify attack outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from mitigation_config import (  # noqa: E402
    CONTEXT_CROP_FACTOR,
    DIFFUSION_RESOLUTION,
    FEATHER_PIXELS,
    GUIDANCE_SCALE,
    HYPER_YOLO_PYTHON,
    METHOD_NAME,
    MINIMUM_CONTEXT_SIDE,
    MITIGATION_SPECS,
    MITIGATIONS_DIR,
    MODEL_PATH,
    NEGATIVE_PROMPT,
    NUM_INFERENCE_STEPS,
    PATCH_NEUTRALIZATION,
    POSITIVE_PROMPT,
    RESTORE_WORKER_SCRIPT,
    RESULTS_DIR,
    SD_MODEL_PATH,
    SD_PYTHON,
    STRENGTH,
    TARGET_MATCH_IOU_THRESHOLD,
    attack_dir,
    ensure_mitigation_structure,
    get_mitigation_spec,
    mitigation_dir,
    restoration_config_dict,
)
from mitigation_reporting import (  # noqa: E402
    build_mitigation_result_row,
    compute_mitigation_metrics,
    create_mitigation_overview,
    draw_mitigation_verification,
    update_global_mitigations_comparison,
    write_comparison_plots,
    write_mitigation_results_csv,
    write_mitigation_summary,
)
from select_mitigation_subset import (  # noqa: E402
    copy_source_images,
    load_attack_tables,
    save_source_attack_package,
    select_mitigation_subset,
    summarize_subset,
    validate_geometry_for_attack,
)


class MitigationError(Exception):
    """Raised when mitigation validation or execution fails."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run known-location diffusion mitigation (or validate only)."
    )
    parser.add_argument(
        "--mitigation-id",
        choices=sorted(MITIGATION_SPECS.keys()),
        default=None,
        help="Optional single mitigation. Default: validate/run all configured mitigations.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate inputs/subset/geometry/SD import without running SD or YOLO.",
    )
    return parser.parse_args()


def file_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def check_sd_img2img_import() -> tuple[bool, str]:
    if not SD_PYTHON.is_file():
        return False, f"SD Python missing: {SD_PYTHON}"
    code = (
        "from diffusers import StableDiffusionImg2ImgPipeline; "
        "print(StableDiffusionImg2ImgPipeline.__name__)"
    )
    result = subprocess.run(
        [str(SD_PYTHON), "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "import failed").strip()
        return False, detail
    return True, (result.stdout or "").strip()


def assert_attack_read_only_snapshots(attack_id: str) -> dict[str, str]:
    """Capture hashes of key attack artifacts for later comparison."""
    root = attack_dir(attack_id)
    targets = [
        root / "results" / "attack_results.csv",
        root / "results" / "selected_images.csv",
        root / "results" / "attack_summary.txt",
    ]
    snapshots: dict[str, str] = {}
    for path in targets:
        if not path.is_file():
            raise MitigationError(f"Missing required attack file: {path}")
        snapshots[str(path)] = file_sha1(path)
    return snapshots


def verify_snapshots_unchanged(snapshots: dict[str, str]) -> None:
    for path_str, expected in snapshots.items():
        current = file_sha1(Path(path_str))
        if current != expected:
            raise MitigationError(f"Attack file was modified (forbidden): {path_str}")


def validate_one(mitigation_id: str, *, create_output_dirs: bool = False) -> dict:
    spec = get_mitigation_spec(mitigation_id)
    source = attack_dir(spec.source_attack_id)
    attack_csv = source / "results" / "attack_results.csv"
    selected_csv = source / "results" / "selected_images.csv"
    patched_dir = source / "patched_images"
    clean_dir = source / "clean_images"
    out_root = mitigation_dir(mitigation_id)

    errors: list[str] = []
    if not attack_csv.is_file():
        errors.append(f"Missing {attack_csv}")
    if not selected_csv.is_file():
        errors.append(f"Missing {selected_csv}")
    if not patched_dir.is_dir():
        errors.append(f"Missing {patched_dir}")
    if not clean_dir.is_dir():
        errors.append(f"Missing clean images dir: {clean_dir}")
    if not MODEL_PATH.is_file():
        errors.append(f"Missing Hyper-YOLO weights: {MODEL_PATH}")
    if not HYPER_YOLO_PYTHON.is_file():
        errors.append(f"Missing Hyper-YOLO Python: {HYPER_YOLO_PYTHON}")
    if not SD_PYTHON.is_file():
        errors.append(f"Missing SD Python: {SD_PYTHON}")
    if not SD_MODEL_PATH.is_dir():
        errors.append(f"Missing SD model: {SD_MODEL_PATH}")
    if not RESTORE_WORKER_SCRIPT.is_file():
        errors.append(f"Missing restore worker: {RESTORE_WORKER_SCRIPT}")
    if not PATCH_NEUTRALIZATION:
        errors.append("PATCH_NEUTRALIZATION must be True for the frozen final config")
    if abs(STRENGTH - 0.55) > 1e-9:
        errors.append(f"Frozen strength must be 0.55, got {STRENGTH}")

    # Active mitigation_01 must not already exist before a fresh final run.
    if mitigation_id == "mitigation_01" and out_root.exists():
        errors.append(
            f"Active {out_root} already exists. "
            "Fresh final run requires it to be absent "
            "(use mitigation_01(archived) for historical outputs only)."
        )

    snapshots = assert_attack_read_only_snapshots(spec.source_attack_id)

    attack_rows, selected_map = load_attack_tables(spec.source_attack_id)
    patched_count = len(list(patched_dir.glob("*.jpg"))) if patched_dir.is_dir() else 0
    clean_count = len(list(clean_dir.glob("*.jpg"))) if clean_dir.is_dir() else 0
    checked, mismatch_count, mismatch_errors = validate_geometry_for_attack(spec.source_attack_id)

    subset_rows = select_mitigation_subset(spec)
    summary = summarize_subset(subset_rows)

    if mitigation_id == "mitigation_01":
        expected = {
            "success_count": 15,
            "control_count": 15,
            "success_person": 4,
            "success_car": 11,
            "control_person": 7,
            "control_car": 8,
        }
        for key, value in expected.items():
            if summary.get(key) != value:
                errors.append(
                    f"Unexpected subset {key}: got {summary.get(key)}, expected {value}"
                )

    # Verify each subset clean + attacked image exists (read-only)
    missing_images = []
    for row in subset_rows:
        for folder, label in ((patched_dir, "attacked"), (clean_dir, "clean")):
            path = folder / row.filename
            if not path.is_file():
                missing_images.append(f"{label}:{path}")
    if missing_images:
        errors.extend(missing_images[:10])
        if len(missing_images) > 10:
            errors.append(f"... and {len(missing_images) - 10} more missing source images")

    if mismatch_count:
        errors.extend(mismatch_errors[:20])

    img2img_ok, img2img_msg = check_sd_img2img_import()
    if not img2img_ok:
        errors.append(f"StableDiffusionImg2ImgPipeline import failed: {img2img_msg}")

    # Validate-only must NOT create active mitigation folders.
    if create_output_dirs:
        ensure_mitigation_structure(mitigation_id)

    verify_snapshots_unchanged(snapshots)

    global_attack_csv = RESULTS_DIR / "all_attacks_comparison.csv"
    global_attack_png = RESULTS_DIR / "all_attacks_comparison.png"
    attack06 = source.parent / "attack_06"
    archived = MITIGATIONS_DIR / "mitigation_01(archived)"
    pilot = SCRIPT_DIR.parent / "mitigation_debug" / "multi_image_pilot_055"

    report = {
        "mitigation_id": mitigation_id,
        "source_attack": spec.source_attack_id,
        "method_name": METHOD_NAME,
        "success_count": summary["success_count"],
        "control_pool_estimate": sum(
            1
            for row in attack_rows
            if row.get("eligible_for_asr") == "Yes"
            and row.get("attack_success") == "No"
            and row.get("attacked_detected") == "Yes"
        ),
        "selected_control_count": summary["control_count"],
        "person_count": summary["person_count"],
        "car_count": summary["car_count"],
        "success_person": summary["success_person"],
        "success_car": summary["success_car"],
        "control_person": summary["control_person"],
        "control_car": summary["control_car"],
        "required_csvs": [str(attack_csv), str(selected_csv)],
        "source_attacked_image_count": patched_count,
        "source_clean_image_count": clean_count,
        "gt_bbox_availability": "YES" if selected_map else "NO",
        "geometry_rows_checked": checked,
        "geometry_mismatches": mismatch_count,
        "patch_size_consistency": "PASS" if mismatch_count == 0 else "FAIL",
        "sd_python": str(SD_PYTHON),
        "sd_model_path": str(SD_MODEL_PATH),
        "hyper_yolo_model": str(MODEL_PATH),
        "img2img_import": img2img_msg if img2img_ok else f"FAIL: {img2img_msg}",
        "expected_output_root": str(out_root),
        "active_mitigation_exists": out_root.exists(),
        "created_output_dirs": bool(create_output_dirs),
        "patch_neutralization": "ON" if PATCH_NEUTRALIZATION else "OFF",
        "strength": STRENGTH,
        "source_clean_images_supported": "YES",
        "source_attacked_images_supported": "YES",
        "source_attack_metadata_supported": "YES",
        "full_restored_images_supported": "YES",
        "mitigated_prediction_images_supported": "YES",
        "mitigated_prediction_labels_supported": "YES",
        "three_panel_verification_supported": "YES",
        "mitigation_results_csv_fields_ready": "YES",
        "attack_files_would_be_modified": "NO",
        "attack_folders_remain_read_only": "YES",
        "attack_06_exists": attack06.exists(),
        "archived_mitigation_01_exists": archived.exists(),
        "pilot_folder_exists": pilot.exists(),
        "all_attacks_comparison_csv_exists": global_attack_csv.is_file(),
        "all_attacks_comparison_png_exists": global_attack_png.is_file(),
        "target_match_iou_threshold": TARGET_MATCH_IOU_THRESHOLD,
        "control_selection_seed": spec.control_selection_seed,
        "restoration_seed": spec.restoration_seed,
        "errors": errors,
        "ok": len(errors) == 0,
    }
    return report


def print_validation_report(report: dict) -> None:
    print("")
    print("=" * 72)
    print(f"Mitigation ID: {report['mitigation_id']}")
    print(f"Source attack: {report['source_attack']}")
    print(f"Method: {report['method_name']}")
    print(f"Success count: {report['success_count']}")
    print(f"Control pool count: {report['control_pool_estimate']}")
    print(f"Selected control count: {report['selected_control_count']}")
    print(
        f"Person/car counts (subset): {report['person_count']}/{report['car_count']} "
        f"(success person/car={report['success_person']}/{report['success_car']}; "
        f"control person/car={report['control_person']}/{report['control_car']})"
    )
    print("Required source CSVs:")
    for path in report["required_csvs"]:
        print(f"  - {path}")
    print(f"Source attacked image count: {report['source_attacked_image_count']}")
    print(f"Source clean image count: {report['source_clean_image_count']}")
    print(f"GT bbox availability: {report['gt_bbox_availability']}")
    print(
        f"Patch geometry consistency: checked={report['geometry_rows_checked']} "
        f"mismatches={report['geometry_mismatches']} "
        f"patch_size_consistency={report['patch_size_consistency']}"
    )
    print(f"SD Python path: {report['sd_python']}")
    print(f"SD model path: {report['sd_model_path']}")
    print(f"Hyper-YOLO model path: {report['hyper_yolo_model']}")
    print(f"StableDiffusionImg2ImgPipeline: {report['img2img_import']}")
    print(f"Expected output folders: {report['expected_output_root']}")
    print(f"Active mitigation folder exists: {report['active_mitigation_exists']}")
    print(f"Created output dirs during validation: {report['created_output_dirs']}")
    print(f"Patch neutralization: {report['patch_neutralization']}")
    print(f"Strength: {report['strength']}")
    print(f"Clean source copies supported: {report['source_clean_images_supported']}")
    print(f"Attacked source copies supported: {report['source_attacked_images_supported']}")
    print(f"Selected attack metadata supported: {report['source_attack_metadata_supported']}")
    print(f"Full restored images supported: {report['full_restored_images_supported']}")
    print(f"Hyper-YOLO restored prediction images supported: {report['mitigated_prediction_images_supported']}")
    print(f"Hyper-YOLO label files supported: {report['mitigated_prediction_labels_supported']}")
    print(f"3-way verification images supported: {report['three_panel_verification_supported']}")
    print(f"mitigation_results.csv fields ready: {report['mitigation_results_csv_fields_ready']}")
    print(f"Any attack file would be modified: {report['attack_files_would_be_modified']}")
    print(f"Attack folders remain read-only: {report['attack_folders_remain_read_only']}")
    print(f"attack_06 exists: {report['attack_06_exists']}")
    print(f"archived mitigation_01 exists: {report.get('archived_mitigation_01_exists')}")
    print(f"pilot folder exists: {report.get('pilot_folder_exists')}")
    print(f"Control selection seed: {report['control_selection_seed']}")
    print(f"Restoration seed: {report['restoration_seed']}")
    if report["errors"]:
        print("ERRORS:")
        for err in report["errors"]:
            print(f"  - {err}")
    else:
        print("Validation: PASS")
    print("=" * 72)


def restore_one_image(subset_row, mitigation_root: Path, restoration_seed: int) -> dict:
    """Restore one image via SD worker subprocess. Uses attacked copy only."""
    # Operate on the mitigation-local attacked copy only.
    copied = mitigation_root / "source_attacked_images" / subset_row.filename
    if not copied.is_file():
        raise MitigationError(f"Missing mitigation attacked copy: {copied}")
    if subset_row.reconstructed_patch_size != subset_row.patch_size_pixels:
        raise MitigationError(
            f"Patch size mismatch before restore for {subset_row.filename}: "
            f"reconstructed={subset_row.reconstructed_patch_size} "
            f"saved={subset_row.patch_size_pixels}"
        )
    stem = Path(subset_row.filename).stem
    mask_path = mitigation_root / "masks" / f"{stem}_patch_mask.png"
    restored_path = mitigation_root / "restored_images" / subset_row.filename
    neut_path = mitigation_root / "neutralized_inputs" / f"{stem}_512.png"
    meta_path = (
        mitigation_root
        / "results"
        / "comparison"
        / f"{stem}_restore_meta.json"
    )

    command = [
        str(SD_PYTHON),
        str(RESTORE_WORKER_SCRIPT),
        "--attacked-image",
        str(copied),
        "--output-restored",
        str(restored_path),
        "--output-mask",
        str(mask_path),
        "--model-path",
        str(SD_MODEL_PATH),
        "--x1",
        str(subset_row.reconstructed_x1),
        "--y1",
        str(subset_row.reconstructed_y1),
        "--x2",
        str(subset_row.reconstructed_x2),
        "--y2",
        str(subset_row.reconstructed_y2),
        "--prompt",
        POSITIVE_PROMPT,
        "--negative-prompt",
        NEGATIVE_PROMPT,
        "--seed",
        str(restoration_seed),
        "--steps",
        str(NUM_INFERENCE_STEPS),
        "--guidance-scale",
        str(GUIDANCE_SCALE),
        "--strength",
        str(STRENGTH),
        "--resolution",
        str(DIFFUSION_RESOLUTION),
        "--context-factor",
        str(CONTEXT_CROP_FACTOR),
        "--min-context-side",
        str(MINIMUM_CONTEXT_SIDE),
        "--feather-pixels",
        str(FEATHER_PIXELS),
        "--output-neutralized-512",
        str(neut_path),
        "--metadata-json",
        str(meta_path),
    ]
    if PATCH_NEUTRALIZATION:
        command.append("--neutralize")
    else:
        raise MitigationError("Frozen config requires PATCH_NEUTRALIZATION=True")

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "unknown restore failure").strip()
        raise MitigationError(f"Restoration failed for {subset_row.filename}:\n{detail}")
    if not restored_path.is_file():
        raise MitigationError(f"Restored image missing after worker success: {restored_path}")
    if not mask_path.is_file():
        raise MitigationError(f"Patch mask missing after worker success: {mask_path}")
    if not neut_path.is_file():
        raise MitigationError(f"Neutralized input missing after worker success: {neut_path}")

    device = "unknown"
    if meta_path.is_file():
        try:
            device = str(json.loads(meta_path.read_text(encoding="utf-8")).get("device", "unknown"))
        except json.JSONDecodeError:
            device = "unknown"
    return {"filename": subset_row.filename, "device": device, "restored": str(restored_path)}


def run_one_mitigation(mitigation_id: str) -> dict:
    """Fully automatic mitigation experiment (subset → restore → YOLO → report)."""
    from pipeline_core import (  # noqa: WPS433
        evaluate_target_match,
        load_yolo,
        parse_label_file,
        run_yolo_inference,
    )

    print("")
    print("=" * 72)
    print(f"FULL AUTOMATIC MITIGATION START: {mitigation_id}")
    print("=" * 72)

    # STAGE 0 — Preflight WITHOUT creating output folders
    preflight = validate_one(mitigation_id, create_output_dirs=False)
    print_validation_report(preflight)
    if not preflight["ok"]:
        raise MitigationError(
            f"Preflight validation FAILED for {mitigation_id}. "
            "Stable Diffusion was NOT started. mitigation folder NOT created."
        )

    global_csv = RESULTS_DIR / "all_attacks_comparison.csv"
    global_png = RESULTS_DIR / "all_attacks_comparison.png"
    global_hashes = {
        str(global_csv): file_sha1(global_csv) if global_csv.is_file() else "",
        str(global_png): file_sha1(global_png) if global_png.is_file() else "",
    }
    archived = MITIGATIONS_DIR / "mitigation_01(archived)"
    pilot = SCRIPT_DIR.parent / "mitigation_debug" / "multi_image_pilot_055"
    archived_hash = ""
    pilot_hash = ""
    if (archived / "source_attack_data" / "selected_subset.csv").is_file():
        archived_hash = file_sha1(archived / "source_attack_data" / "selected_subset.csv")
    if (pilot / "pilot_summary.txt").is_file():
        pilot_hash = file_sha1(pilot / "pilot_summary.txt")

    # Create fresh mitigation folder ONLY after preflight PASS
    if mitigation_dir(mitigation_id).exists():
        raise MitigationError(
            f"Refusing to overwrite existing {mitigation_dir(mitigation_id)}"
        )
    spec = get_mitigation_spec(mitigation_id)
    snapshots = assert_attack_read_only_snapshots(spec.source_attack_id)
    mitigation_root = ensure_mitigation_structure(mitigation_id)
    print(f"[STAGE 0] Dynamically created fresh folder: {mitigation_root}")
    results_dir = mitigation_root / "results"
    config = restoration_config_dict(spec)
    config["pilot_excluded_from_final_metrics"] = True
    config["archived_outputs_reused"] = False
    (results_dir / "restoration_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    # STAGE 1 — Subset
    subset_rows = select_mitigation_subset(spec)
    summary = summarize_subset(subset_rows)
    expected_success = summary["success_count"]
    expected_control = spec.control_count
    if summary["control_count"] != expected_control:
        raise MitigationError(
            f"Control count mismatch: got {summary['control_count']}, "
            f"expected {expected_control}"
        )
    if expected_success <= 0:
        raise MitigationError("No success cases selected; refusing to run.")
    print(
        f"[STAGE 1] Subset ready: total={summary['total']} "
        f"success={summary['success_count']} control={summary['control_count']} "
        f"(person success/control="
        f"{summary['success_person']}/{summary['control_person']}; "
        f"car success/control={summary['success_car']}/{summary['control_car']})"
    )

    # STAGE 2–3 — Preserve source images + attack metadata (fresh copies only)
    save_source_attack_package(mitigation_root, spec, subset_rows)
    copy_source_images(mitigation_root, subset_rows)
    print("[STAGE 2/3] Source clean/attacked copies + attack metadata preserved")

    # STAGE 4–7 — Geometry already validated; restore each image with neutralization
    restore_failures: list[str] = []
    restore_ok = 0
    sd_device = "unknown"
    total = len(subset_rows)
    for index, row in enumerate(subset_rows, start=1):
        print(
            f"[STAGE 5-12] Restoring {index}/{total}: {row.filename} "
            f"({row.subset_role}, {row.target_class}) neutralize=ON strength={STRENGTH}"
        )
        try:
            meta = restore_one_image(row, mitigation_root, spec.restoration_seed)
            restore_ok += 1
            if sd_device == "unknown":
                sd_device = meta["device"]
        except Exception as exc:  # noqa: BLE001 — record exact failure then stop
            restore_failures.append(f"{row.filename}: {exc}")
            raise MitigationError(
                f"Restoration failed for {row.filename}. "
                f"Completed before failure: {restore_ok}/{total}. "
                f"Detail: {exc}"
            ) from exc

    restored_dir = mitigation_root / "restored_images"
    restored_count = len(list(restored_dir.glob("*.jpg")))
    mask_count = len(list((mitigation_root / "masks").glob("*_patch_mask.png")))
    neut_count = len(list((mitigation_root / "neutralized_inputs").glob("*_512.png")))
    if restored_count != total:
        raise MitigationError(
            f"Restored image count mismatch before Hyper-YOLO: "
            f"expected {total}, found {restored_count}"
        )
    if mask_count != total:
        raise MitigationError(
            f"Mask count mismatch before Hyper-YOLO: expected {total}, found {mask_count}"
        )
    if neut_count != total:
        raise MitigationError(
            f"Neutralized input count mismatch: expected {total}, found {neut_count}"
        )
    print(f"[STAGE 7] Full restored images saved: {restored_count}/{total}")
    print(f"[STAGE 5] Patch masks saved: {mask_count}/{total}")
    print(f"[STAGE 11] Neutralized inputs saved: {neut_count}/{total}")
    print(f"[STAGE 6] Stable Diffusion device: {sd_device}")

    config["actual_sd_device"] = sd_device
    (results_dir / "restoration_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    # STAGE 8–9 — Hyper-YOLO on restored images
    print("[STAGE 8] Running Hyper-YOLO on restored images...")
    model = load_yolo()
    pred_images = results_dir / "mitigated_predictions" / "images"
    pred_labels = results_dir / "mitigated_predictions" / "labels"
    run_yolo_inference(
        model,
        restored_dir,
        pred_images,
        pred_labels,
        results_dir / "mitigated_predictions",
    )
    pred_image_count = len(list(pred_images.glob("*.jpg")))
    pred_label_count = len(list(pred_labels.glob("*.txt")))
    print(
        f"[STAGE 9] Hyper-YOLO outputs: images={pred_image_count} labels={pred_label_count}"
    )

    # STAGE 10–12 — Matching, CSV, verification
    result_rows = []
    for row in subset_rows:
        label_path = pred_labels / f"{Path(row.filename).stem}.txt"
        detections = parse_label_file(
            label_path, row.image_width, row.image_height, model
        )
        match = evaluate_target_match(
            detections,
            row.target_class,
            (row.bbox_x, row.bbox_y, row.bbox_width, row.bbox_height),
        )
        result = build_mitigation_result_row(
            row,
            str(match["detected"]),
            float(match["confidence"]),
            float(match["iou"]),
            spec.restoration_seed,
        )
        result_rows.append(result)
        draw_mitigation_verification(
            mitigation_root / "source_clean_images" / row.filename,
            mitigation_root / "source_attacked_images" / row.filename,
            restored_dir / row.filename,
            results_dir / "verification" / f"verify_{Path(row.filename).stem}.jpg",
            row,
            result,
        )

    write_mitigation_results_csv(results_dir / "mitigation_results.csv", result_rows)
    verification_count = len(list((results_dir / "verification").glob("verify_*.jpg")))
    print(f"[STAGE 11/12] Results CSV + verification images: {verification_count}")

    # STAGE 13–15 — Metrics, plots, summary
    metrics = compute_mitigation_metrics(result_rows)
    write_mitigation_summary(results_dir / "mitigation_summary.txt", metrics, config)
    write_comparison_plots(results_dir / "comparison", metrics, result_rows)
    create_mitigation_overview(
        results_dir / "verification",
        result_rows,
        results_dir / "mitigation_overview.jpg",
    )
    update_global_mitigations_comparison(metrics)

    verify_snapshots_unchanged(snapshots)
    for path_str, expected in global_hashes.items():
        if expected and file_sha1(Path(path_str)) != expected:
            raise MitigationError(f"Global attack comparison file changed: {path_str}")
    if archived_hash and file_sha1(archived / "source_attack_data" / "selected_subset.csv") != archived_hash:
        raise MitigationError("Archived mitigation_01 was modified (forbidden)")
    if pilot_hash and file_sha1(pilot / "pilot_summary.txt") != pilot_hash:
        raise MitigationError("Pilot folder was modified (forbidden)")

    run_report = {
        "mitigation_id": mitigation_id,
        "source_attack_id": spec.source_attack_id,
        "subset_total": summary["total"],
        "success_count": summary["success_count"],
        "control_count": summary["control_count"],
        "restorations_completed": restore_ok,
        "restorations_failed": len(restore_failures),
        "restore_failures": restore_failures,
        "restored_images_saved": restored_count,
        "masks_saved": mask_count,
        "neutralized_inputs_saved": neut_count,
        "pred_images_saved": pred_image_count,
        "pred_labels_saved": pred_label_count,
        "verification_images_saved": verification_count,
        "sd_device": sd_device,
        "patch_neutralization": True,
        "strength": STRENGTH,
        "archived_outputs_reused": False,
        "pilot_included_in_metrics": False,
        "metrics": metrics,
        "output_root": str(mitigation_root),
    }
    (results_dir / "run_report.json").write_text(
        json.dumps(run_report, indent=2), encoding="utf-8"
    )

    print(
        f"Completed {mitigation_id}: "
        f"recovery={float(metrics['detection_recovery_rate']):.2f}% "
        f"preserved={float(metrics['control_preservation_rate']):.2f}%"
    )
    return run_report


def main() -> None:
    args = parse_args()

    if args.validate_only:
        targets = (
            [args.mitigation_id]
            if args.mitigation_id
            else ["mitigation_01"]  # final workflow: validate the approved experiment
        )
        # If user asks for all/default without id, still validate only mitigation_01
        # for this frozen final step; mitigation_02 remains deferred.
        if args.mitigation_id is None:
            targets = ["mitigation_01"]
        all_ok = True
        for mitigation_id in targets:
            report = validate_one(mitigation_id, create_output_dirs=False)
            print_validation_report(report)
            all_ok = all_ok and report["ok"]
            if mitigation_dir(mitigation_id).exists():
                print(
                    f"ERROR: validate-only must not leave/create {mitigation_dir(mitigation_id)}"
                )
                all_ok = False

        print("")
        print("Mitigation implementation validation summary")
        print(f"- mitigations checked: {', '.join(targets)}")
        print(f"- active mitigation_01 created: NO")
        print(f"- Stable Diffusion inference run: NO")
        print(f"- Hyper-YOLO mitigation inference run: NO")
        print(f"- attack_06 created: NO")
        print(f"- attack files modified: NO")
        print(f"- all_attacks_comparison files modified: NO")
        if not all_ok:
            raise SystemExit(1)
        print("Overall validation: PASS")
        return

    # Real run requires an explicit mitigation id (never silently run all).
    if not args.mitigation_id:
        raise SystemExit(
            "ERROR: --mitigation-id is required for a real mitigation run. "
            "Example: python scripts\\run_mitigation_pipeline.py --mitigation-id mitigation_01"
        )

    run_one_mitigation(args.mitigation_id)


if __name__ == "__main__":
    main()
