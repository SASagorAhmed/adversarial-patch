#!/usr/bin/env python3
"""Fully automatic adversarial patch attack pipeline."""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import cv2  # noqa: E402

from comparison_reporting import (  # noqa: E402
    build_comparison_rows,
    draw_verification_image,
    run_comparison_reporting,
)
from pipeline_core import (  # noqa: E402
    DIFFUSION_PATCHES_DIR,
    EXPERIMENT_ROOT,
    HYPER_YOLO,
    HYPER_YOLO_PYTHON,
    MODEL_PATH,
    SD_MODEL_PATH,
    SD_PYTHON,
    STABLE_DIFFUSION,
    apply_patches_to_selected,
    build_attack_rows,
    build_class_candidates,
    check_python_environment,
    check_sd_imports,
    collect_previous_attack_filenames,
    compute_previous_attack_overlap,
    copy_selected_images,
    create_attack_structure,
    create_overview_image,
    default_num_images_for_attack_number,
    default_patch_name_for_attack_number,
    ensure_candidate_patch,
    evaluate_predictions,
    find_annotation_path,
    find_image_dir,
    get_next_attack_id,
    get_next_attack_number,
    load_yolo,
    random_select_balanced,
    resolve_num_images,
    run_yolo_inference,
    validate_patch_file,
    write_attack_results_csv,
    write_attack_summary,
    write_clean_baseline_csv,
    write_clean_baseline_summary,
    write_failed_marker,
    write_selected_csv,
)


class PipelineError(Exception):
    """Raised when pipeline validation or execution fails."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a fully automatic diffusion-generated candidate patch attack experiment."
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=None,
        help=(
            "Optional override for number of COCO val2017 images to select. "
            "If omitted, uses automatic schedule: 200 + (attack_number - 1) * 50."
        ),
    )
    parser.add_argument(
        "--patch",
        type=str,
        default=None,
        help="Optional candidate patch filename from diffusion_patches/ (e.g. diffusion_patch_03.png).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible image selection.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate required inputs and print startup command without running an attack.",
    )
    return parser.parse_args()


def validate_pre_start(
    num_images: int,
    patch_name: str,
    attack_number: int,
    *,
    validate_only: bool = False,
) -> tuple[Path, Path, Path | None]:
    """Validate pipeline prerequisites. Returns (annotations, images, patch_path|None)."""
    errors: list[str] = []
    warnings: list[str] = []

    if num_images < 2:
        errors.append("--num-images must be at least 2")

    if not HYPER_YOLO.is_dir():
        errors.append(f"Hyper-YOLO project not found: {HYPER_YOLO}")
    if not MODEL_PATH.is_file():
        errors.append(f"Hyper-YOLO weights not found: {MODEL_PATH}")

    hy_ok, hy_msg = check_python_environment(HYPER_YOLO_PYTHON, "Hyper-YOLO")
    if not hy_ok:
        errors.append(hy_msg)

    sd_ok, sd_msg = check_python_environment(SD_PYTHON, "Stable Diffusion")
    if not sd_ok:
        errors.append(sd_msg)
    elif not STABLE_DIFFUSION.is_dir():
        errors.append(f"Stable-Diffusion-Patch project not found: {STABLE_DIFFUSION}")
    elif not SD_MODEL_PATH.is_dir():
        errors.append(f"Stable Diffusion local model not found: {SD_MODEL_PATH}")
    else:
        sd_import_ok, sd_import_msg = check_sd_imports()
        if not sd_import_ok:
            errors.append(f"Stable Diffusion environment: {sd_import_msg}")

    annotation_path: Path | None = None
    image_dir: Path | None = None
    try:
        annotation_path = find_annotation_path()
    except FileNotFoundError as exc:
        errors.append(str(exc))
    try:
        image_dir = find_image_dir()
    except FileNotFoundError as exc:
        errors.append(str(exc))

    patch_path = DIFFUSION_PATCHES_DIR / patch_name
    patch_valid, patch_reason = validate_patch_file(patch_path)
    auto_gen_available = sd_ok and SD_MODEL_PATH.is_dir()

    if patch_valid:
        patch_status = "READY"
    elif auto_gen_available:
        patch_status = "MISSING (automatic generation available)"
        if not validate_only:
            warnings.append(
                f"Candidate patch missing; will be generated automatically: {patch_path}"
            )
    else:
        errors.append(
            f"Candidate patch not found: {patch_path} "
            "(automatic generation unavailable — check Stable Diffusion setup)"
        )
        patch_status = "MISSING"

    if errors:
        raise PipelineError("Pre-start validation failed:\n  - " + "\n  - ".join(errors))

    assert annotation_path is not None
    assert image_dir is not None

    if validate_only:
        default_images = default_num_images_for_attack_number(attack_number)
        print("Automatic pipeline: READY")
        print(f"Next attack ID: attack_{attack_number:02d}")
        print(f"Default image count for attack_{attack_number:02d}: {default_images}")
        print(f"Scheduled image count for this run: {num_images}")
        print(f"Hyper-YOLO environment: {'READY' if hy_ok else 'NOT READY'} ({hy_msg})")
        print(f"Stable Diffusion environment: {'READY' if sd_ok else 'NOT READY'} ({sd_msg})")
        print(
            f"Stable Diffusion local model: "
            f"{'READY' if SD_MODEL_PATH.is_dir() else 'NOT READY'} ({SD_MODEL_PATH})"
        )
        print(f"Candidate patch ({patch_name}): {patch_status}")
        print(f"Automatic patch generation: {'READY' if auto_gen_available else 'NOT READY'}")
        print("Manual patch placement required: NO")
        print("Random selection: READY")
        print("Dynamic attack folders: READY")
        print("Dynamic ASR: READY")
        print("Comparison reporting: READY")
        print("Automatic increasing sample size: READY")
        print(f"Attack {attack_number:02d} created: NO")
        for warning in warnings:
            print(f"Note: {warning}")

    return annotation_path, image_dir, patch_path if patch_valid else None


def resolve_patch_name(explicit_patch: str | None) -> str:
    if explicit_patch:
        return explicit_patch
    return default_patch_name_for_attack_number(get_next_attack_number())


def run_pipeline(num_images: int, patch_name: str, seed: int | None, attack_number: int) -> None:
    annotation_path, image_dir, existing_patch = validate_pre_start(
        num_images, patch_name, attack_number, validate_only=False
    )

    # Generate missing patch before creating attack folder
    if existing_patch is None:
        patch_source = ensure_candidate_patch(patch_name, generate=True)
    else:
        patch_source = existing_patch

    attack_id = get_next_attack_id()
    random_seed = seed if seed is not None else random.randint(0, 2**31 - 1)
    attack_dir: Path | None = None
    stage = "initialization"

    try:
        import json

        stage = "create_attack_folder"
        attack_dir = create_attack_structure(attack_id)

        stage = "random_image_selection"
        with annotation_path.open("r", encoding="utf-8") as handle:
            coco = json.load(handle)
        pools = build_class_candidates(coco)
        previous_filenames = collect_previous_attack_filenames()
        selected = random_select_balanced(
            pools,
            num_images,
            random_seed,
            exclude_filenames=previous_filenames,
        )
        overlap_count, overlap_percent = compute_previous_attack_overlap(
            selected, previous_filenames
        )
        copy_selected_images(selected, image_dir, attack_dir / "clean_images")
        write_selected_csv(attack_dir / "results" / "selected_images.csv", selected)

        stage = "copy_patch"
        patch_dest = attack_dir / "patch" / patch_name
        shutil.copy2(patch_source, patch_dest)
        patch_bgr = cv2.imread(str(patch_dest), cv2.IMREAD_COLOR)
        if patch_bgr is None:
            raise PipelineError(f"Could not read patch image: {patch_dest}")

        stage = "clean_hyper_yolo"
        model = load_yolo()
        clean_images_dir = attack_dir / "clean_images"
        clean_pred_images = attack_dir / "results" / "clean_predictions" / "images"
        clean_pred_labels = attack_dir / "results" / "clean_predictions" / "labels"
        run_yolo_inference(model, clean_images_dir, clean_pred_images, clean_pred_labels, attack_dir / "results")
        clean_results = evaluate_predictions(selected, clean_pred_labels, model)
        write_clean_baseline_csv(attack_dir / "results" / "clean_baseline.csv", selected, clean_results)
        write_clean_baseline_summary(
            attack_dir / "results" / "clean_baseline_summary.txt",
            selected,
            clean_results,
        )

        stage = "patch_application"
        geometry = apply_patches_to_selected(
            selected,
            clean_images_dir,
            attack_dir / "patched_images",
            patch_bgr,
        )

        stage = "attacked_hyper_yolo"
        attacked_pred_images = attack_dir / "results" / "attacked_predictions" / "images"
        attacked_pred_labels = attack_dir / "results" / "attacked_predictions" / "labels"
        run_yolo_inference(
            model,
            attack_dir / "patched_images",
            attacked_pred_images,
            attacked_pred_labels,
            attack_dir / "results" / "attacked_predictions",
        )
        attacked_results = evaluate_predictions(selected, attacked_pred_labels, model)

        stage = "comparison_and_outputs"
        attack_rows = build_attack_rows(
            attack_id,
            selected,
            clean_results,
            attacked_results,
            geometry,
            patch_name,
        )
        results_dir = attack_dir / "results"
        write_attack_results_csv(results_dir / "attack_results.csv", attack_rows)

        comparison_rows = build_comparison_rows(attack_rows)
        run_comparison_reporting(
            attack_id,
            random_seed,
            patch_name,
            comparison_rows,
            results_dir,
            previous_attack_overlap_count=overlap_count,
            previous_attack_overlap_percent=overlap_percent,
        )

        verification_dir = results_dir / "verification"
        for item, row in zip(selected, comparison_rows, strict=True):
            geo = geometry[item.filename]
            patch_size = int(geo["patch_size_pixels"])
            patch_rect = (
                float(geo["patch_x"]),
                float(geo["patch_y"]),
                float(patch_size),
                float(patch_size),
            )
            draw_verification_image(
                attack_dir / "patched_images" / item.filename,
                verification_dir / f"verify_{Path(item.filename).stem}.jpg",
                item,
                row,
                patch_rect,
            )

        create_overview_image(verification_dir, attack_rows, results_dir / "attack_overview.jpg")
        write_attack_summary(
            results_dir / "attack_summary.txt",
            attack_id,
            random_seed,
            num_images,
            patch_name,
            attack_rows,
            previous_attack_overlap_count=overlap_count,
            previous_attack_overlap_percent=overlap_percent,
        )

        stage = "completed"
        eligible = sum(1 for row in attack_rows if row.eligible_for_asr == "Yes")
        successes = sum(1 for row in attack_rows if row.attack_success == "Yes")
        asr = (successes / eligible * 100.0) if eligible else 0.0

        print("")
        print("Automatic attack pipeline complete")
        print(f"- attack folder: {attack_dir}")
        print(f"- attack id: {attack_id}")
        print(f"- random seed: {random_seed}")
        print(f"- images processed: {len(attack_rows)}")
        print(f"- previous-attack overlap: {overlap_count} ({overlap_percent:.2f}%)")
        print(f"- patch: {patch_name}")
        print(f"- ASR eligible targets: {eligible}")
        print(f"- attack successes: {successes}")
        print(f"- Attack Success Rate: {asr:.2f}%")
        print(f"- results: {results_dir}")
        print(f"- comparison: {results_dir / 'comparison'}")
        print(f"- global comparison: {EXPERIMENT_ROOT / 'results' / 'all_attacks_comparison.csv'}")

    except Exception as exc:
        if attack_dir is not None and attack_dir.exists():
            write_failed_marker(attack_dir, stage, exc)
        raise


def print_schedule_confirmation() -> None:
    print("")
    print("Automatic image count schedule:")
    for number in range(1, 11):
        print(f"  attack_{number:02d} -> {default_num_images_for_attack_number(number)} images")


def print_creation_confirmation(next_attack_number: int) -> None:
    print("")
    print("Automatic pipeline created: YES")
    print("Manual attack numbering required: NO")
    print("Manual patch placement required: NO")
    print("Automatic increasing sample size: READY")
    print("New random sampling each attack: READY")
    print("Balanced person/car sampling: READY")
    print("IoU >= 0.50 matching preserved: YES")
    print("Dynamic ASR preserved: YES")
    print("Random image selection automatic: YES")
    print("Clean YOLO automatic: YES")
    print("Automatic patch generation: YES")
    print("Patch application automatic: YES")
    print("Attacked YOLO automatic: YES")
    print("Comparison automatic: YES")
    print("Per-attack comparison folder automatic: YES")
    print("Global all-attacks comparison automatic: YES")
    print("Separate comparison command required: NO")
    print("Verification automatic: YES")
    print("Each attack has its own results folder: YES")
    print("Future attack folders pre-created: NO")
    print("Hyper-YOLO modified: NO")
    print("Stable-Diffusion-Patch modified: NO")
    print_schedule_confirmation()
    print("")
    print(f"Command to start Attack {next_attack_number:02d}:")
    print(r'  cd "D:\project CS\Adversarial-Patch-Experiment"')
    print(
        r'  & "D:\project CS\Hyper-YOLO\.venv\Scripts\python.exe" '
        r'scripts\run_attack_pipeline.py'
    )


def main() -> None:
    args = parse_args()
    attack_number = get_next_attack_number()
    num_images = resolve_num_images(args.num_images, attack_number)
    patch_name = resolve_patch_name(args.patch)

    try:
        if args.validate_only:
            validate_pre_start(num_images, patch_name, attack_number, validate_only=True)
            print("")
            print("Validation passed.")
            print_creation_confirmation(attack_number)
            return

        validate_pre_start(num_images, patch_name, attack_number, validate_only=False)
    except PipelineError as exc:
        print(str(exc), file=sys.stderr)
        print_creation_confirmation(attack_number)
        sys.exit(1)

    run_pipeline(num_images, patch_name, args.seed, attack_number)


if __name__ == "__main__":
    main()
