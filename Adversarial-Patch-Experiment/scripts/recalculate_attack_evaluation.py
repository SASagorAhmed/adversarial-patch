#!/usr/bin/env python3
"""Recalculate attack evaluation from saved predictions without rerunning YOLO."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from comparison_reporting import (  # noqa: E402
    build_comparison_rows,
    draw_verification_image,
    run_comparison_reporting,
)
from pipeline_core import (  # noqa: E402
    ATTACKS_DIR,
    TARGET_MATCH_IOU_THRESHOLD,
    create_overview_image,
    recalculate_attack_evaluation,
    write_attack_results_csv,
    write_attack_summary,
    write_clean_baseline_csv,
    write_clean_baseline_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recalculate attack evaluation from saved YOLO label files."
    )
    parser.add_argument(
        "--attack-id",
        default="attack_01",
        help="Attack folder name to recalculate (default: attack_01).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    attack_dir = ATTACKS_DIR / args.attack_id
    if not attack_dir.is_dir():
        print(f"ERROR: attack folder not found: {attack_dir}", file=sys.stderr)
        sys.exit(1)

    attack_rows, selected, geometry, clean_results = recalculate_attack_evaluation(attack_dir)
    results_dir = attack_dir / "results"
    random_seed = selected[0].random_seed if selected else 0
    patch_filename = attack_rows[0].patch_filename if attack_rows else "unknown"

    write_clean_baseline_csv(results_dir / "clean_baseline.csv", selected, clean_results)
    write_clean_baseline_summary(
        results_dir / "clean_baseline_summary.txt",
        selected,
        clean_results,
    )
    write_attack_results_csv(results_dir / "attack_results.csv", attack_rows)

    comparison_rows = build_comparison_rows(attack_rows)
    run_comparison_reporting(
        args.attack_id,
        random_seed,
        patch_filename,
        comparison_rows,
        results_dir,
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
        args.attack_id,
        random_seed,
        len(selected),
        patch_filename,
        attack_rows,
    )

    eligible = sum(1 for row in attack_rows if row.eligible_for_asr == "Yes")
    successes = sum(1 for row in attack_rows if row.attack_success == "Yes")
    asr = (successes / eligible * 100.0) if eligible else 0.0

    person_eligible = [r for r in attack_rows if r.eligible_for_asr == "Yes" and r.target_class == "person"]
    car_eligible = [r for r in attack_rows if r.eligible_for_asr == "Yes" and r.target_class == "car"]
    person_successes = sum(1 for r in attack_rows if r.attack_success == "Yes" and r.target_class == "person")
    car_successes = sum(1 for r in attack_rows if r.attack_success == "Yes" and r.target_class == "car")
    person_asr = (person_successes / len(person_eligible) * 100.0) if person_eligible else 0.0
    car_asr = (car_successes / len(car_eligible) * 100.0) if car_eligible else 0.0

    print(f"Recalculated: {args.attack_id}")
    print(f"Target matching threshold: {TARGET_MATCH_IOU_THRESHOLD:.2f}")
    print(f"Clean eligible targets: {eligible}")
    print(f"Attack successes: {successes}")
    print(f"Corrected ASR: {asr:.2f}%")
    print(f"Person corrected ASR: {person_asr:.2f}%")
    print(f"Car corrected ASR: {car_asr:.2f}%")


if __name__ == "__main__":
    main()
