#!/usr/bin/env python
"""Verify external READ-ONLY resources exist; never write outside DISPATCH root."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.paths_config import (
    EXTERNAL_ATTACK_ROOT,
    EXTERNAL_HYPER_YOLO_ROOT,
    EXTERNAL_HYPER_YOLO_WEIGHTS,
    EXTERNAL_SD_PATCH_ROOT,
    PROJECT_ROOT,
    assert_under_project_root,
)


def main() -> int:
    ok = True
    checks = {
        "PROJECT_ROOT": PROJECT_ROOT,
        "Hyper-YOLO": EXTERNAL_HYPER_YOLO_ROOT,
        "hyper-yolon.pt": EXTERNAL_HYPER_YOLO_WEIGHTS,
        "Adversarial-Patch-Experiment": EXTERNAL_ATTACK_ROOT,
        "Stable-Diffusion-Patch": EXTERNAL_SD_PATCH_ROOT,
    }
    for name, path in checks.items():
        exists = path.exists()
        print(f"{name}: {'PASS' if exists else 'FAIL'} ({path})")
        ok = ok and exists

    # Output root assertion self-test
    try:
        assert_under_project_root(PROJECT_ROOT / "smoke_test" / "report" / "x.txt")
        print("assert_under_project_root self-test: PASS")
    except Exception as e:
        print("assert_under_project_root self-test: FAIL", e)
        ok = False

    try:
        assert_under_project_root(EXTERNAL_ATTACK_ROOT / "should_not_write.txt")
        print("escape detection: FAIL (should have raised)")
        ok = False
    except RuntimeError:
        print("escape detection: PASS")

    # Confirm previous mitigation folders are NOT used as sources here
    print("previous_mitigation_results_reused: NO (by policy)")
    print("OVERALL", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
