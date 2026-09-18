#!/usr/bin/env python
"""Reporting helpers for DISPATCH runs."""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config.paths_config import PROJECT_ROOT, REPORTS_DIR, assert_under_project_root


STORAGE_GUARANTEE = (
    "ALL GENERATED DISPATCH ARTIFACTS ARE STORED UNDER D:\\project CS\\DISPATCH-Defense\\"
)


def write_summary_files(run_dir: Path, summary: dict[str, Any]) -> None:
    run_dir = assert_under_project_root(run_dir)
    results = run_dir / "results"
    results.mkdir(parents=True, exist_ok=True)
    summary = {
        **summary,
        "storage_guarantee": STORAGE_GUARANTEE,
        "external_resources_readonly": True,
        "previous_mitigation_results_reused": False,
        "true_patch_coords_supplied_to_defense": False,
    }
    (results / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [f"{k}: {v}" for k, v in summary.items()]
    (results / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_experiment_inventory(row: dict[str, Any]) -> None:
    inv = assert_under_project_root(REPORTS_DIR / "experiment_inventory.csv")
    inv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "run_id",
        "datetime",
        "source_attack",
        "n_selected",
        "n_completed",
        "n_failed",
        "checkpoint_sha256",
        "ldm_commit",
        "seed",
        "grid_n",
        "diffusion_steps",
        "sampler",
        "experiment_folder",
        "status",
    ]
    exists = inv.is_file() and inv.stat().st_size > 0
    with inv.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if not exists:
            w.writeheader()
        if "datetime" not in row:
            row = {**row, "datetime": datetime.now(timezone.utc).isoformat()}
        w.writerow(row)


def make_run_id(experiments_dir: Path) -> str:
    experiments_dir = assert_under_project_root(experiments_dir)
    experiments_dir.mkdir(parents=True, exist_ok=True)
    n = 1
    while True:
        rid = f"dispatch_run_{n:03d}"
        if not (experiments_dir / rid).exists():
            return rid
        n += 1
