#!/usr/bin/env python3
"""Remove pre-created empty attack folders (no files inside)."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ATTACKS_DIR = Path(__file__).resolve().parents[1] / "attacks"


def main() -> None:
    if not ATTACKS_DIR.is_dir():
        print("attacks folder missing")
        return

    blocked: list[tuple[str, list[str]]] = []
    removed: list[str] = []

    for attack_dir in sorted(ATTACKS_DIR.iterdir()):
        if not attack_dir.is_dir() or not attack_dir.name.startswith("attack_"):
            continue

        files = list(attack_dir.rglob("*"))
        file_paths = [path for path in files if path.is_file()]

        if file_paths:
            blocked.append(
                (
                    str(attack_dir),
                    [str(path.relative_to(ATTACKS_DIR)) for path in file_paths],
                )
            )
            continue

        shutil.rmtree(attack_dir)
        removed.append(attack_dir.name)

    if blocked:
        print("ERROR: Found attack folders with files. Stopping without deletion.")
        for folder, contents in blocked:
            print(f"- {folder}")
            for item in contents:
                print(f"    {item}")
        sys.exit(1)

    print(f"Removed {len(removed)} empty attack folders:")
    for name in removed:
        print(f"  - {name}")

    remaining = list(ATTACKS_DIR.iterdir()) if ATTACKS_DIR.is_dir() else []
    print(f"attacks folder empty: {len(remaining) == 0}")


if __name__ == "__main__":
    main()
