#!/usr/bin/env python3
"""Inspect attack folder state and report the next dynamic attack number.

This script does NOT pre-create attack folders.
Attack folders are created only when a real attack experiment starts,
via attack_utils.create_attack_folder().
"""

from __future__ import annotations

from pathlib import Path

from attack_utils import (
    ATTACKS_DIR,
    attacks_folder_is_empty,
    get_next_attack_id,
    list_existing_attack_numbers,
)

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    existing = list_existing_attack_numbers()
    next_attack = get_next_attack_id()

    print("Attack folder status")
    print(f"- attacks root: {ATTACKS_DIR}")
    print(f"- attacks folder empty: {attacks_folder_is_empty()}")
    print(f"- existing attacks: {len(existing)}")
    if existing:
        print(
            "- attack numbers found: "
            + ", ".join(f"attack_{number:02d}" for number in existing)
        )
    print(f"- next attack when started: {next_attack}")
    print("")
    print("Note: attack folders are created dynamically by the attack runner,")
    print("not pre-created by this script.")


if __name__ == "__main__":
    main()
