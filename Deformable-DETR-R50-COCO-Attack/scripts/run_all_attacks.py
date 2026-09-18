"""Thin wrappers for reproducibility (delegate to run_attack.py)."""
from __future__ import annotations

import run_attack


def main():
    raise SystemExit(run_attack.main())


if __name__ == "__main__":
    main()
