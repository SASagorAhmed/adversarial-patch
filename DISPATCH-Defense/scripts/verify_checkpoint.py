#!/usr/bin/env python
"""Verify official CompVis inpainting checkpoint exists and is loadable."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.paths_config import LDM_INPAINT_CKPT, LDM_INPAINT_CONFIG


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ckpt = LDM_INPAINT_CKPT
    cfg = LDM_INPAINT_CONFIG
    print("config", cfg, "exists", cfg.is_file())
    print("ckpt", ckpt, "exists", ckpt.is_file())
    if not cfg.is_file() or not ckpt.is_file():
        print("CHECKPOINT VALIDATION: FAIL (missing files)")
        return 1

    size = ckpt.stat().st_size
    print("size_bytes", size)
    print("size_gb", round(size / (1024**3), 3))
    if size < 100 * 1024 * 1024:
        print("CHECKPOINT VALIDATION: FAIL (too small)")
        return 1

    with ckpt.open("rb") as f:
        head = f.read(64)
    if b"<html" in head.lower() or b"<!doctype" in head.lower():
        print("CHECKPOINT VALIDATION: FAIL (HTML)")
        return 1

    digest = sha256_file(ckpt)
    print("sha256", digest)

    try:
        import torch

        obj = torch.load(str(ckpt), map_location="cpu")
        if not isinstance(obj, dict) or "state_dict" not in obj:
            print("CHECKPOINT VALIDATION: FAIL (no state_dict)")
            return 1
        print("state_dict_keys", len(obj["state_dict"]))
        print("CHECKPOINT VALIDATION: PASS")
        return 0
    except Exception as e:
        print("CHECKPOINT VALIDATION: FAIL", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
