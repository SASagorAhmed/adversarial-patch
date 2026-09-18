#!/usr/bin/env python
"""Verify DISPATCH-Defense Python environment and key imports."""
from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.paths_config import LDM_ROOT, PROJECT_ROOT

TAMING_ROOT = PROJECT_ROOT / "third_party" / "taming-transformers"


def _try_import(name: str):
    try:
        m = importlib.import_module(name)
        ver = getattr(m, "__version__", "unknown")
        return True, str(ver)
    except Exception as e:
        return False, repr(e)


def main() -> int:
    print("PROJECT_ROOT", PROJECT_ROOT)
    print("python", sys.version.replace("\n", " "))
    print("executable", sys.executable)
    print("platform", platform.platform())

    # Compatibility: editable installs can fail when project path contains spaces.
    # Always put third-party roots on sys.path (documented deviation).
    for p in (LDM_ROOT, TAMING_ROOT):
        if p.is_dir() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
    print("compat_note: LDM_ROOT/TAMING_ROOT prepended to sys.path (space-in-path safe)")

    ok = True
    for mod in [
        "torch",
        "torchvision",
        "numpy",
        "PIL",
        "cv2",
        "sklearn",
        "omegaconf",
        "einops",
        "pytorch_lightning",
        "taming",
        "ldm",
    ]:
        success, info = _try_import(mod)
        print(f"import {mod}: {'PASS' if success else 'FAIL'} ({info})")
        ok = ok and success

    try:
        import torch

        print("torch_version", torch.__version__)
        print("cuda_built", torch.version.cuda)
        print("cuda_available", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("gpu_name", torch.cuda.get_device_name(0))
            props = torch.cuda.get_device_properties(0)
            print("gpu_vram_gb", round(props.total_memory / (1024**3), 2))
    except Exception as e:
        print("torch detail FAIL", e)
        ok = False

    try:
        from main import instantiate_from_config  # noqa: F401

        print("instantiate_from_config: PASS")
    except Exception as e:
        print("instantiate_from_config: FAIL", e)
        ok = False

    print("OVERALL", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
