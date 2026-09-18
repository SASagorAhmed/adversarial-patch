"""Write-safety: only DISPATCH-Teacher-Notebook may receive new files."""
from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(r"D:\project CS\DISPATCH-Teacher-Notebook").resolve()
BLOCKED_ROOTS = [
    Path(r"D:\project CS\DISPATCH-Defense").resolve(),
    Path(r"D:\project CS\DISPATCH-Paper-Faithful").resolve(),
    Path(r"D:\project CS\Hyper-YOLO").resolve(),
    Path(r"D:\project CS\Adversarial-Patch-Experiment").resolve(),
    Path(r"D:\project CS\Stable-Diffusion-Patch").resolve(),
]


def load_config() -> dict:
    return json.loads((PROJECT_ROOT / "config.json").read_text(encoding="utf-8"))


def assert_writable(path: Path | str) -> Path:
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise RuntimeError(f"WRITE BLOCKED: {resolved} is outside {PROJECT_ROOT}") from exc
    for blocked in BLOCKED_ROOTS:
        try:
            resolved.relative_to(blocked)
        except ValueError:
            continue
        raise RuntimeError(f"WRITE BLOCKED: {resolved} is under read-only project {blocked}")
    return resolved


def ensure_dir(path: Path | str) -> Path:
    p = assert_writable(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def subprocess_env(extra_pythonpath: list[str] | None = None) -> dict:
    """Env for DISPATCH/Hyper-YOLO subprocesses launched from Jupyter."""
    import os

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["PYTHONUNBUFFERED"] = "1"
    if extra_pythonpath:
        env["PYTHONPATH"] = os.pathsep.join(extra_pythonpath)
    return env


def sha256_file(path: Path | str) -> dict:
    import hashlib

    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return {"sha256": h.hexdigest(), "size_bytes": p.stat().st_size}
