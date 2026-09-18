"""Load official YOLOv12-S COCO checkpoint (sunsmarterjie/yolov12 v1.0)."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch

PROJECT = Path(r"D:\project CS\YOLOv12-S-COCO-Attack").resolve()
YOLOV12_ROOT = PROJECT / "third_party" / "yolov12"
CHECKPOINT_PATH = PROJECT / "weights" / "yolov12s.pt"
CHECKPOINT_URL = "https://github.com/sunsmarterjie/yolov12/releases/download/v1.0/yolov12s.pt"
EXPECTED_SIZE_BYTES = 19006455

if str(YOLOV12_ROOT) not in sys.path:
    sys.path.insert(0, str(YOLOV12_ROOT))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_device() -> str:
    return "0" if torch.cuda.is_available() else "cpu"


def load_yolov12s(device: str | None = None):
    from ultralytics import YOLO

    if device is None:
        device = resolve_device()
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(CHECKPOINT_PATH)
    size = CHECKPOINT_PATH.stat().st_size
    if size != EXPECTED_SIZE_BYTES:
        raise RuntimeError(
            f"Checkpoint size mismatch for official v1.0 yolov12s.pt.\n"
            f"expected={EXPECTED_SIZE_BYTES} actual={size}\nSTOP."
        )
    if CHECKPOINT_PATH.name.lower() != "yolov12s.pt":
        raise RuntimeError(f"Checkpoint name is not yolov12s.pt: {CHECKPOINT_PATH.name}")

    digest = sha256_file(CHECKPOINT_PATH)
    model = YOLO(str(CHECKPOINT_PATH))

    names = model.names
    if names.get(0) != "person":
        raise RuntimeError(f"Class map mismatch: id 0 is {names.get(0)!r}, expected 'person'")
    if names.get(2) != "car":
        raise RuntimeError(f"Class map mismatch: id 2 is {names.get(2)!r}, expected 'car'")

    # Probe model task / variant
    task = getattr(model, "task", None) or "detect"
    if str(task).lower() not in {"detect", "detection"}:
        raise RuntimeError(f"Unexpected task={task!r}; expected detection")

    repo_commit = None
    try:
        import subprocess

        repo_commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(YOLOV12_ROOT),
                text=True,
            ).strip()
        )
    except Exception:
        repo_commit = None

    meta = {
        "model": "YOLOv12-S",
        "model_variant": "yolov12s",
        "task": "detection",
        "dataset": "MS COCO 2017",
        "checkpoint": str(CHECKPOINT_PATH),
        "checkpoint_filename": CHECKPOINT_PATH.name,
        "checkpoint_url": CHECKPOINT_URL,
        "checkpoint_release": "v1.0",
        "checkpoint_size_bytes": size,
        "checkpoint_sha256": digest,
        "input_size": 640,
        "official_coco_mAP50_95": 48.0,
        "person_class_id": 0,
        "car_class_id": 2,
        "repo": "https://github.com/sunsmarterjie/yolov12",
        "repo_commit": repo_commit,
        "repo_tag": "v1.0",
        "flash_attn": False,
        "attention_note": (
            "Official v1.0 yolov12s.pt + v1.0 source tag; "
            "FlashAttention unavailable on Windows — local SDPA/manual attention fallback only "
            "(weights and architecture unchanged)."
        ),
        "paper": "YOLOv12: Attention-Centric Real-Time Object Detectors",
        "publication": "NeurIPS 2025",
        "device": device,
    }
    return model, device, names, meta
