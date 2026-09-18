"""Load standard Deformable DETR R50 (SenseTime / official paper checkpoint via transformers).

Windows note:
  Official fundamentalvision CUDA ms_deform_attn operators require Linux-style
  CUDA toolchain (nvcc + MSVC). Those tools are unavailable on this machine.
  We therefore load the SAME architecture and SenseTime R50 COCO weights through
  Hugging Face Transformers (`SenseTime/deformable-detr`), which implements
  standard multi-scale Deformable DETR R50 with:
    two_stage=False, with_box_refine=False, num_feature_levels=4, backbone=resnet50
  Official reported COCO box AP: 44.5
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch

PROJECT = Path(r"D:\project CS\Deformable-DETR-R50-COCO-Attack").resolve()
HF_MODEL_ID = "SenseTime/deformable-detr"
LOCAL_WEIGHTS_DIR = PROJECT / "weights" / "SenseTime_deformable-detr"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_deformable_detr_r50(device: torch.device | None = None):
    from transformers import AutoImageProcessor, DeformableDetrForObjectDetection

    if device is None:
        device = resolve_device()

    model_src = str(LOCAL_WEIGHTS_DIR) if (LOCAL_WEIGHTS_DIR / "config.json").is_file() else HF_MODEL_ID
    processor = AutoImageProcessor.from_pretrained(model_src)
    model = DeformableDetrForObjectDetection.from_pretrained(model_src)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    cfg = model.config
    # Hard verification: standard / base Deformable DETR R50 only
    if str(getattr(cfg, "backbone", "")).lower() not in {"resnet50", "resnet-50"}:
        raise RuntimeError(f"Unexpected backbone={cfg.backbone!r}; expected ResNet-50")
    if bool(getattr(cfg, "two_stage", False)):
        raise RuntimeError("Loaded TWO-STAGE Deformable DETR — STOP (need standard/base)")
    if bool(getattr(cfg, "with_box_refine", False)):
        raise RuntimeError("Loaded iterative box-refinement variant — STOP (need standard/base)")
    if int(getattr(cfg, "num_feature_levels", 0)) < 4:
        raise RuntimeError(
            f"num_feature_levels={cfg.num_feature_levels}; expected multi-scale (>=4), not single-scale"
        )

    id2label = dict(cfg.id2label)
    # COCO category ids in this checkpoint: person=1, car=3
    person_ids = [i for i, n in id2label.items() if str(n).lower() == "person"]
    car_ids = [i for i, n in id2label.items() if str(n).lower() == "car"]
    if 1 not in person_ids and "person" not in [str(id2label.get(1, "")).lower()]:
        # also accept string keys
        pass
    if str(id2label.get(1, id2label.get("1", ""))).lower() != "person":
        raise RuntimeError(f"Class map mismatch: id 1 is {id2label.get(1)!r}, expected person")
    if str(id2label.get(3, id2label.get("3", ""))).lower() != "car":
        raise RuntimeError(f"Class map mismatch: id 3 is {id2label.get(3)!r}, expected car")

    # Contiguous YOLO-style ids for reporting (person=0, car=2)
    coco_cat_to_contiguous = {1: 0, 3: 2}
    contiguous_to_name = {0: "person", 2: "car"}

    weight_files = list(Path(model_src).glob("*.bin")) + list(Path(model_src).glob("*.safetensors")) if Path(model_src).is_dir() else []
    weight_hash = None
    weight_size = None
    weight_path = None
    for wf in weight_files:
        weight_path = str(wf)
        weight_size = wf.stat().st_size
        weight_hash = sha256_file(wf)
        break

    meta = {
        "model": "Deformable DETR R50",
        "backbone": "ResNet-50",
        "variant": "standard/base multi-scale (NOT single-scale, NOT box-refine, NOT two-stage)",
        "official_config": "configs/r50_deformable_detr.sh",
        "official_repo": "https://github.com/fundamentalvision/Deformable-DETR",
        "paper": "https://arxiv.org/abs/2010.04159",
        "official_coco_AP": 44.5,
        "checkpoint_source": HF_MODEL_ID,
        "checkpoint_local": model_src,
        "checkpoint_path": weight_path,
        "checkpoint_sha256": weight_hash,
        "checkpoint_size_bytes": weight_size,
        "two_stage": False,
        "with_box_refine": False,
        "num_feature_levels": int(cfg.num_feature_levels),
        "num_queries": int(cfg.num_queries),
        "implementation": (
            "Hugging Face Transformers DeformableDetrForObjectDetection "
            "(SenseTime weights). Official CUDA ops not compiled: no nvcc/MSVC on this Windows host."
        ),
        "person_coco_category_id": 1,
        "car_coco_category_id": 3,
        "person_contiguous_id": 0,
        "car_contiguous_id": 2,
        "device": str(device),
    }
    return model, processor, device, id2label, coco_cat_to_contiguous, contiguous_to_name, meta
