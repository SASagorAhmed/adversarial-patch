"""Load official Torchvision Faster R-CNN ResNet-50 FPN V2 (COCO_V1), frozen."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)

PROJECT_ROOT = Path(r"D:\project CS\Faster-RCNN-COCO-Attack").resolve()


def load_config() -> dict[str, Any]:
    path = PROJECT_ROOT / "configs" / "model_config.json"
    return json.loads(path.read_text(encoding="utf-8"))


def get_weights():
    """Official pretrained COCO_V1 weights enum member."""
    return FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1


def resolve_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(device: torch.device | None = None, *, freeze: bool = True):
    """
    Load fasterrcnn_resnet50_fpn_v2 with FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1.

    Model remains pretrained. freeze=True sets eval mode and requires_grad=False.
    """
    weights = get_weights()
    if device is None:
        device = resolve_device()

    model = fasterrcnn_resnet50_fpn_v2(weights=weights)
    model.to(device)
    model.eval()

    if freeze:
        for p in model.parameters():
            p.requires_grad_(False)

    transforms = weights.transforms()
    return model, weights, transforms, device


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())
