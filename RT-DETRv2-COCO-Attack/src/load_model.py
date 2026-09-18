"""Load official RT-DETRv2-S (rtdetrv2_r18vd) COCO checkpoint via lyuwenyu/RT-DETR."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T

PROJECT = Path(r"D:\project CS\RT-DETRv2-COCO-Attack").resolve()
RTDETR_ROOT = PROJECT / "third_party" / "RT-DETR" / "rtdetrv2_pytorch"
CONFIG_PATH = RTDETR_ROOT / "configs" / "rtdetrv2" / "rtdetrv2_r18vd_120e_coco.yml"
CHECKPOINT_PATH = PROJECT / "weights" / "rtdetrv2_r18vd_120e_coco_rerun_48.1.pth"

if str(RTDETR_ROOT) not in sys.path:
    sys.path.insert(0, str(RTDETR_ROOT))


def resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_rtdetrv2_s(device: torch.device | None = None):
    """Load frozen pretrained RT-DETRv2-S in deploy mode."""
    from src.core import YAMLConfig
    from src.data.dataset.coco_dataset import mscoco_category2name, mscoco_label2category

    if device is None:
        device = resolve_device()
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(CONFIG_PATH)
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(CHECKPOINT_PATH)

    cfg = YAMLConfig(str(CONFIG_PATH), resume=str(CHECKPOINT_PATH))
    ckpt = torch.load(str(CHECKPOINT_PATH), map_location="cpu", weights_only=False)
    state = ckpt["ema"]["module"] if "ema" in ckpt else ckpt["model"]
    cfg.model.load_state_dict(state)

    class DeployModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            return self.postprocessor(outputs, orig_target_sizes)

    model = DeployModel().to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    transforms = T.Compose([T.Resize((640, 640)), T.ToTensor()])
    # Deploy mode returns contiguous 0..79 labels; map via official tables.
    label_to_coco_id = dict(mscoco_label2category)
    coco_id_to_name = dict(mscoco_category2name)

    meta = {
        "model_name": "RT-DETRv2-S",
        "config": str(CONFIG_PATH),
        "checkpoint": str(CHECKPOINT_PATH),
        "input_size": 640,
        "person_contiguous_label": 0,
        "car_contiguous_label": 2,
        "person_coco_id": 1,
        "car_coco_id": 3,
    }
    return model, transforms, device, label_to_coco_id, coco_id_to_name, meta
