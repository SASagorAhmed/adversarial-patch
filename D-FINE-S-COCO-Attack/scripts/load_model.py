"""Load official D-FINE-S COCO checkpoint (Peterande/D-FINE)."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T

PROJECT = Path(r"D:\project CS\D-FINE-S-COCO-Attack").resolve()
DFINE_ROOT = PROJECT / "third_party" / "D-FINE"
CONFIG_PATH = DFINE_ROOT / "configs" / "dfine" / "dfine_hgnetv2_s_coco.yml"
CHECKPOINT_PATH = PROJECT / "weights" / "dfine_s_coco.pth"
EXPECTED_SHA256 = "48a6c8cc43eb57186843f752e2e8461ddd3326e0d3c575e71e6e960844683e89"

if str(DFINE_ROOT) not in sys.path:
    sys.path.insert(0, str(DFINE_ROOT))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_dfine_s(device: torch.device | None = None):
    from src.core import YAMLConfig
    from src.data.dataset.coco_dataset import mscoco_category2name, mscoco_label2category

    if device is None:
        device = resolve_device()
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(CONFIG_PATH)
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(CHECKPOINT_PATH)

    digest = sha256_file(CHECKPOINT_PATH)
    if digest != EXPECTED_SHA256:
        raise RuntimeError(
            f"Checkpoint hash mismatch for D-FINE-S COCO.\n"
            f"expected={EXPECTED_SHA256}\nactual={digest}\nSTOP."
        )
    if "dfine_s_coco" not in CHECKPOINT_PATH.name.lower():
        raise RuntimeError(f"Checkpoint name does not look like D-FINE-S COCO: {CHECKPOINT_PATH.name}")

    cfg = YAMLConfig(str(CONFIG_PATH), resume=str(CHECKPOINT_PATH))
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    ckpt = torch.load(str(CHECKPOINT_PATH), map_location="cpu", weights_only=False)
    state = ckpt["ema"]["module"] if "ema" in ckpt else ckpt["model"]
    cfg.model.load_state_dict(state)

    class DeployModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            return self.postprocessor(self.model(images), orig_target_sizes)

    model = DeployModel().to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    transforms = T.Compose([T.Resize((640, 640)), T.ToTensor()])
    label_to_coco_id = dict(mscoco_label2category)
    coco_id_to_name = dict(mscoco_category2name)

    # Verify required contiguous IDs
    if label_to_coco_id[0] != 1 or coco_id_to_name[1] != "person":
        raise RuntimeError("Class map mismatch: contiguous 0 is not person")
    if label_to_coco_id[2] != 3 or coco_id_to_name[3] != "car":
        raise RuntimeError("Class map mismatch: contiguous 2 is not car")

    meta = {
        "model": "D-FINE-S",
        "config": str(CONFIG_PATH),
        "checkpoint": str(CHECKPOINT_PATH),
        "checkpoint_sha256": digest,
        "checkpoint_url": "https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_s_coco.pth",
        "input_size": 640,
        "person_contiguous_class_id": 0,
        "car_contiguous_class_id": 2,
        "official_coco_AP": 48.5,
    }
    return model, transforms, device, label_to_coco_id, coco_id_to_name, meta
