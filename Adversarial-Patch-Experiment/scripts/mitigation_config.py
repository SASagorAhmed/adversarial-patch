#!/usr/bin/env python3
"""Fixed configuration for known-location diffusion mitigation experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
HYPER_YOLO = EXPERIMENT_ROOT.parent / "Hyper-YOLO"
STABLE_DIFFUSION = EXPERIMENT_ROOT.parent / "Stable-Diffusion-Patch"
ATTACKS_DIR = EXPERIMENT_ROOT / "attacks"
MITIGATIONS_DIR = EXPERIMENT_ROOT / "mitigations"
RESULTS_DIR = EXPERIMENT_ROOT / "results"

HYPER_YOLO_PYTHON = HYPER_YOLO / ".venv" / "Scripts" / "python.exe"
SD_PYTHON = STABLE_DIFFUSION / ".venv" / "Scripts" / "python.exe"
SD_MODEL_PATH = STABLE_DIFFUSION / "model"
MODEL_PATH = HYPER_YOLO / "weights" / "hyper-yolon.pt"
RESTORE_WORKER_SCRIPT = EXPERIMENT_ROOT / "scripts" / "restore_patch_region.py"

# Frozen Hyper-YOLO settings (must match attack phase)
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.7
IMAGE_SIZE = 640
DEVICE = "cpu"
TARGET_MATCH_IOU_THRESHOLD = 0.50

# Frozen attack patch geometry (must match attack phase)
PATCH_SCALE = 0.30

# Fixed Img2Img restoration configuration (same for all mitigations)
DIFFUSION_RESOLUTION = 512
NUM_INFERENCE_STEPS = 30
GUIDANCE_SCALE = 7.5
STRENGTH = 0.55
CONTEXT_CROP_FACTOR = 3.0
MINIMUM_CONTEXT_SIDE = 128
FEATHER_PIXELS = 4
RESTORATION_SEED = 20260727
PATCH_NEUTRALIZATION = True  # FROZEN: required before Img2Img

POSITIVE_PROMPT = (
    "realistic natural photograph, restore the original object surface, "
    "natural texture, consistent lighting, no sticker or artificial patch"
)
NEGATIVE_PROMPT = (
    "sticker, adversarial patch, logo, text, watermark, abstract pattern, "
    "cartoon, distortion"
)

METHOD_NAME = "Simplified known-location diffusion-based restoration baseline"


@dataclass(frozen=True)
class MitigationSpec:
    mitigation_id: str
    source_attack_id: str
    control_count: int
    control_selection_seed: int
    restoration_seed: int = RESTORATION_SEED


MITIGATION_SPECS: dict[str, MitigationSpec] = {
    "mitigation_01": MitigationSpec(
        mitigation_id="mitigation_01",
        source_attack_id="attack_02",
        control_count=15,
        control_selection_seed=22021501,
    ),
    "mitigation_02": MitigationSpec(
        mitigation_id="mitigation_02",
        source_attack_id="attack_05",
        control_count=20,
        control_selection_seed=22021505,
    ),
}


def get_mitigation_spec(mitigation_id: str) -> MitigationSpec:
    if mitigation_id not in MITIGATION_SPECS:
        raise KeyError(f"Unknown mitigation_id: {mitigation_id}")
    return MITIGATION_SPECS[mitigation_id]


def mitigation_dir(mitigation_id: str) -> Path:
    return MITIGATIONS_DIR / mitigation_id


def attack_dir(attack_id: str) -> Path:
    return ATTACKS_DIR / attack_id


def restoration_config_dict(spec: MitigationSpec) -> dict:
    return {
        "method_name": METHOD_NAME,
        "mitigation_id": spec.mitigation_id,
        "source_attack_id": spec.source_attack_id,
        "control_count": spec.control_count,
        "control_selection_seed": spec.control_selection_seed,
        "restoration_seed": spec.restoration_seed,
        "diffusion_resolution": DIFFUSION_RESOLUTION,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "strength": STRENGTH,
        "patch_neutralization": PATCH_NEUTRALIZATION,
        "context_crop_factor": CONTEXT_CROP_FACTOR,
        "minimum_context_side": MINIMUM_CONTEXT_SIDE,
        "feather_pixels": FEATHER_PIXELS,
        "positive_prompt": POSITIVE_PROMPT,
        "negative_prompt": NEGATIVE_PROMPT,
        "patch_scale": PATCH_SCALE,
        "target_match_iou_threshold": TARGET_MATCH_IOU_THRESHOLD,
        "yolo_conf": CONF_THRESHOLD,
        "yolo_iou": IOU_THRESHOLD,
        "yolo_imgsz": IMAGE_SIZE,
        "yolo_device": DEVICE,
        "sd_model_path": str(SD_MODEL_PATH),
        "local_files_only": True,
        "clean_pixels_used_for_restoration": False,
        "spec": asdict(spec),
    }


MITIGATION_SUBFOLDERS = (
    "source_clean_images",
    "source_attacked_images",
    "source_attack_data",
    "masks",
    "neutralized_inputs",
    "restored_images",
    "results/mitigated_predictions/images",
    "results/mitigated_predictions/labels",
    "results/verification",
    "results/comparison",
)


def ensure_mitigation_structure(mitigation_id: str) -> Path:
    root = mitigation_dir(mitigation_id)
    for relative in MITIGATION_SUBFOLDERS:
        (root / relative).mkdir(parents=True, exist_ok=True)
    return root
