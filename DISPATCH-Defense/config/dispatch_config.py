"""Central DISPATCH method/config settings (single source of truth)."""
from __future__ import annotations

from .paths_config import LDM_INPAINT_CKPT, LDM_INPAINT_CONFIG, PAPER_ARXIV

# Reproducibility — do not scatter alternate seeds elsewhere
DISPATCH_SEED = 20260817

# Paper method
METHOD_NAME = "DISPATCH"
PAPER_ID = "arXiv:2509.04597"
PAPER_URL = PAPER_ARXIV

# Regeneration
RESOLUTION = 512
CHECKERBOARD_GRID_N = 32
DIFFUSION_STEPS = 5
SAMPLER = "DDIM"

# Rectification / localization
KMEANS_CLUSTERS = 2
# Smoothing kernel for difference map (Gaussian). Paper-style local smoothing.
DIFFERENCE_BLUR_KERNEL = 5
DIFFERENCE_BLUR_SIGMA = 1.0

# Target matching (canonical project rule)
TARGET_MATCH_IOU = 0.50

# Hyper-YOLO inference (external adapter; do not modify Hyper-YOLO)
HYPER_YOLO_CONF = 0.25
HYPER_YOLO_NMS_IOU = 0.70
HYPER_YOLO_IMGSZ = 640
HYPER_YOLO_DEVICE = "cpu"

# Checkpoint / model
CHECKPOINT_PATH = str(LDM_INPAINT_CKPT)
CHECKPOINT_CONFIG_PATH = str(LDM_INPAINT_CONFIG)

# Disk safety thresholds (GB)
MIN_FREE_GB_FOR_CKPT_DOWNLOAD = 8.0
MIN_FREE_GB_FOR_REAL_RUN = 15.0


def as_run_config_dict(**overrides) -> dict:
    """Snapshot used in every experiment run_config.json."""
    cfg = {
        "method": METHOD_NAME,
        "paper": PAPER_ID,
        "paper_url": PAPER_URL,
        "seed": DISPATCH_SEED,
        "resolution": RESOLUTION,
        "checkerboard_grid_n": CHECKERBOARD_GRID_N,
        "diffusion_steps": DIFFUSION_STEPS,
        "sampler": SAMPLER,
        "kmeans_clusters": KMEANS_CLUSTERS,
        "difference_blur_kernel": DIFFERENCE_BLUR_KERNEL,
        "difference_blur_sigma": DIFFERENCE_BLUR_SIGMA,
        "target_match_iou": TARGET_MATCH_IOU,
        "hyper_yolo_conf": HYPER_YOLO_CONF,
        "hyper_yolo_nms_iou": HYPER_YOLO_NMS_IOU,
        "hyper_yolo_imgsz": HYPER_YOLO_IMGSZ,
        "hyper_yolo_device": HYPER_YOLO_DEVICE,
        "checkpoint_path": CHECKPOINT_PATH,
        "checkpoint_config_path": CHECKPOINT_CONFIG_PATH,
        "true_patch_coords_supplied_to_defense": False,
        "previous_mitigation_results_reused": False,
    }
    cfg.update(overrides)
    return cfg


def seed_everything(seed: int | None = None) -> int:
    """Apply central seed to Python / NumPy / Torch where available."""
    import os
    import random

    s = DISPATCH_SEED if seed is None else int(seed)
    os.environ["PYTHONHASHSEED"] = str(s)
    random.seed(s)
    try:
        import numpy as np

        np.random.seed(s)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)
    except Exception:
        pass
    return s
