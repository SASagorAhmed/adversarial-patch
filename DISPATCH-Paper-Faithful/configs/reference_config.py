"""Paper/source-faithful DisPatch settings. Isolated project only."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(r"D:\project CS\DISPATCH-Paper-Faithful").resolve()
DISPATCH_DEFENSE = Path(r"D:\project CS\DISPATCH-Defense").resolve()
HYPER_YOLO_ROOT = Path(r"D:\project CS\Hyper-YOLO").resolve()
HYPER_YOLO_WEIGHTS = HYPER_YOLO_ROOT / "weights" / "hyper-yolon.pt"

OFFICIAL_REPO = ROOT / "reference_source" / "DisPatch"
OFFICIAL_COMMIT = "80c59b9b4a6c4060107a8b4485d3885128c2ae81"
OFFICIAL_LDM_ROOT = OFFICIAL_REPO / "LDM"

# Read-only CompVis checkpoint already present in DISPATCH-Defense (do not copy).
LDM_CKPT = DISPATCH_DEFENSE / "third_party" / "latent-diffusion" / "models" / "ldm" / "inpainting_big" / "last.ckpt"
LDM_CONFIG = OFFICIAL_LDM_ROOT / "models" / "ldm" / "inpainting_big" / "config.yaml"
TAMING_ROOT = DISPATCH_DEFENSE / "third_party" / "taming-transformers"

PAPER_ARXIV = "https://arxiv.org/abs/2509.04597"
PAPER_VERSION = "arXiv:2509.04597v2"
GITHUB = "https://github.com/MaJinWakeUp/DisPatch"

# Official d3_inference.py defaults
RESOLUTION = 512
NUM_GRIDS = 32
DIFFUSION_STEPS = 5
KMEANS_CLUSTERS = 2
GAUSSIAN_SIGMA = 1.0  # torchvision GaussianBlur sigma in official source
# kernel_size = min(15, size // num_grids - 1)  -> 15 at 512/32
MORPH_OPEN_KERNEL = 3  # official shape_completion uses MORPH_OPEN 3x3
RESIZE_RESAMPLING = "BICUBIC"  # PIL Image.resize default used by official source

# Diagnostic reproducibility (official KMeans has no random_state)
RUN_SEED = 20260817

# Canonical detector settings from DISPATCH-Defense (unchanged)
HYPER_YOLO_CONF = 0.25
HYPER_YOLO_NMS_IOU = 0.70
HYPER_YOLO_IMGSZ = 640
HYPER_YOLO_DEVICE = "cpu"
TARGET_MATCH_IOU = 0.50

SELECTED_IDS = [
    "000000127263",
    "000000393093",
    "000000026926",
    "000000407083",
    "000000181499",
    "000000347370",
    "000000385029",
    "000000542776",
    "000000331604",
    "000000357737",
]

CURRENT_AUTO_RESULTS = (
    DISPATCH_DEFENSE
    / "testing_mitigations"
    / "mitigation_02"
    / "automatic"
    / "results"
    / "per_image_results.csv"
)
CURRENT_KNOWN_RESULTS = (
    DISPATCH_DEFENSE
    / "testing_mitigations"
    / "mitigation_02"
    / "known_location"
    / "results"
    / "per_image_results.csv"
)
CURRENT_SELECTED = (
    DISPATCH_DEFENSE
    / "testing_mitigations"
    / "mitigation_02"
    / "automatic"
    / "results"
    / "selected_images.csv"
)
CURRENT_ATTACKED = (
    DISPATCH_DEFENSE
    / "testing_mitigations"
    / "mitigation_02"
    / "automatic"
    / "source_attacked_images"
)
CURRENT_CLEAN = (
    DISPATCH_DEFENSE
    / "testing_mitigations"
    / "mitigation_02"
    / "automatic"
    / "source_clean_images"
)


def assert_isolated(path: Path | str) -> Path:
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError as exc:
        raise RuntimeError(f"STOP: write path outside DISPATCH-Paper-Faithful: {resolved}") from exc
    return resolved
