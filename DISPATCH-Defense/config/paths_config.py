"""Central path configuration for DISPATCH-Defense.

All generated outputs MUST resolve under PROJECT_ROOT.
External resources are READ-ONLY.
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(r"D:\project CS\DISPATCH-Defense").resolve()

# Third-party CompVis Latent Diffusion
LDM_ROOT = PROJECT_ROOT / "third_party" / "latent-diffusion"
LDM_INPAINT_SCRIPT = LDM_ROOT / "scripts" / "inpaint.py"
LDM_INPAINT_CONFIG = LDM_ROOT / "models" / "ldm" / "inpainting_big" / "config.yaml"
LDM_INPAINT_CKPT = LDM_ROOT / "models" / "ldm" / "inpainting_big" / "last.ckpt"
LDM_INPAINT_EXAMPLES = LDM_ROOT / "data" / "inpainting_examples"

# Local project trees
CONFIG_DIR = PROJECT_ROOT / "config"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
MODELS_DIR = PROJECT_ROOT / "models"
ENVIRONMENT_DIR = PROJECT_ROOT / "environment"
DATASET_DIR = PROJECT_ROOT / "dataset"
SMOKE_TEST_DIR = PROJECT_ROOT / "smoke_test"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
MITIGATIONS_DIR = PROJECT_ROOT / "mitigations"
COMPARISONS_DIR = PROJECT_ROOT / "comparisons"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"
REPORTS_DIR = PROJECT_ROOT / "reports"

# External READ-ONLY resources (never write here)
EXTERNAL_HYPER_YOLO_ROOT = Path(r"D:\project CS\Hyper-YOLO").resolve()
EXTERNAL_HYPER_YOLO_WEIGHTS = EXTERNAL_HYPER_YOLO_ROOT / "weights" / "hyper-yolon.pt"
EXTERNAL_ATTACK_ROOT = Path(r"D:\project CS\Adversarial-Patch-Experiment").resolve()
EXTERNAL_SD_PATCH_ROOT = Path(r"D:\project CS\Stable-Diffusion-Patch").resolve()

# Official URLs (documentation / download)
COMPVIS_REPO_URL = "https://github.com/CompVis/latent-diffusion.git"
OFFICIAL_CKPT_URL = "https://heibox.uni-heidelberg.de/f/4d9ac7ea40c64582b7c9/?dl=1"
PAPER_ARXIV = "https://arxiv.org/abs/2509.04597"
PAPER_HTML = "https://arxiv.org/html/2509.04597"


def assert_under_project_root(path: Path | str) -> Path:
    """Refuse any write target that escapes PROJECT_ROOT."""
    resolved = Path(path).resolve()
    root = PROJECT_ROOT.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(
            f"OUTPUT ROOT ASSERTION FAILED: {resolved} is outside {root}"
        ) from exc
    return resolved


def ensure_dir(path: Path | str) -> Path:
    p = assert_under_project_root(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
