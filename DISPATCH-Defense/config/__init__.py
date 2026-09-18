"""DISPATCH-Defense config package."""
from .dispatch_config import DISPATCH_SEED, as_run_config_dict, seed_everything
from .paths_config import PROJECT_ROOT, assert_under_project_root

__all__ = [
    "PROJECT_ROOT",
    "DISPATCH_SEED",
    "as_run_config_dict",
    "seed_everything",
    "assert_under_project_root",
]
