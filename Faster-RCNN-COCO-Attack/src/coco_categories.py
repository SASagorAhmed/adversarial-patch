"""COCO category helpers for FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1."""
from __future__ import annotations

from typing import Sequence


def categories_from_weights(weights) -> list[str]:
    """Return the official category list from weights.meta['categories']."""
    cats = weights.meta.get("categories")
    if cats is None:
        raise RuntimeError("weights.meta missing 'categories'")
    return list(cats)


def class_name_from_label(categories: Sequence[str], label: int) -> str:
    """Map a model label index to the official COCO category name."""
    idx = int(label)
    if idx < 0 or idx >= len(categories):
        raise IndexError(f"label {idx} out of range for {len(categories)} categories")
    return str(categories[idx])


def count_categories(categories: Sequence[str], *, include_background: bool = True) -> int:
    if include_background:
        return len(categories)
    return sum(1 for c in categories if c != "__background__")
