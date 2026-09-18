"""Official-source L2 + GaussianBlur + KMeans k=2 + MORPH_OPEN.

Faithful to detect_pixel_level_kmeans(..., with_loc=False) in d3_inference.py.
Does not use true patch coordinates, GT boxes, or target class.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
from sklearn.cluster import KMeans
from torchvision.transforms import GaussianBlur


def l2_distance(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    """Per-pixel RGB L2 on tensors in [0,1], shape (3,H,W)."""
    return torch.norm(t1 - t2, p=2, dim=0, keepdim=False)


def gaussian_kernel_size(size: int, num_grids: int) -> int:
    ksize = size // num_grids - 1
    return int(min(15, ksize))


def detect_adversarial_mask(
    original_01: torch.Tensor,
    generated_01: torch.Tensor,
    size: int = 512,
    num_grids: int = 32,
    kmeans_random_state: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Return adv_mask (H,W) in {0,1}, smoothed distances, stats.

    original/generated: (3,H,W) float [0,1] on same device.
    """
    assert original_01.shape == generated_01.shape
    assert original_01.shape[-1] == size
    device = original_01.device
    distances = l2_distance(original_01, generated_01)
    ksize = gaussian_kernel_size(size, num_grids)
    distances_s = GaussianBlur(kernel_size=ksize, sigma=1.0)(distances.unsqueeze(0)).squeeze(0)

    distances_np = distances_s.detach().cpu().numpy().reshape(-1, 1)
    km_kw = {"n_clusters": 2, "n_init": "auto"}
    if kmeans_random_state is not None:
        km_kw["random_state"] = int(kmeans_random_state)
    kmeans = KMeans(**km_kw).fit(distances_np)
    centroids = kmeans.cluster_centers_.reshape(-1)
    labels = kmeans.labels_
    if centroids[1] <= centroids[0]:
        pos_label = 0
    else:
        pos_label = 1
    labels_t = torch.tensor(labels, device=device).reshape(distances_s.shape)
    adv_mask = torch.zeros_like(distances_s)
    adv_mask[labels_t == pos_label] = 1
    adv_mask = shape_completion(adv_mask)

    stats = {
        "gaussian_kernel_size": ksize,
        "gaussian_sigma": 1.0,
        "centroids": [float(c) for c in centroids],
        "adversarial_label": int(pos_label),
        "predicted_mask_area_512": int((adv_mask > 0.5).sum().item()),
        "morph_open": True,
        "with_loc": False,
        "kmeans_n_init": "auto",
        "kmeans_random_state": kmeans_random_state,
    }
    return adv_mask, distances_s, stats


def shape_completion(mask: torch.Tensor) -> torch.Tensor:
    """Official: cv2.MORPH_OPEN 3x3. Dilate is commented out in source."""
    mask_np = mask.detach().cpu().numpy().astype(np.uint8)
    mask_np_ = cv2.morphologyEx(mask_np, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return torch.from_numpy(mask_np_).float().to(mask.device)
