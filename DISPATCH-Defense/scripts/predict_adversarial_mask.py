#!/usr/bin/env python
"""KMeans(k=2) adversarial region prediction from smoothed difference map."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.dispatch_config import DISPATCH_SEED, KMEANS_CLUSTERS
from config.paths_config import assert_under_project_root


def predict_adversarial_mask(
    smoothed_d: np.ndarray,
    n_clusters: int = KMEANS_CLUSTERS,
    seed: int = DISPATCH_SEED,
) -> tuple[np.ndarray, dict]:
    """Return binary uint8 mask A (255=adversarial) and cluster stats.

    Higher-centroid cluster = suspicious/adversarial.
    Does NOT use true patch coordinates.
    """
    flat = smoothed_d.reshape(-1, 1).astype(np.float64)
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = km.fit_predict(flat)
    centers = km.cluster_centers_.reshape(-1)
    adv_label = int(np.argmax(centers))
    mask = (labels.reshape(smoothed_d.shape) == adv_label).astype(np.uint8) * 255
    stats = {
        "centroids": [float(c) for c in centers],
        "adversarial_label": adv_label,
        "benign_label": int(np.argmin(centers)),
        "predicted_mask_area_pixels": int((mask > 0).sum()),
        "total_pixels": int(mask.size),
        "predicted_mask_area_ratio": float((mask > 0).mean()),
        "n_clusters": n_clusters,
        "seed": seed,
    }
    return mask, stats


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--smoothed-npy", required=True)
    p.add_argument("--out-mask", required=True)
    p.add_argument("--out-stats-json", required=True)
    args = p.parse_args()

    d = np.load(args.smoothed_npy)
    mask, stats = predict_adversarial_mask(d)
    out_m = assert_under_project_root(args.out_mask)
    out_m.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask, mode="L").save(out_m)
    out_s = assert_under_project_root(args.out_stats_json)
    out_s.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
