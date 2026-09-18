#!/usr/bin/env python
"""Hyper-YOLO adapter using READ-ONLY Hyper-YOLO package (custom MANet).

Loads Hyper-YOLO from EXTERNAL_HYPER_YOLO_ROOT so custom modules resolve.
All outputs are written only under DISPATCH-Defense.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.dispatch_config import (
    HYPER_YOLO_CONF,
    HYPER_YOLO_DEVICE,
    HYPER_YOLO_IMGSZ,
    HYPER_YOLO_NMS_IOU,
)
from config.paths_config import (
    EXTERNAL_HYPER_YOLO_ROOT,
    EXTERNAL_HYPER_YOLO_WEIGHTS,
    assert_under_project_root,
)


def _purge_ultralytics_modules() -> None:
    for k in list(sys.modules):
        if k == "ultralytics" or k.startswith("ultralytics."):
            del sys.modules[k]


def load_hyper_yolo_model(weights: Path | None = None):
    """Import YOLO from the Hyper-YOLO repository (not pip ultralytics)."""
    hy_root = EXTERNAL_HYPER_YOLO_ROOT
    if not hy_root.is_dir():
        raise FileNotFoundError(f"Hyper-YOLO root missing: {hy_root}")
    weights = Path(weights or EXTERNAL_HYPER_YOLO_WEIGHTS)
    if not weights.is_file():
        raise FileNotFoundError(f"Hyper-YOLO weights not found: {weights}")
    _purge_ultralytics_modules()
    # Prefer Hyper-YOLO sources on sys.path
    hy = str(hy_root)
    if hy in sys.path:
        sys.path.remove(hy)
    sys.path.insert(0, hy)
    from ultralytics import YOLO  # type: ignore

    return YOLO(str(weights))


def predictions_from_result(r0) -> list[dict[str, Any]]:
    preds: list[dict[str, Any]] = []
    names = r0.names if hasattr(r0, "names") else {}
    if r0.boxes is not None and len(r0.boxes):
        xyxy = r0.boxes.xyxy.cpu().numpy()
        confs = r0.boxes.conf.cpu().numpy()
        clss = r0.boxes.cls.cpu().numpy().astype(int)
        for box, c, cls_id in zip(xyxy, confs, clss):
            preds.append(
                {
                    "class_id": int(cls_id),
                    "class_name": str(names.get(int(cls_id), cls_id)),
                    "confidence": float(c),
                    "box_xyxy": [float(x) for x in box.tolist()],
                }
            )
    return preds


def run_hyper_yolo(
    image_path: Path,
    out_dir: Path,
    image_id: str,
    conf: float = HYPER_YOLO_CONF,
    iou: float = HYPER_YOLO_NMS_IOU,
    imgsz: int = HYPER_YOLO_IMGSZ,
    device: str = HYPER_YOLO_DEVICE,
    weights: Path | None = None,
    model=None,
) -> dict[str, Any]:
    out_dir = assert_under_project_root(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if model is None:
        model = load_hyper_yolo_model(weights)

    results = model.predict(
        source=str(image_path),
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        verbose=False,
        save=False,
    )
    r0 = results[0]
    preds = predictions_from_result(r0)

    labels_path = assert_under_project_root(out_dir / "labels" / f"{image_id}_hyper_yolo.json")
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "image_id": image_id,
        "image_path": str(image_path),
        "weights": str(weights or EXTERNAL_HYPER_YOLO_WEIGHTS),
        "conf": conf,
        "iou": iou,
        "imgsz": imgsz,
        "device": device,
        "predictions": preds,
    }
    labels_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    viz_path = assert_under_project_root(out_dir / "images" / f"{image_id}_hyper_yolo.jpg")
    viz_path.parent.mkdir(parents=True, exist_ok=True)
    import cv2

    cv2.imwrite(str(viz_path), r0.plot())
    return payload


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--image-id", required=True)
    args = p.parse_args()
    out = run_hyper_yolo(Path(args.image), Path(args.out_dir), args.image_id)
    print(json.dumps({"n_preds": len(out["predictions"]), "labels": "ok"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
