"""Clean-image inference for Faster R-CNN ResNet-50 FPN V2 (official COCO_V1)."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw, ImageFont

from coco_categories import categories_from_weights, class_name_from_label
from load_model import PROJECT_ROOT, count_parameters, load_config, load_model


def _font(size: int = 16):
    for p in (
        r"C:\Windows\Fonts\arialbd.ttf",
        r"C:\Windows\Fonts\arial.ttf",
    ):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def detections_to_records(
    output: dict,
    *,
    categories: list[str],
    image_id: str,
    model_name: str,
    weights_name: str,
) -> list[dict[str, Any]]:
    boxes = output["boxes"].detach().cpu().tolist()
    scores = output["scores"].detach().cpu().tolist()
    labels = output["labels"].detach().cpu().tolist()
    records = []
    for box, score, label in zip(boxes, scores, labels):
        records.append(
            {
                "model": model_name,
                "weights": weights_name,
                "image_id": image_id,
                "class_id": int(label),
                "class_name": class_name_from_label(categories, label),
                "confidence": float(score),
                "bbox": [float(x) for x in box],
            }
        )
    return records


def draw_detections(image: Image.Image, records: list[dict], conf_thresh: float) -> Image.Image:
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    font = _font(16)
    for det in records:
        if det["confidence"] < conf_thresh:
            continue
        x1, y1, x2, y2 = det["bbox"]
        color = (0, 180, 0) if det["class_name"] in {"person", "car"} else (40, 120, 220)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"{det['class_name']} {det['confidence']:.2f}"
        draw.text((x1 + 2, max(0, y1 - 18)), label, fill=color, font=font)
    return out


def run_inference(
    image_path: Path | None = None,
    *,
    conf_thresh: float | None = None,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    cfg = load_config()
    if image_path is None:
        image_path = PROJECT_ROOT / cfg["test_image_path"]
    image_path = Path(image_path).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    if conf_thresh is None:
        conf_thresh = float(cfg["visualization_confidence"])
    if out_dir is None:
        out_dir = PROJECT_ROOT / "results" / "clean_inference"
    out_dir.mkdir(parents=True, exist_ok=True)

    model, weights, transforms, device = load_model()
    categories = categories_from_weights(weights)
    image_id = image_path.stem
    image = Image.open(image_path).convert("RGB")

    # Official Torchvision preprocessing
    batch = transforms(image)

    t0 = time.perf_counter()
    with torch.inference_mode():
        outputs = model([batch.to(device)])
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    output = outputs[0]
    model_name = cfg["model_name"]
    weights_name = cfg["weights_enum"]
    raw = detections_to_records(
        output,
        categories=categories,
        image_id=image_id,
        model_name=model_name,
        weights_name=weights_name,
    )

    # Save original + annotated (vis filtered at conf_thresh; raw JSON keeps all)
    original_out = out_dir / f"{image_id}_original.jpg"
    annotated_out = out_dir / f"{image_id}_annotated.jpg"
    image.save(original_out, quality=95)
    draw_detections(image, raw, conf_thresh).save(annotated_out, quality=95)

    raw_json = out_dir / f"{image_id}_detections_raw.json"
    raw_json.write_text(json.dumps(raw, indent=2), encoding="utf-8")

    person_n = sum(1 for d in raw if d["class_name"] == "person" and d["confidence"] >= conf_thresh)
    car_n = sum(1 for d in raw if d["class_name"] == "car" and d["confidence"] >= conf_thresh)
    person_raw = sum(1 for d in raw if d["class_name"] == "person")
    car_raw = sum(1 for d in raw if d["class_name"] == "car")

    meta = {
        "model": model_name,
        "weights": weights_name,
        "image_id": image_id,
        "image_path": str(image_path),
        "image_source": cfg.get("test_image_source"),
        "device": str(device),
        "inference_time_sec": elapsed,
        "num_params": count_parameters(model),
        "num_categories_meta": len(categories),
        "visualization_confidence": conf_thresh,
        "num_detections_raw": len(raw),
        "num_detections_vis": sum(1 for d in raw if d["confidence"] >= conf_thresh),
        "person_detections_raw": person_raw,
        "car_detections_raw": car_raw,
        "person_detections_vis": person_n,
        "car_detections_vis": car_n,
        "original_image": str(original_out),
        "annotated_image": str(annotated_out),
        "raw_detections_json": str(raw_json),
        "focus_classes": cfg["focus_classes"],
        "frozen": True,
        "eval_mode": True,
    }
    meta_path = out_dir / f"{image_id}_inference_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    meta["metadata_path"] = str(meta_path)
    return meta


if __name__ == "__main__":
    result = run_inference()
    print(json.dumps(result, indent=2))
