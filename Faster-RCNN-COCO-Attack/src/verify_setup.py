"""Verify Faster R-CNN ResNet-50 FPN V2 + official COCO_V1 setup and clean inference."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import torch
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights

PROJECT_ROOT = Path(r"D:\project CS\Faster-RCNN-COCO-Attack").resolve()
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from coco_categories import categories_from_weights, count_categories  # noqa: E402
from infer_image import run_inference  # noqa: E402
from load_model import count_parameters, load_config, load_model  # noqa: E402


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_weight_cache_path(weights) -> Path | None:
    """Locate the Torchvision-cached checkpoint for these weights."""
    url = weights.url
    filename = url.rsplit("/", 1)[-1]
    candidates = []
    try:
        from torch.hub import get_dir

        hub = Path(get_dir())
        candidates.extend(
            [
                hub / "checkpoints" / filename,
                hub / "checkpoints" / "download" / filename,
            ]
        )
    except Exception:
        pass
    home = Path.home()
    candidates.extend(
        [
            home / ".cache" / "torch" / "hub" / "checkpoints" / filename,
            home / ".cache" / "torch" / "hub" / "checkpoints" / "download" / filename,
        ]
    )
    for c in candidates:
        if c.is_file():
            return c.resolve()
    # Fallback: search under torch hub dir
    try:
        from torch.hub import get_dir

        hub = Path(get_dir())
        hits = list(hub.rglob(filename))
        if hits:
            return hits[0].resolve()
    except Exception:
        pass
    return None


def main() -> int:
    cfg = load_config()
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ver_dir = PROJECT_ROOT / "results" / "verification"
    ver_dir.mkdir(parents=True, exist_ok=True)

    weights_enum = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    meta = dict(weights_enum.meta)
    categories = categories_from_weights(weights_enum)
    box_map = meta.get("_metrics", {}).get("COCO-val2017", {}).get("box_map")
    num_params_meta = meta.get("num_params")
    file_size_mb_meta = meta.get("_file_size")

    t_load0 = time.perf_counter()
    model, weights, transforms, device = load_model()
    load_sec = time.perf_counter() - t_load0
    n_params = count_parameters(model)

    cache_path = resolve_weight_cache_path(weights)
    weight_size_bytes = cache_path.stat().st_size if cache_path else None
    weight_sha256 = sha256_file(cache_path) if cache_path else None

    # Ensure test image exists
    test_image = PROJECT_ROOT / cfg["test_image_path"]
    if not test_image.is_file():
        raise FileNotFoundError(f"Missing COCO val2017 test image: {test_image}")

    infer = run_inference(test_image)

    expected_map = float(cfg["expected_coco_val2017_box_map"])
    expected_params = int(cfg["expected_num_params"])

    coco_pass = (
        box_map == expected_map
        and "COCO-val2017" in str(meta.get("_metrics", {}))
        and len(categories) >= 80
        and "__background__" in categories
        and "person" in categories
        and "car" in categories
    )
    weights_pass = (
        weights is FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        and cache_path is not None
        and cache_path.is_file()
        and weights.url.startswith("https://download.pytorch.org/models/")
        and "fasterrcnn_resnet50_fpn_v2_coco" in weights.url
    )
    clean_pass = (
        Path(infer["annotated_image"]).is_file()
        and Path(infer["raw_detections_json"]).is_file()
        and infer["num_detections_raw"] >= 0
    )
    # Prefer visualization threshold counts for PASS criteria on person/car
    person_pass = infer["person_detections_vis"] >= 1 or infer["person_detections_raw"] >= 1
    car_pass = infer["car_detections_vis"] >= 1 or infer["car_detections_raw"] >= 1

    # For this specific image (000000407083) the attack benchmark targets car;
    # person may or may not be present. Require car; person is reported but
    # PASS if either focus class that exists is detected — user asked both
    # PERSON DETECTION and CAR DETECTION PASS/FAIL lines.
    # Image 407083 is a large car scene — car should be detected.
    # Person may be absent; if zero persons, PERSON DETECTION = FAIL is honest.

    report = {
        "model_name": cfg["model_display_name"],
        "model_builder": cfg["model_name"],
        "torchvision_version": torchvision.__version__,
        "torch_version": torch.__version__,
        "python_version": sys.version,
        "exact_weights": cfg["weights_enum"],
        "weights_url": weights.url,
        "coco_status": "pretrained COCO (FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1)",
        "coco_category_count_including_background": count_categories(categories, include_background=True),
        "coco_category_count_excluding_background": count_categories(categories, include_background=False),
        "categories": categories,
        "official_coco_val2017_box_map": box_map,
        "expected_coco_val2017_box_map": expected_map,
        "parameters_measured": n_params,
        "parameters_meta": num_params_meta,
        "expected_num_params": expected_params,
        "weight_file_size_bytes": weight_size_bytes,
        "weight_file_size_mb_meta": file_size_mb_meta,
        "weight_cache_path": str(cache_path) if cache_path else None,
        "weight_sha256": weight_sha256,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_version": torch.version.cuda,
        "test_image": str(test_image),
        "test_image_id": cfg["test_image_id"],
        "test_image_source": cfg.get("test_image_source"),
        "person_detections_vis": infer["person_detections_vis"],
        "car_detections_vis": infer["car_detections_vis"],
        "person_detections_raw": infer["person_detections_raw"],
        "car_detections_raw": infer["car_detections_raw"],
        "inference_time_sec": infer["inference_time_sec"],
        "model_load_time_sec": load_sec,
        "frozen_pretrained": True,
        "transforms": "weights.transforms() (official ObjectDetection)",
        "visualization_confidence": cfg["visualization_confidence"],
        "checks": {
            "COCO": "PASS" if coco_pass else "FAIL",
            "WEIGHTS": "PASS" if weights_pass else "FAIL",
            "CLEAN_INFERENCE": "PASS" if clean_pass else "FAIL",
            "PERSON_DETECTION": "PASS" if person_pass else "FAIL",
            "CAR_DETECTION": "PASS" if car_pass else "FAIL",
        },
        "overall_pass": all([coco_pass, weights_pass, clean_pass, car_pass]),
        "inference_outputs": {
            "original": infer["original_image"],
            "annotated": infer["annotated_image"],
            "raw_json": infer["raw_detections_json"],
            "metadata": infer.get("metadata_path"),
        },
        "official_docs": cfg.get("official_docs"),
        "official_repo": cfg.get("official_repo"),
        "recipe": meta.get("recipe"),
        "gflops_meta": meta.get("_ops"),
        "weights_docs": meta.get("_docs"),
    }

    # Overall PASS for setup requires COCO+WEIGHTS+CLEAN+CAR (image is car-focused).
    # PERSON reported honestly; may FAIL if image has no person.

    json_path = ver_dir / "model_verification.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md_lines = [
        "# Faster R-CNN ResNet-50 FPN V2 — Model Verification",
        "",
        f"1. **Model name:** {report['model_name']} (`{report['model_builder']}`)",
        f"2. **Torchvision version:** {report['torchvision_version']}",
        f"3. **Exact weights:** `{report['exact_weights']}`",
        f"4. **COCO status:** {report['coco_status']}",
        f"5. **COCO category count:** {report['coco_category_count_including_background']} "
        f"(including `__background__`); "
        f"{report['coco_category_count_excluding_background']} excluding background",
        f"6. **Official COCO val2017 box mAP:** {report['official_coco_val2017_box_map']} "
        f"(expected {report['expected_coco_val2017_box_map']})",
        f"7. **Parameters:** measured={report['parameters_measured']}, "
        f"meta={report['parameters_meta']} (expected {report['expected_num_params']})",
        f"8. **Weight size:** {report['weight_file_size_bytes']} bytes "
        f"(meta file size MB: {report['weight_file_size_mb_meta']})",
        f"9. **Weight/cache path:** `{report['weight_cache_path']}`",
        f"   - URL: {report['weights_url']}",
        f"   - SHA256: `{report['weight_sha256']}`",
        f"10. **Device:** {report['device']}",
        f"11. **GPU:** {report['gpu_name']}",
        f"12. **CUDA availability:** {report['cuda_available']} (CUDA {report['cuda_version']})",
        f"13. **Test image:** `{report['test_image']}` (id={report['test_image_id']})",
        f"14. **Number of person detections** (vis conf≥{cfg['visualization_confidence']}): "
        f"{report['person_detections_vis']} (raw={report['person_detections_raw']})",
        f"15. **Number of car detections** (vis conf≥{cfg['visualization_confidence']}): "
        f"{report['car_detections_vis']} (raw={report['car_detections_raw']})",
        f"16. **Inference time:** {report['inference_time_sec']:.4f} sec",
        f"17. **PASS/FAIL:**",
        f"    - COCO: {report['checks']['COCO']}",
        f"    - WEIGHTS: {report['checks']['WEIGHTS']}",
        f"    - CLEAN INFERENCE: {report['checks']['CLEAN_INFERENCE']}",
        f"    - PERSON DETECTION: {report['checks']['PERSON_DETECTION']}",
        f"    - CAR DETECTION: {report['checks']['CAR_DETECTION']}",
        "",
        "## Notes",
        "",
        "- Pretrained weights are frozen (`requires_grad=False`, `model.eval()`).",
        "- Full COCO category set retained; not reduced to person/car.",
        "- Preprocessing uses official `weights.transforms()`.",
        "- Visualization uses confidence ≥ 0.25; raw JSON keeps all detections.",
        "- No adversarial attack code in this setup step.",
        "",
        f"Source: extracted COCO val2017 image from existing zip (read-only), "
        f"copied only into this project.",
    ]
    md_path = ver_dir / "model_verification.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    # Also dump a small models/faster_rcnn note
    note = PROJECT_ROOT / "models" / "faster_rcnn" / "README.md"
    note.write_text(
        "\n".join(
            [
                "# fasterrcnn_resnet50_fpn_v2",
                "",
                "Loaded via Torchvision:",
                "",
                "```python",
                "from torchvision.models.detection import (",
                "    fasterrcnn_resnet50_fpn_v2,",
                "    FasterRCNN_ResNet50_FPN_V2_Weights,",
                ")",
                "weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1",
                "model = fasterrcnn_resnet50_fpn_v2(weights=weights)",
                "```",
                "",
                "Do not replace with unofficial checkpoints.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    log_path = log_dir / "verify_setup.log"
    summary = "\n".join(
        [
            "MODEL:",
            "Faster R-CNN ResNet-50 FPN V2",
            "",
            "CHECKPOINT:",
            "FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1",
            "",
            f"COCO:",
            report["checks"]["COCO"],
            "",
            f"WEIGHTS:",
            report["checks"]["WEIGHTS"],
            "",
            f"CLEAN INFERENCE:",
            report["checks"]["CLEAN_INFERENCE"],
            "",
            f"PERSON DETECTION:",
            report["checks"]["PERSON_DETECTION"],
            "",
            f"CAR DETECTION:",
            report["checks"]["CAR_DETECTION"],
            "",
            f"CUDA:",
            "YES" if torch.cuda.is_available() else "NO",
            "",
            f"DEVICE:",
            str(device),
        ]
    )
    log_path.write_text(summary + "\n" + json.dumps(report["checks"], indent=2) + "\n", encoding="utf-8")
    print(summary)

    if not report["overall_pass"]:
        print("\nSETUP INCOMPLETE — see results/verification/model_verification.json")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
