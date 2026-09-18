"""
Hyper-YOLO → Deformable DETR R50 transfer attack evaluation.

Reads ONLY from local source_data/ copies.
Writes ONLY under Deformable-DETR-R50-COCO-Attack/results/.
No combined_analysis. No patch generation/optimization.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import statistics
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont

PROJECT = Path(r"D:\project CS\Deformable-DETR-R50-COCO-Attack").resolve()
sys.path.insert(0, str(PROJECT / "scripts"))

from evaluation_protocol import (  # noqa: E402
    CONF_THRESHOLD,
    IOU_THRESHOLD,
    gt_xywh_to_xyxy,
    match_target,
)
from load_model import HF_MODEL_ID, LOCAL_WEIGHTS_DIR, load_deformable_detr_r50  # noqa: E402

ATTACK_IDS = [f"attack_0{i}" for i in range(1, 6)]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _font(size: int = 14):
    for p in (r"C:\Windows\Fonts\arialbd.ttf", r"C:\Windows\Fonts\arial.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def load_selected(attack_id: str) -> list[dict[str, str]]:
    path = PROJECT / "source_data" / attack_id / "metadata" / "selected_images.csv"
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_pair_verified(attack_id: str) -> set[str]:
    path = PROJECT / "results" / "source_verification" / f"{attack_id}_pair_verification.csv"
    ok = set()
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("pair_verified", "").lower() in {"yes", "true", "1"}:
                ok.add(row["clean_filename"])
    return ok


def ensure_local_checkpoint() -> None:
    """Download SenseTime checkpoint into project weights/ for reproducibility."""
    if (LOCAL_WEIGHTS_DIR / "config.json").is_file():
        return
    from transformers import AutoImageProcessor, DeformableDetrForObjectDetection

    LOCAL_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {HF_MODEL_ID} → {LOCAL_WEIGHTS_DIR}")
    processor = AutoImageProcessor.from_pretrained(HF_MODEL_ID)
    model = DeformableDetrForObjectDetection.from_pretrained(HF_MODEL_ID)
    processor.save_pretrained(LOCAL_WEIGHTS_DIR)
    model.save_pretrained(LOCAL_WEIGHTS_DIR)
    print("Checkpoint cached locally.")


def run_detector(model, processor, device, image_path: Path, id2label, coco_cat_to_contiguous):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    t0 = time.perf_counter()
    with torch.inference_mode():
        outputs = model(**inputs)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    target_sizes = torch.tensor([image.size[::-1]], device=device)  # (h, w)
    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=CONF_THRESHOLD
    )[0]

    dets = []
    scores = results["scores"].detach().cpu()
    labels = results["labels"].detach().cpu()
    boxes = results["boxes"].detach().cpu()
    for i in range(scores.shape[0]):
        coco_cat = int(labels[i].item())
        name = str(id2label.get(coco_cat, id2label.get(str(coco_cat), str(coco_cat))))
        if name.lower() == "n/a":
            continue
        conf = float(scores[i].item())
        x1, y1, x2, y2 = [float(v) for v in boxes[i].tolist()]
        contig = coco_cat_to_contiguous.get(coco_cat)
        dets.append(
            {
                "class_id": int(contig) if contig is not None else coco_cat,
                "coco_category_id": coco_cat,
                "class_name": name,
                "confidence": conf,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "bbox": [x1, y1, x2, y2],
            }
        )
    return dets, elapsed_ms


def save_predictions_json(path: Path, image_name: str, dets: list[dict], inference_time_ms: float) -> None:
    slim = [
        {
            "image_name": image_name,
            "class_id": d["class_id"],
            "class_name": d["class_name"],
            "confidence": d["confidence"],
            "x1": d["x1"],
            "y1": d["y1"],
            "x2": d["x2"],
            "y2": d["y2"],
            "coco_category_id": d.get("coco_category_id"),
        }
        for d in dets
    ]
    path.write_text(
        json.dumps(
            {"image_name": image_name, "inference_time_ms": inference_time_ms, "detections": slim},
            indent=2,
        ),
        encoding="utf-8",
    )


def draw_panel(image, dets, gt_xyxy, match, *, title: str, status: str):
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    font = _font(14)
    gx1, gy1, gx2, gy2 = gt_xyxy
    draw.rectangle([gx1, gy1, gx2, gy2], outline=(255, 220, 0), width=3)
    draw.text((gx1 + 2, max(0, gy1 - 16)), "GT", fill=(255, 220, 0), font=font)
    for d in dets:
        if d["confidence"] < CONF_THRESHOLD:
            continue
        if match.get("bbox") and d["bbox"] == match["bbox"]:
            continue
        name = str(d["class_name"]).lower()
        color = (0, 180, 255) if name in {"person", "car"} else (80, 140, 255)
        x1, y1, x2, y2 = d["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        draw.text((x1 + 2, max(0, y1 - 14)), f"{d['class_name']} {d['confidence']:.2f}", fill=color, font=font)
    if match.get("detected") and match.get("bbox"):
        x1, y1, x2, y2 = match["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline=(0, 200, 0), width=3)
        label = f"{match['class_name']} {match['confidence']:.2f} IoU={match['iou']:.2f}"
        draw.text((x1 + 2, max(0, y1 - 16)), label, fill=(0, 200, 0), font=font)
    draw.text((8, 8), f"{title} | {status}", fill=(255, 255, 255), font=font)
    return out


def save_side_by_side(clean_img, patch_img, clean_dets, patch_dets, gt, clean_match, patch_match, out_path: Path, fn: str):
    left = draw_panel(
        clean_img,
        clean_dets,
        gt,
        clean_match,
        title="CLEAN",
        status="VALID" if clean_match["detected"] else "MISS / INVALID",
    )
    right = draw_panel(
        patch_img,
        patch_dets,
        gt,
        patch_match,
        title="PATCHED",
        status="VALID" if patch_match["detected"] else "MISS / INVALID",
    )
    # Match heights
    h = max(left.height, right.height)
    canvas = Image.new("RGB", (left.width + right.width + 8, h), (20, 20, 20))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width + 8, 0))
    canvas.save(out_path / fn, quality=92)


def summarize_rows(rows: list[dict], attack_id: str, meta: dict) -> dict[str, Any]:
    eligible = [r for r in rows if r["eligible"] is True]
    successes = [r for r in eligible if r["attack_success"] is True]
    both_valid = [r for r in rows if r["both_valid"] is True]

    def class_stats(cls: str):
        e = [r for r in eligible if r["target_class"] == cls]
        s = [r for r in e if r["attack_success"] is True]
        return len(e), len(s), (len(s) / len(e) * 100.0) if e else 0.0

    pe, ps, pa = class_stats("person")
    ce, cs, ca = class_stats("car")
    conf_drops = [float(r["confidence_drop"]) for r in both_valid if r["confidence_drop"] is not None]
    iou_drops = [float(r["iou_drop"]) for r in both_valid if r["iou_drop"] is not None]
    clean_counts = [int(r["clean_prediction_count"]) for r in rows]
    patch_counts = [int(r["patched_prediction_count"]) for r in rows]
    clean_ms = [float(r["inference_time_clean_ms"]) for r in rows]
    patch_ms = [float(r["inference_time_patched_ms"]) for r in rows]

    return {
        "model": "Deformable DETR R50",
        "backbone": "ResNet-50",
        "variant": meta.get("variant"),
        "checkpoint": meta.get("checkpoint_source"),
        "dataset": "MS COCO 2017",
        "confidence_threshold": CONF_THRESHOLD,
        "target_iou_threshold": IOU_THRESHOLD,
        "attack_id": attack_id,
        "total_images": len(rows),
        "eligible": len(eligible),
        "success": len(successes),
        "asr": (len(successes) / len(eligible) * 100.0) if eligible else 0.0,
        "person_eligible": pe,
        "person_success": ps,
        "person_asr": pa,
        "car_eligible": ce,
        "car_success": cs,
        "car_asr": ca,
        "both_valid_n": len(both_valid),
        "mean_confidence_drop": statistics.fmean(conf_drops) if conf_drops else None,
        "median_confidence_drop": statistics.median(conf_drops) if conf_drops else None,
        "mean_iou_drop": statistics.fmean(iou_drops) if iou_drops else None,
        "median_iou_drop": statistics.median(iou_drops) if iou_drops else None,
        "mean_clean_detections": statistics.fmean(clean_counts) if clean_counts else 0.0,
        "mean_patched_detections": statistics.fmean(patch_counts) if patch_counts else 0.0,
        "mean_clean_inference_ms": statistics.fmean(clean_ms) if clean_ms else None,
        "mean_patched_inference_ms": statistics.fmean(patch_ms) if patch_ms else None,
        "device": meta.get("device"),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "implementation_note": meta.get("implementation"),
    }


def write_summary_result_txt(summary: dict, path: Path) -> None:
    def fmt(x):
        return "N/A" if x is None else f"{x:.6f}"

    interp = (
        f"On {summary['total_images']} images, {summary['eligible']} targets were eligible "
        f"(clean-valid). Of these, {summary['success']} became invalid after the Hyper-YOLO patch "
        f"(ASR={summary['asr']:.2f}%). Person ASR={summary['person_asr']:.2f}%; "
        f"car ASR={summary['car_asr']:.2f}%. Among {summary['both_valid_n']} both-valid targets, "
        f"mean confidence drop={fmt(summary['mean_confidence_drop'])}, "
        f"mean IoU drop={fmt(summary['mean_iou_drop'])}. "
        "These results measure transfer of existing Hyper-YOLO adversarial images to Deformable DETR R50; "
        "patches were not optimized for this detector."
    )
    lines = [
        "========================================================",
        "Deformable DETR R50 Transfer Attack Summary",
        "========================================================",
        "",
        f"Attack ID: {summary['attack_id']}",
        "Model: Deformable DETR R50",
        "Dataset: COCO 2017",
        "Target classes: Person, Car",
        "Confidence threshold: 0.25",
        "Target-match IoU threshold: 0.50",
        "",
        f"Total images: {summary['total_images']}",
        f"Eligible targets: {summary['eligible']}",
        f"Successful attacks: {summary['success']}",
        f"Attack Success Rate: {summary['asr']:.2f}%",
        "",
        "Person:",
        f"Eligible: {summary['person_eligible']}",
        f"Success: {summary['person_success']}",
        f"ASR: {summary['person_asr']:.2f}%",
        "",
        "Car:",
        f"Eligible: {summary['car_eligible']}",
        f"Success: {summary['car_success']}",
        f"ASR: {summary['car_asr']:.2f}%",
        "",
        f"Mean confidence drop: {fmt(summary['mean_confidence_drop'])}",
        f"Median confidence drop: {fmt(summary['median_confidence_drop'])}",
        f"Mean IoU drop: {fmt(summary['mean_iou_drop'])}",
        f"Median IoU drop: {fmt(summary['median_iou_drop'])}",
        f"Both-valid n: {summary['both_valid_n']}",
        "",
        f"Average clean detections/image: {summary['mean_clean_detections']:.4f}",
        f"Average patched detections/image: {summary['mean_patched_detections']:.4f}",
        "",
        f"Average clean inference time: {fmt(summary['mean_clean_inference_ms'])} ms",
        f"Average patched inference time: {fmt(summary['mean_patched_inference_ms'])} ms",
        "",
        "========================================================",
        "",
        "Interpretation:",
        interp,
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_report_md(summary: dict, path: Path) -> None:
    def fmt(x):
        return "N/A" if x is None else f"{x:.6f}"

    lines = [
        "# Deformable DETR R50 Transfer Attack Evaluation",
        "",
        "## 1. Model information",
        "",
        "- Model: Deformable DETR R50 (standard/base multi-scale)",
        "- Backbone: ResNet-50",
        "- Config correspondence: `configs/r50_deformable_detr.sh`",
        "- Official COCO box AP: 44.5",
        "- Paper: https://arxiv.org/abs/2010.04159",
        "- GitHub: https://github.com/fundamentalvision/Deformable-DETR",
        f"- Checkpoint: {summary['checkpoint']}",
        "- Variant flags: two_stage=False, with_box_refine=False, num_feature_levels=4",
        "",
        "## 2. Dataset information",
        "",
        "- MS COCO 2017",
        "- Target classes: person, car",
        "- Selected GT boxes from Hyper-YOLO attack `selected_images.csv`",
        "",
        "## 3. Transfer setup",
        "",
        "The attack images were generated previously for Hyper-YOLO and are being evaluated here "
        "as transferred adversarial examples against Deformable DETR R50. "
        "These patches were **not** optimized for Deformable DETR.",
        "",
        "## 4. Evaluation protocol",
        "",
        "- Confidence >= 0.25",
        "- IoU >= 0.50 vs selected GT (highest-IoU same-class match; confidence tie-break)",
        "- Eligible = clean target valid",
        "- Success = clean valid AND patched invalid",
        "- Confidence/IoU drops = both-valid only",
        "",
        "## 5. Attack-wise results",
        "",
        f"| Metric | Value |",
        f"| ------ | ----: |",
        f"| Total images | {summary['total_images']} |",
        f"| Eligible | {summary['eligible']} |",
        f"| Success | {summary['success']} |",
        f"| ASR (%) | {summary['asr']:.2f} |",
        "",
        "## 6. Person vs Car",
        "",
        "| Class | Eligible | Success | ASR (%) |",
        "| ----- | -------: | ------: | ------: |",
        f"| Person | {summary['person_eligible']} | {summary['person_success']} | {summary['person_asr']:.2f} |",
        f"| Car | {summary['car_eligible']} | {summary['car_success']} | {summary['car_asr']:.2f} |",
        "",
        "## 7. Confidence analysis",
        "",
        f"- Both-valid n: {summary['both_valid_n']}",
        f"- Mean confidence drop: {fmt(summary['mean_confidence_drop'])}",
        f"- Median confidence drop: {fmt(summary['median_confidence_drop'])}",
        "",
        "## 8. IoU analysis",
        "",
        f"- Mean IoU drop: {fmt(summary['mean_iou_drop'])}",
        f"- Median IoU drop: {fmt(summary['median_iou_drop'])}",
        "",
        "## 9. Important observations",
        "",
        f"ASR={summary['asr']:.2f}% on {summary['eligible']} eligible targets "
        f"({summary['success']} successes). Car ASR={summary['car_asr']:.2f}%, "
        f"person ASR={summary['person_asr']:.2f}%.",
        "",
        "## 10. Limitations",
        "",
        "- Transfer evaluation only; no white-box attack on Deformable DETR.",
        "- Official CUDA ms_deform_attn was not compiled on this Windows host; "
        "SenseTime weights were loaded via Transformers (same standard R50 architecture).",
        "",
        "## 11. Reproducibility",
        "",
        f"- Device: {summary['device']}",
        f"- Python: {summary['python_version']}",
        f"- PyTorch: {summary['torch_version']}",
        f"- Torchvision: {summary['torchvision_version']}",
        f"- Note: {summary.get('implementation_note')}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_per_image_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "image_name",
        "attack_id",
        "target_class",
        "gt_x1",
        "gt_y1",
        "gt_x2",
        "gt_y2",
        "clean_valid",
        "patched_valid",
        "clean_confidence",
        "patched_confidence",
        "confidence_drop",
        "clean_iou",
        "patched_iou",
        "iou_drop",
        "attack_success",
        "clean_prediction_count",
        "patched_prediction_count",
        "clean_target_class",
        "patched_target_class",
        "inference_time_clean_ms",
        "inference_time_patched_ms",
        "both_valid",
        "eligible",
        "status",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            out = {k: ("" if r.get(k) is None else r.get(k)) for k in fields}
            w.writerow(out)


def process_attack(
    attack_id: str,
    model,
    processor,
    device,
    id2label,
    coco_cat_to_contiguous,
    meta,
    *,
    filenames_filter: set[str] | None = None,
    write_visuals: bool = True,
    failure_log: list[dict] | None = None,
) -> list[dict]:
    selected = load_selected(attack_id)
    verified = load_pair_verified(attack_id)
    out_root = PROJECT / "results" / attack_id
    clean_dir = out_root / "clean_predictions"
    patch_dir = out_root / "patched_predictions"
    vis_dir = out_root / "visualizations"
    for d in (clean_dir, patch_dir, vis_dir):
        d.mkdir(parents=True, exist_ok=True)

    rows = []
    for item in selected:
        fn = item["filename"]
        if filenames_filter is not None and fn not in filenames_filter:
            continue
        if fn not in verified:
            print(f"  SKIP unresolved {attack_id}/{fn}")
            continue
        clean_path = PROJECT / "source_data" / attack_id / "clean_images" / fn
        patch_path = PROJECT / "source_data" / attack_id / "patched_images" / fn
        if not clean_path.is_file() or not patch_path.is_file():
            msg = f"missing copy {attack_id}/{fn}"
            print(f"  SKIP {msg}")
            if failure_log is not None:
                failure_log.append({"attack_id": attack_id, "filename": fn, "stage": "io", "reason": msg})
            continue
        try:
            target_class = item["target_class"]
            gt = gt_xywh_to_xyxy(
                float(item["bbox_x"]),
                float(item["bbox_y"]),
                float(item["bbox_width"]),
                float(item["bbox_height"]),
            )
            clean_dets, clean_ms = run_detector(
                model, processor, device, clean_path, id2label, coco_cat_to_contiguous
            )
            save_predictions_json(clean_dir / f"{Path(fn).stem}.json", fn, clean_dets, clean_ms)
            clean_match = match_target(clean_dets, target_class, gt)
            eligible = bool(clean_match["detected"])

            patch_dets, patch_ms = run_detector(
                model, processor, device, patch_path, id2label, coco_cat_to_contiguous
            )
            save_predictions_json(patch_dir / f"{Path(fn).stem}.json", fn, patch_dets, patch_ms)
            patch_match = match_target(patch_dets, target_class, gt)
            attack_success = bool(eligible and (not patch_match["detected"]))
            both_valid = bool(clean_match["detected"] and patch_match["detected"])
            if both_valid:
                conf_drop = float(clean_match["confidence"]) - float(patch_match["confidence"])
                iou_drop = float(clean_match["iou"]) - float(patch_match["iou"])
            else:
                conf_drop = None
                iou_drop = None

            row = {
                "image_name": fn,
                "attack_id": attack_id,
                "target_class": target_class,
                "gt_x1": gt[0],
                "gt_y1": gt[1],
                "gt_x2": gt[2],
                "gt_y2": gt[3],
                "clean_valid": clean_match["detected"],
                "patched_valid": patch_match["detected"],
                "clean_confidence": clean_match["confidence"],
                "patched_confidence": patch_match["confidence"],
                "confidence_drop": conf_drop,
                "clean_iou": clean_match["iou"],
                "patched_iou": patch_match["iou"],
                "iou_drop": iou_drop,
                "attack_success": attack_success,
                "clean_prediction_count": len(clean_dets),
                "patched_prediction_count": len(patch_dets),
                "clean_target_class": clean_match["class_name"],
                "patched_target_class": patch_match["class_name"],
                "inference_time_clean_ms": clean_ms,
                "inference_time_patched_ms": patch_ms,
                "both_valid": both_valid,
                "eligible": eligible,
                "status": "ok",
            }
            rows.append(row)

            if write_visuals:
                clean_img = Image.open(clean_path).convert("RGB")
                patch_img = Image.open(patch_path).convert("RGB")
                save_side_by_side(
                    clean_img, patch_img, clean_dets, patch_dets, gt, clean_match, patch_match, vis_dir, fn
                )

            print(
                f"  {attack_id}/{fn} clean={clean_match['detected']} patched={patch_match['detected']} "
                f"eligible={eligible} success={attack_success}"
            )
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            print(f"  FAIL {attack_id}/{fn}: {err}")
            if failure_log is not None:
                failure_log.append(
                    {
                        "attack_id": attack_id,
                        "filename": fn,
                        "stage": "inference",
                        "reason": err,
                        "traceback": traceback.format_exc(),
                    }
                )
            continue
    return rows


def write_experiment_config(device, meta) -> None:
    cfg = {
        "experiment_name": "Hyper-YOLO-to-Deformable-DETR-R50 Transfer Attack",
        "date_time_utc": datetime.now(timezone.utc).isoformat(),
        "model": "Deformable DETR R50",
        "backbone": "ResNet-50",
        "variant": meta.get("variant"),
        "official_config": "configs/r50_deformable_detr.sh",
        "official_coco_AP": 44.5,
        "paper": "https://arxiv.org/abs/2010.04159",
        "github": "https://github.com/fundamentalvision/Deformable-DETR",
        "checkpoint_source": meta.get("checkpoint_source"),
        "checkpoint_local": meta.get("checkpoint_local"),
        "checkpoint_sha256": meta.get("checkpoint_sha256"),
        "confidence_threshold": CONF_THRESHOLD,
        "iou_threshold": IOU_THRESHOLD,
        "target_classes": ["person", "car"],
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "device": str(device),
        "os": platform.platform(),
        "implementation": meta.get("implementation"),
        "do_not_regenerate_patches": True,
        "fine_tune": False,
        "create_combined_analysis": False,
        "coco_annotations": r"D:\project CS\Hyper-YOLO\coco\annotations\instances_val2017.json",
        "coco_val2017": r"D:\project CS\Hyper-YOLO\coco\val2017\val2017",
        "gt_source": "attack selected_images.csv (selected COCO GT bbox)",
    }
    (PROJECT / "logs" / "experiment_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def verify_integrity() -> bool:
    before_path = PROJECT / "results" / "source_integrity_hashes_before.json"
    before = json.loads(before_path.read_text(encoding="utf-8"))
    changed = []
    for path_str, meta in before.items():
        p = Path(path_str)
        if not p.is_file():
            changed.append({"path": path_str, "reason": "missing"})
            continue
        digest = sha256_file(p)
        if digest != meta["sha256"] or p.stat().st_size != meta["size_bytes"]:
            changed.append({"path": path_str, "reason": "changed"})
    report = {
        "original_source_project": "UNCHANGED" if not changed else "INTEGRITY FAILURE",
        "files_checked": len(before),
        "changed_count": len(changed),
        "changed": changed,
    }
    (PROJECT / "results" / "source_integrity_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return len(changed) == 0


def print_startup(device, meta) -> None:
    print("========== STARTUP ==========")
    print("Python version:", sys.version.split()[0])
    print("PyTorch version:", torch.__version__)
    print("Torchvision version:", torchvision.__version__)
    print("CUDA version:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    print("Model:", meta["model"])
    print("Backbone:", meta["backbone"])
    print("Variant:", meta["variant"])
    print("Checkpoint:", meta["checkpoint_source"])
    print("two_stage:", meta["two_stage"], "with_box_refine:", meta["with_box_refine"])
    print("num_feature_levels:", meta["num_feature_levels"])
    print("Device:", device)
    print("Implementation:", meta["implementation"])
    print("================================")


def finalize_attack(rows, attack_id, meta):
    out_root = PROJECT / "results" / attack_id
    write_per_image_csv(out_root / "per_image_results.csv", rows)
    summary = summarize_rows(rows, attack_id, meta)
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_summary_result_txt(summary, out_root / "summary_result.txt")
    write_report_md(summary, out_root / "report.md")
    print(
        f"{attack_id} | Total={summary['total_images']} Eligible={summary['eligible']} "
        f"Success={summary['success']} ASR={summary['asr']:.2f}% "
        f"PersonASR={summary['person_asr']:.2f}% CarASR={summary['car_asr']:.2f}% "
        f"MeanConfDrop={summary['mean_confidence_drop']} MedianConfDrop={summary['median_confidence_drop']} "
        f"MeanIoUDrop={summary['mean_iou_drop']} MedianIoUDrop={summary['median_iou_drop']} "
        f"BothValidN={summary['both_valid_n']}"
    )
    return summary


def smoke_test(model, processor, device, id2label, coco_cat_to_contiguous, meta) -> bool:
    print("=== SMOKE TEST attack_01 (1 pair) ===")
    assert str(id2label[1]).lower() == "person"
    assert str(id2label[3]).lower() == "car"
    selected = load_selected("attack_01")
    verified = load_pair_verified("attack_01")
    picks = [r["filename"] for r in selected if r["filename"] in verified][:1]
    if not picks:
        print("SMOKE FAIL: no verified pair")
        return False
    failures = []
    rows = process_attack(
        "attack_01",
        model,
        processor,
        device,
        id2label,
        coco_cat_to_contiguous,
        meta,
        filenames_filter=set(picks),
        write_visuals=True,
        failure_log=failures,
    )
    stem = Path(picks[0]).stem
    det_path = PROJECT / "results" / "attack_01" / "clean_predictions" / f"{stem}.json"
    if det_path.is_file():
        payload = json.loads(det_path.read_text(encoding="utf-8"))
        print("Sample clean detections (person/car):")
        for d in payload["detections"]:
            if d["class_name"] in {"person", "car"}:
                print(f"  {d['class_name']} {d['confidence']:.2f} (class_id={d['class_id']})")
    ok = (
        len(rows) == 1
        and not failures
        and det_path.is_file()
        and (PROJECT / "results" / "attack_01" / "patched_predictions" / f"{stem}.json").is_file()
        and (PROJECT / "results" / "attack_01" / "visualizations" / picks[0]).is_file()
    )
    print("SMOKE", "PASS" if ok else "FAIL", picks)
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack", default="all", help="attack_01..05, all, or smoke")
    args = parser.parse_args()

    t0 = time.perf_counter()
    ensure_local_checkpoint()
    model, processor, device, id2label, coco_cat_to_contiguous, _contig_names, meta = load_deformable_detr_r50()
    print_startup(device, meta)
    write_experiment_config(device, meta)

    if args.attack == "smoke":
        return 0 if smoke_test(model, processor, device, id2label, coco_cat_to_contiguous, meta) else 1

    attack_ids = ATTACK_IDS if args.attack == "all" else [args.attack]
    all_failures: list[dict] = []
    summaries = []
    for aid in attack_ids:
        print(f"\n===== {aid} =====")
        rows = process_attack(
            aid,
            model,
            processor,
            device,
            id2label,
            coco_cat_to_contiguous,
            meta,
            write_visuals=True,
            failure_log=all_failures,
        )
        summaries.append(finalize_attack(rows, aid, meta))

    (PROJECT / "logs" / "inference_failures.json").write_text(
        json.dumps(
            {
                "successful_inference_count": sum(s["total_images"] for s in summaries),
                "failed_inference_count": len(all_failures),
                "failures": all_failures,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    ok = verify_integrity()

    tot_images = sum(s["total_images"] for s in summaries)
    tot_elig = sum(s["eligible"] for s in summaries)
    tot_succ = sum(s["success"] for s in summaries)
    tot_pe = sum(s["person_eligible"] for s in summaries)
    tot_ps = sum(s["person_success"] for s in summaries)
    tot_ce = sum(s["car_eligible"] for s in summaries)
    tot_cs = sum(s["car_success"] for s in summaries)
    both_sum = sum(s["both_valid_n"] for s in summaries)

    conf_num = iou_num = 0.0
    weight = 0
    medians_conf = []
    medians_iou = []
    for s in summaries:
        n = s["both_valid_n"]
        if n and s["mean_confidence_drop"] is not None:
            conf_num += float(s["mean_confidence_drop"]) * n
            iou_num += float(s["mean_iou_drop"]) * n
            weight += n
        if s["median_confidence_drop"] is not None:
            medians_conf.append(float(s["median_confidence_drop"]))
        if s["median_iou_drop"] is not None:
            medians_iou.append(float(s["median_iou_drop"]))

    print("\n========== Deformable DETR R50 ==========")
    for s in summaries:
        print(f"{s['attack_id']} = ASR {s['asr']:.2f}% ({s['success']}/{s['eligible']})")
    overall_asr = (tot_succ / tot_elig * 100.0) if tot_elig else 0.0
    print()
    print(f"Overall total: {tot_images}")
    print(f"Overall eligible: {tot_elig}")
    print(f"Overall success: {tot_succ}")
    print(f"Overall ASR: {overall_asr:.2f}%")
    print()
    print(f"Overall person: {tot_ps}/{tot_pe}")
    print(f"Overall person ASR: {(tot_ps / tot_pe * 100.0) if tot_pe else 0.0:.2f}%")
    print()
    print(f"Overall car: {tot_cs}/{tot_ce}")
    print(f"Overall car ASR: {(tot_cs / tot_ce * 100.0) if tot_ce else 0.0:.2f}%")
    print()
    print(f"Weighted mean confidence drop: {(conf_num / weight) if weight else None}")
    print(
        f"Median of attack-level median confidence drops: "
        f"{statistics.median(medians_conf) if medians_conf else None}"
    )
    print(f"Weighted mean IoU drop: {(iou_num / weight) if weight else None}")
    print(
        f"Median of attack-level median IoU drops: "
        f"{statistics.median(medians_iou) if medians_iou else None}"
    )
    print(f"Sum both-valid: {both_sum}")
    print("Failed images:", len(all_failures))
    print("Source integrity:", "PASS" if ok else "FAIL")
    print(f"Elapsed sec: {time.perf_counter() - t0:.1f}")
    print("NOTE: combined_analysis was NOT created.")
    return 0 if ok and not all_failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
