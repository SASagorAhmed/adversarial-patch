"""
Hyper-YOLO-to-Faster-R-CNN Transfer Attack benchmark.

Reads ONLY from local source_data/ copies.
Writes ONLY under Faster-RCNN-COCO-Attack/results/.
"""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont

PROJECT = Path(r"D:\project CS\Faster-RCNN-COCO-Attack").resolve()
sys.path.insert(0, str(PROJECT / "src"))

from evaluation_protocol import (  # noqa: E402
    CONF_THRESHOLD,
    TARGET_MATCH_IOU_THRESHOLD,
    gt_xywh_to_xyxy,
    match_target,
)
from load_model import load_model  # noqa: E402

ATTACK_IDS = [f"attack_0{i}" for i in range(1, 6)]
SOURCE_ROOT = Path(r"D:\project CS\Adversarial-Patch-Experiment\attacks").resolve()


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


def run_detector(model, transforms, device, image_path: Path, categories: list[str]) -> list[dict]:
    image = Image.open(image_path).convert("RGB")
    batch = transforms(image)
    with torch.inference_mode():
        out = model([batch.to(device)])[0]
    if device.type == "cuda":
        torch.cuda.synchronize()
    boxes = out["boxes"].detach().cpu().tolist()
    scores = out["scores"].detach().cpu().tolist()
    labels = out["labels"].detach().cpu().tolist()
    dets = []
    for box, score, label in zip(boxes, scores, labels):
        lid = int(label)
        name = categories[lid] if 0 <= lid < len(categories) else str(lid)
        dets.append(
            {
                "class_id": lid,
                "class_name": name,
                "confidence": float(score),
                "x1": float(box[0]),
                "y1": float(box[1]),
                "x2": float(box[2]),
                "y2": float(box[3]),
                "bbox": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
            }
        )
    return dets


def save_detections_json(path: Path, image_id: str, dets: list[dict]) -> None:
    slim = [
        {
            "class_id": d["class_id"],
            "class_name": d["class_name"],
            "confidence": d["confidence"],
            "x1": d["x1"],
            "y1": d["y1"],
            "x2": d["x2"],
            "y2": d["y2"],
        }
        for d in dets
    ]
    path.write_text(
        json.dumps({"image_id": image_id, "detections": slim}, indent=2),
        encoding="utf-8",
    )


def draw_vis(
    image: Image.Image,
    dets: list[dict],
    gt_xyxy: list[float],
    match: dict,
    *,
    title: str,
    conf_vis: float = CONF_THRESHOLD,
) -> Image.Image:
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    font = _font(14)
    # GT in yellow
    gx1, gy1, gx2, gy2 = gt_xyxy
    draw.rectangle([gx1, gy1, gx2, gy2], outline=(255, 220, 0), width=3)
    draw.text((gx1 + 2, max(0, gy1 - 16)), "GT", fill=(255, 220, 0), font=font)
    # other detections (above vis conf)
    for d in dets:
        if d["confidence"] < conf_vis:
            continue
        if match.get("bbox") and d["bbox"] == match["bbox"]:
            continue
        x1, y1, x2, y2 = d["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline=(80, 140, 255), width=2)
    # matched target in green/red
    if match.get("detected") and match.get("bbox"):
        x1, y1, x2, y2 = match["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline=(0, 200, 0), width=3)
        label = f"{match['class_name']} {match['confidence']:.2f} IoU={match['iou']:.2f}"
        draw.text((x1 + 2, max(0, y1 - 16)), label, fill=(0, 200, 0), font=font)
    draw.text((8, 8), title, fill=(255, 255, 255), font=font)
    return out


def summarize_rows(rows: list[dict], attack_id: str) -> dict[str, Any]:
    eligible = [r for r in rows if r["eligible"] is True]
    successes = [r for r in eligible if r["attack_success"] is True]
    both_valid = [
        r
        for r in eligible
        if r["clean_detected"] is True and r["patched_detected"] is True
        and r["confidence_drop"] is not None
        and r["iou_drop"] is not None
    ]

    def class_stats(cls: str):
        e = [r for r in eligible if r["target_class"] == cls]
        s = [r for r in e if r["attack_success"] is True]
        return {
            "eligible": len(e),
            "successful": len(s),
            "ASR": (len(s) / len(e) * 100.0) if e else 0.0,
        }

    person = class_stats("person")
    car = class_stats("car")
    conf_drops = [float(r["confidence_drop"]) for r in both_valid]
    iou_drops = [float(r["iou_drop"]) for r in both_valid]

    return {
        "attack_id": attack_id,
        "model": "Faster R-CNN ResNet-50 FPN V2",
        "weights": "FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1",
        "dataset": "COCO 2017",
        "attack_source": "Existing Hyper-YOLO adversarial images",
        "attack_type": "Hyper-YOLO-to-Faster-R-CNN Transfer Attack",
        "total_images": len(rows),
        "eligible_targets": len(eligible),
        "successful_attacks": len(successes),
        "denominator": len(eligible),
        "ASR": (len(successes) / len(eligible) * 100.0) if eligible else 0.0,
        "person_eligible": person["eligible"],
        "person_successful": person["successful"],
        "person_ASR": person["ASR"],
        "car_eligible": car["eligible"],
        "car_successful": car["successful"],
        "car_ASR": car["ASR"],
        "mean_confidence_drop": statistics.fmean(conf_drops) if conf_drops else None,
        "median_confidence_drop": statistics.median(conf_drops) if conf_drops else None,
        "mean_IoU_drop": statistics.fmean(iou_drops) if iou_drops else None,
        "n_both_valid_for_drop_metrics": len(both_valid),
        "confidence_drop_note": "mean/median over eligible where BOTH clean and patched matches are valid; misses excluded",
        "confidence_threshold": CONF_THRESHOLD,
        "target_match_iou_threshold": TARGET_MATCH_IOU_THRESHOLD,
    }


def write_attack_report(summary: dict, path: Path) -> None:
    def fmt(x, pct=False):
        if x is None:
            return "N/A (no both-valid pairs)"
        return f"{x:.2f}%" if pct else f"{x:.6f}"

    lines = [
        f"# {summary['attack_id']} — Transfer Attack Report",
        "",
        "Model:",
        "Faster R-CNN ResNet-50 FPN V2",
        "",
        "Weights:",
        "FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1",
        "",
        "Dataset:",
        "COCO 2017",
        "",
        "Attack source:",
        "Existing Hyper-YOLO adversarial images",
        "",
        "Attack type:",
        "Transfer Attack",
        "",
        f"Total images:",
        f"{summary['total_images']}",
        "",
        f"Eligible targets:",
        f"{summary['eligible_targets']}",
        "",
        f"Successful attacks:",
        f"{summary['successful_attacks']}",
        "",
        f"ASR:",
        f"{summary['ASR']:.2f}%",
        "",
        "Person:",
        f"eligible = {summary['person_eligible']}",
        f"successful = {summary['person_successful']}",
        f"ASR = {summary['person_ASR']:.2f}%",
        "",
        "Car:",
        f"eligible = {summary['car_eligible']}",
        f"successful = {summary['car_successful']}",
        f"ASR = {summary['car_ASR']:.2f}%",
        "",
        "Mean confidence drop:",
        fmt(summary["mean_confidence_drop"]),
        "",
        "Median confidence drop:",
        fmt(summary["median_confidence_drop"]),
        "",
        "Mean IoU drop:",
        fmt(summary["mean_IoU_drop"]),
        "",
        f"Note: {summary['confidence_drop_note']}",
        f"(n={summary['n_both_valid_for_drop_metrics']} both-valid eligible pairs)",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_hyper_yolo_summary(path: Path) -> dict[str, float]:
    """Parse existing attack_summary.txt — do not invent missing fields."""
    text = path.read_text(encoding="utf-8")
    out: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("Attack Success Rate:"):
            # "9 / 169 = 5.33%"
            if "=" in line:
                out["ASR"] = float(line.split("=")[-1].replace("%", "").strip())
        elif line.startswith("Person ASR:"):
            out["person_ASR"] = float(line.split(":")[-1].replace("%", "").strip())
        elif line.startswith("Car ASR:"):
            out["car_ASR"] = float(line.split(":")[-1].replace("%", "").strip())
    return out


def process_attack(
    attack_id: str,
    model,
    transforms,
    device,
    categories: list[str],
    *,
    filenames_filter: set[str] | None = None,
    write_visuals: bool = True,
) -> list[dict]:
    selected = load_selected(attack_id)
    verified = load_pair_verified(attack_id)
    out_root = PROJECT / "results" / attack_id
    clean_det_dir = out_root / "clean_detections"
    patch_det_dir = out_root / "patched_detections"
    vis_clean = out_root / "visualizations" / "clean"
    vis_patch = out_root / "visualizations" / "patched"
    for d in (clean_det_dir, patch_det_dir, vis_clean, vis_patch):
        d.mkdir(parents=True, exist_ok=True)

    rows = []
    for item in selected:
        fn = item["filename"]
        if filenames_filter is not None and fn not in filenames_filter:
            continue
        if fn not in verified:
            print(f"  SKIP unresolved pair {attack_id}/{fn}")
            continue

        clean_path = PROJECT / "source_data" / attack_id / "clean_images" / fn
        patch_path = PROJECT / "source_data" / attack_id / "patched_images" / fn
        if not clean_path.is_file() or not patch_path.is_file():
            print(f"  SKIP missing copy {attack_id}/{fn}")
            continue

        image_id = item["image_id"]
        target_class = item["target_class"]
        gt = gt_xywh_to_xyxy(
            float(item["bbox_x"]),
            float(item["bbox_y"]),
            float(item["bbox_width"]),
            float(item["bbox_height"]),
        )

        # STEP A/B — clean first
        clean_dets = run_detector(model, transforms, device, clean_path, categories)
        save_detections_json(clean_det_dir / f"{Path(fn).stem}.json", image_id, clean_dets)
        clean_match = match_target(clean_dets, target_class, gt)
        eligible = bool(clean_match["detected"])

        # STEP D/E — patched
        patch_dets = run_detector(model, transforms, device, patch_path, categories)
        save_detections_json(patch_det_dir / f"{Path(fn).stem}.json", image_id, patch_dets)
        patch_match = match_target(patch_dets, target_class, gt)

        attack_success = bool(eligible and (not patch_match["detected"]))

        # Drops only when both valid
        if clean_match["detected"] and patch_match["detected"]:
            conf_drop = float(clean_match["confidence"]) - float(patch_match["confidence"])
            iou_drop = float(clean_match["iou"]) - float(patch_match["iou"])
        else:
            conf_drop = None
            iou_drop = None

        cb = clean_match["bbox"]
        pb = patch_match["bbox"]
        row = {
            "attack_id": attack_id,
            "image_id": image_id,
            "target_class": target_class,
            "clean_image": str(clean_path),
            "patched_image": str(patch_path),
            "clean_detected": clean_match["detected"],
            "patched_detected": patch_match["detected"],
            "clean_confidence": clean_match["confidence"],
            "patched_confidence": patch_match["confidence"],
            "clean_bbox_x1": cb[0] if cb else None,
            "clean_bbox_y1": cb[1] if cb else None,
            "clean_bbox_x2": cb[2] if cb else None,
            "clean_bbox_y2": cb[3] if cb else None,
            "patched_bbox_x1": pb[0] if pb else None,
            "patched_bbox_y1": pb[1] if pb else None,
            "patched_bbox_x2": pb[2] if pb else None,
            "patched_bbox_y2": pb[3] if pb else None,
            "clean_iou": clean_match["iou"],
            "patched_iou": patch_match["iou"],
            "confidence_drop": conf_drop,
            "iou_drop": iou_drop,
            "eligible": eligible,
            "attack_success": attack_success,
        }
        rows.append(row)

        if write_visuals:
            clean_img = Image.open(clean_path).convert("RGB")
            patch_img = Image.open(patch_path).convert("RGB")
            ctitle = (
                f"CLEAN | {target_class} | eligible={eligible} | "
                f"det={clean_match['detected']} conf={clean_match['confidence']} iou={clean_match['iou']}"
            )
            ptitle = (
                f"PATCHED | {target_class} | success={attack_success} | "
                f"det={patch_match['detected']} conf={patch_match['confidence']} iou={patch_match['iou']}"
            )
            draw_vis(clean_img, clean_dets, gt, clean_match, title=ctitle).save(vis_clean / fn, quality=92)
            draw_vis(patch_img, patch_dets, gt, patch_match, title=ptitle).save(vis_patch / fn, quality=92)

        print(
            f"  {attack_id}/{fn} clean={clean_match['detected']} patched={patch_match['detected']} "
            f"eligible={eligible} success={attack_success}"
        )

    return rows


def write_per_image_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "attack_id",
        "image_id",
        "target_class",
        "clean_image",
        "patched_image",
        "clean_detected",
        "patched_detected",
        "clean_confidence",
        "patched_confidence",
        "clean_bbox_x1",
        "clean_bbox_y1",
        "clean_bbox_x2",
        "clean_bbox_y2",
        "patched_bbox_x1",
        "patched_bbox_y1",
        "patched_bbox_x2",
        "patched_bbox_y2",
        "clean_iou",
        "patched_iou",
        "confidence_drop",
        "iou_drop",
        "eligible",
        "attack_success",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            out = {k: ("" if r.get(k) is None else r.get(k)) for k in fields}
            w.writerow(out)


def write_environment(device: torch.device) -> None:
    info = {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "operating_system": platform.platform(),
        "device": str(device),
    }
    path = PROJECT / "results" / "environment_info.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, indent=2), encoding="utf-8")

    # Enrich experiment_config with runtime env
    cfg_path = PROJECT / "configs" / "experiment_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg.update(
        {
            "device": str(device),
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "cuda": torch.cuda.is_available(),
            "gpu": info["gpu"],
        }
    )
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def verify_integrity() -> tuple[bool, dict]:
    before_path = PROJECT / "results" / "source_integrity_hashes_before.json"
    before = json.loads(before_path.read_text(encoding="utf-8"))
    after = {}
    changed = []
    for path_str, meta in before.items():
        p = Path(path_str)
        if not p.is_file():
            changed.append({"path": path_str, "reason": "missing"})
            continue
        digest = sha256_file(p)
        after[path_str] = {"sha256": digest, "size_bytes": p.stat().st_size, **{k: v for k, v in meta.items() if k not in {"sha256", "size_bytes"}}}
        if digest != meta["sha256"] or p.stat().st_size != meta["size_bytes"]:
            changed.append({"path": path_str, "reason": "hash_or_size_changed", "before": meta["sha256"], "after": digest})
    ok = len(changed) == 0
    report = {
        "original_source_project": "UNCHANGED" if ok else "INTEGRITY FAILURE",
        "files_checked": len(before),
        "changed_count": len(changed),
        "changed": changed,
        "hashes_after": after,
    }
    (PROJECT / "results" / "source_integrity_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = [
        "# Source Integrity Report",
        "",
        f"Original source project:",
        report["original_source_project"],
        "",
        f"Files checked: {report['files_checked']}",
        f"Changed: {report['changed_count']}",
    ]
    if changed:
        md.append("")
        md.append("Changed files:")
        for c in changed:
            md.append(f"- {c}")
    (PROJECT / "results" / "source_integrity_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return ok, report


def write_combined(summaries: list[dict]) -> None:
    comb = PROJECT / "results" / "combined"
    comb.mkdir(parents=True, exist_ok=True)
    fields = [
        "attack_id",
        "total_images",
        "eligible_targets",
        "successful_attacks",
        "ASR",
        "person_eligible",
        "person_successful",
        "person_ASR",
        "car_eligible",
        "car_successful",
        "car_ASR",
        "mean_confidence_drop",
        "median_confidence_drop",
        "mean_IoU_drop",
    ]
    csv_path = comb / "faster_rcnn_transfer_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in summaries:
            w.writerow({k: ("" if s.get(k) is None else s.get(k)) for k in fields})

    # Combined totals
    tot_img = sum(s["total_images"] for s in summaries)
    tot_elig = sum(s["eligible_targets"] for s in summaries)
    tot_succ = sum(s["successful_attacks"] for s in summaries)
    p_e = sum(s["person_eligible"] for s in summaries)
    p_s = sum(s["person_successful"] for s in summaries)
    c_e = sum(s["car_eligible"] for s in summaries)
    c_s = sum(s["car_successful"] for s in summaries)
    confs = [s["mean_confidence_drop"] for s in summaries if s["mean_confidence_drop"] is not None]
    ious = [s["mean_IoU_drop"] for s in summaries if s["mean_IoU_drop"] is not None]
    combined = {
        "summaries": summaries,
        "total_images": tot_img,
        "eligible_targets": tot_elig,
        "successful_attacks": tot_succ,
        "overall_ASR": (tot_succ / tot_elig * 100.0) if tot_elig else 0.0,
        "person_ASR": (p_s / p_e * 100.0) if p_e else 0.0,
        "car_ASR": (c_s / c_e * 100.0) if c_e else 0.0,
        "mean_confidence_drop_across_attacks": statistics.fmean(confs) if confs else None,
        "mean_IoU_drop_across_attacks": statistics.fmean(ious) if ious else None,
    }
    (comb / "faster_rcnn_transfer_results.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")

    lines = [
        "# Faster R-CNN Transfer Attack — Combined Report",
        "",
        "Attack type: Hyper-YOLO-to-Faster-R-CNN Transfer Attack",
        "Model: Faster R-CNN ResNet-50 FPN V2",
        "Weights: FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1",
        "",
        f"Total images: {tot_img}",
        f"Eligible targets: {tot_elig}",
        f"Successful attacks: {tot_succ}",
        f"Overall ASR: {combined['overall_ASR']:.2f}%",
        f"Person ASR: {combined['person_ASR']:.2f}%",
        f"Car ASR: {combined['car_ASR']:.2f}%",
        f"Mean confidence drop (avg of attack means, both-valid only): {combined['mean_confidence_drop_across_attacks']}",
        f"Mean IoU drop (avg of attack means, both-valid only): {combined['mean_IoU_drop_across_attacks']}",
        "",
        "## Per attack",
    ]
    for s in summaries:
        lines.append(
            f"- {s['attack_id']}: ASR={s['ASR']:.2f}% "
            f"(success {s['successful_attacks']}/{s['eligible_targets']}) "
            f"person={s['person_ASR']:.2f}% car={s['car_ASR']:.2f}%"
        )
    (comb / "faster_rcnn_transfer_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Detector comparison from existing Hyper-YOLO summaries
    comp_rows = []
    for s in summaries:
        aid = s["attack_id"]
        hy_path = PROJECT / "source_data" / aid / "metadata" / "hyper_yolo_attack_summary.txt"
        hy = parse_hyper_yolo_summary(hy_path)
        row = {"attack_id": aid}
        row["hyper_yolo_ASR"] = hy.get("ASR", "")
        row["faster_rcnn_ASR"] = s["ASR"]
        row["hyper_yolo_person_ASR"] = hy.get("person_ASR", "")
        row["faster_rcnn_person_ASR"] = s["person_ASR"]
        row["hyper_yolo_car_ASR"] = hy.get("car_ASR", "")
        row["faster_rcnn_car_ASR"] = s["car_ASR"]
        comp_rows.append(row)
    with (comb / "detector_comparison.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "attack_id",
                "hyper_yolo_ASR",
                "faster_rcnn_ASR",
                "hyper_yolo_person_ASR",
                "faster_rcnn_person_ASR",
                "hyper_yolo_car_ASR",
                "faster_rcnn_car_ASR",
            ],
        )
        w.writeheader()
        w.writerows(comp_rows)


def smoke_test(model, transforms, device, categories) -> bool:
    print("=== SMOKE TEST attack_01 (2 images) ===")
    selected = load_selected("attack_01")
    verified = load_pair_verified("attack_01")
    picks = [r["filename"] for r in selected if r["filename"] in verified][:2]
    if len(picks) < 2:
        print("SMOKE FAIL: fewer than 2 verified pairs")
        return False
    smoke_root = PROJECT / "results" / "smoke_test"
    # temporarily process into smoke dirs by filter
    rows = process_attack(
        "attack_01",
        model,
        transforms,
        device,
        categories,
        filenames_filter=set(picks),
        write_visuals=True,
    )
    # Also copy smoke marker summary
    smoke_root.mkdir(parents=True, exist_ok=True)
    (smoke_root / "smoke_rows.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    ok = len(rows) == 2
    for r in rows:
        stem = Path(r["clean_image"]).stem
        if not (PROJECT / "results" / "attack_01" / "clean_detections" / f"{stem}.json").is_file():
            ok = False
        if not (PROJECT / "results" / "attack_01" / "patched_detections" / f"{stem}.json").is_file():
            ok = False
    print("SMOKE", "PASS" if ok else "FAIL", "files=", picks)
    return ok


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    t0 = time.perf_counter()
    model, weights, transforms, device = load_model()
    categories = list(weights.meta["categories"])
    write_environment(device)

    if mode == "smoke":
        ok = smoke_test(model, transforms, device, categories)
        return 0 if ok else 1

    # Full run: if smoke artifacts from prior run exist in attack_01, we re-run all images
    summaries = []
    all_rows = []
    for aid in ATTACK_IDS:
        print(f"\n===== {aid} =====")
        rows = process_attack(aid, model, transforms, device, categories, write_visuals=True)
        write_per_image_csv(PROJECT / "results" / aid / "per_image_results.csv", rows)
        summary = summarize_rows(rows, aid)
        (PROJECT / "results" / aid / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        write_attack_report(summary, PROJECT / "results" / aid / "report.md")
        summaries.append(summary)
        all_rows.extend(rows)
        print(
            f"{aid} ASR={summary['ASR']:.2f}% "
            f"({summary['successful_attacks']}/{summary['eligible_targets']})"
        )

    write_combined(summaries)
    ok, irep = verify_integrity()
    if not ok:
        print("INTEGRITY FAILURE — STOP")
        return 2

    # Final console block
    tot_img = sum(s["total_images"] for s in summaries)
    tot_elig = sum(s["eligible_targets"] for s in summaries)
    tot_succ = sum(s["successful_attacks"] for s in summaries)
    p_e = sum(s["person_eligible"] for s in summaries)
    p_s = sum(s["person_successful"] for s in summaries)
    c_e = sum(s["car_eligible"] for s in summaries)
    c_s = sum(s["car_successful"] for s in summaries)
    confs = [s["mean_confidence_drop"] for s in summaries if s["mean_confidence_drop"] is not None]
    ious = [s["mean_IoU_drop"] for s in summaries if s["mean_IoU_drop"] is not None]

    print("\n===============================================")
    print("FASTER R-CNN TRANSFER ATTACK BENCHMARK COMPLETE")
    print("===============================================")
    print()
    print("MODEL:")
    print("Faster R-CNN ResNet-50 FPN V2")
    print()
    print("WEIGHTS:")
    print("FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1")
    print()
    print("DATASET:")
    print("COCO 2017")
    print()
    print("ATTACK SOURCE:")
    print("Hyper-YOLO")
    print()
    print("ATTACK TYPE:")
    print("Hyper-YOLO-to-Faster-R-CNN Transfer Attack")
    print()
    for i, s in enumerate(summaries, 1):
        print("-----------------------------------------------")
        print(f"ATTACK 0{i}")
        print("-----------------------------------------------")
        print()
        print("Total images:")
        print(s["total_images"])
        print()
        print("Eligible targets:")
        print(s["eligible_targets"])
        print()
        print("Successful attacks:")
        print(s["successful_attacks"])
        print()
        print("ASR:")
        print(f"{s['ASR']:.2f}%")
        print()
        print("Person ASR:")
        print(f"{s['person_ASR']:.2f}%")
        print()
        print("Car ASR:")
        print(f"{s['car_ASR']:.2f}%")
        print()
    print("-----------------------------------------------")
    print("COMBINED")
    print("-----------------------------------------------")
    print()
    print("Total images:")
    print(tot_img)
    print()
    print("Eligible targets:")
    print(tot_elig)
    print()
    print("Successful attacks:")
    print(tot_succ)
    print()
    print("Overall ASR:")
    print(f"{(tot_succ / tot_elig * 100.0) if tot_elig else 0.0:.2f}%")
    print()
    print("Person ASR:")
    print(f"{(p_s / p_e * 100.0) if p_e else 0.0:.2f}%")
    print()
    print("Car ASR:")
    print(f"{(c_s / c_e * 100.0) if c_e else 0.0:.2f}%")
    print()
    print("Mean confidence drop:")
    print(statistics.fmean(confs) if confs else "N/A")
    print()
    print("Mean IoU drop:")
    print(statistics.fmean(ious) if ious else "N/A")
    print()
    print("-----------------------------------------------")
    print("SOURCE INTEGRITY")
    print("-----------------------------------------------")
    print()
    print("PASS" if ok else "FAIL")
    print()
    print("-----------------------------------------------")
    print("RESULT LOCATION")
    print("-----------------------------------------------")
    print()
    print(r"D:\project CS\Faster-RCNN-COCO-Attack\results\\")
    print()
    print("===============================================")
    print(f"Elapsed sec: {time.perf_counter() - t0:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
