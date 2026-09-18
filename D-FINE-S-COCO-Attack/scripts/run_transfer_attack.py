"""
Hyper-YOLO-to-D-FINE-S Transfer Attack benchmark.

Reads ONLY from local source_data/ copies.
Writes ONLY under D-FINE-S-COCO-Attack/results/.
No combined_analysis in this script.
Does NOT generate or optimize adversarial patches.
"""
from __future__ import annotations

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
from PIL import Image, ImageDraw, ImageFont

PROJECT = Path(r"D:\project CS\D-FINE-S-COCO-Attack").resolve()
sys.path.insert(0, str(PROJECT / "scripts"))

from evaluation_protocol import (  # noqa: E402
    CONF_THRESHOLD,
    IOU_THRESHOLD,
    gt_xywh_to_xyxy,
    match_target,
)
from load_model import CHECKPOINT_PATH, load_dfine_s  # noqa: E402

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


def run_detector(model, transforms, device, image_path: Path, label_to_coco_id, coco_id_to_name):
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    orig = torch.tensor([[w, h]], device=device)
    batch = transforms(image)[None].to(device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        labels, boxes, scores = model(batch, orig)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    lab = labels[0].detach().cpu()
    box = boxes[0].detach().cpu()
    sco = scores[0].detach().cpu()
    dets = []
    for i in range(lab.shape[0]):
        contig = int(lab[i].item())
        coco_id = int(label_to_coco_id[contig])
        name = str(coco_id_to_name[coco_id])
        conf = float(sco[i].item())
        x1, y1, x2, y2 = [float(v) for v in box[i].tolist()]
        dets.append(
            {
                # Contiguous deploy IDs: person=0, car=2 (user protocol)
                "class_id": contig,
                "coco_category_id": coco_id,
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


def save_detections_json(path: Path, image_name: str, dets: list[dict], inference_time_ms: float) -> None:
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
            "coco_category_id": d["coco_category_id"],
        }
        for d in dets
    ]
    path.write_text(
        json.dumps(
            {
                "image_name": image_name,
                "inference_time_ms": inference_time_ms,
                "detections": slim,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def draw_vis(image, dets, gt_xyxy, match, *, title: str, conf_vis: float = CONF_THRESHOLD):
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    font = _font(14)
    gx1, gy1, gx2, gy2 = gt_xyxy
    draw.rectangle([gx1, gy1, gx2, gy2], outline=(255, 220, 0), width=3)
    draw.text((gx1 + 2, max(0, gy1 - 16)), "GT", fill=(255, 220, 0), font=font)
    for d in dets:
        if d["confidence"] < conf_vis:
            continue
        if match.get("bbox") and d["bbox"] == match["bbox"]:
            continue
        name = str(d["class_name"]).lower()
        # Emphasize person/car
        color = (0, 180, 255) if name in {"person", "car"} else (80, 140, 255)
        x1, y1, x2, y2 = d["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        draw.text((x1 + 2, max(0, y1 - 14)), f"{d['class_name']} {d['confidence']:.2f}", fill=color, font=font)
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
    both_valid = [r for r in rows if r["both_valid"] is True]

    def class_stats(cls: str):
        e = [r for r in eligible if r["target_class"] == cls]
        s = [r for r in e if r["attack_success"] is True]
        return {
            "eligible": len(e),
            "success": len(s),
            "asr_percent": (len(s) / len(e) * 100.0) if e else 0.0,
        }

    person = class_stats("person")
    car = class_stats("car")
    conf_drops = [float(r["confidence_drop"]) for r in both_valid if r["confidence_drop"] is not None]
    iou_drops = [float(r["iou_drop"]) for r in both_valid if r["iou_drop"] is not None]
    return {
        "model": "D-FINE-S",
        "attack_id": attack_id,
        "total_images": len(rows),
        "eligible": len(eligible),
        "successful_attacks": len(successes),
        "asr_percent": (len(successes) / len(eligible) * 100.0) if eligible else 0.0,
        "person": person,
        "car": car,
        "both_valid_n": len(both_valid),
        "mean_confidence_drop": statistics.fmean(conf_drops) if conf_drops else None,
        "median_confidence_drop": statistics.median(conf_drops) if conf_drops else None,
        "mean_iou_drop": statistics.fmean(iou_drops) if iou_drops else None,
        "median_iou_drop": statistics.median(iou_drops) if iou_drops else None,
        "confidence_threshold": CONF_THRESHOLD,
        "iou_threshold": IOU_THRESHOLD,
        "checkpoint": str(CHECKPOINT_PATH.name),
        "attack_source": "Existing Hyper-YOLO adversarial images",
        "attack_type": "transfer_attack_black_box",
    }


def write_summary_result_txt(summary: dict, path: Path) -> None:
    def fmt(x):
        return "N/A" if x is None else f"{x:.6f}"

    lines = [
        "# D-FINE-S TRANSFER ATTACK RESULT",
        "",
        f"Attack: {summary['attack_id']}",
        f"Total Images: {summary['total_images']}",
        "",
        f"Eligible: {summary['eligible']}",
        f"Successful Attacks: {summary['successful_attacks']}",
        f"ASR: {summary['asr_percent']:.2f}%",
        "",
        "## PERSON",
        "",
        f"Eligible: {summary['person']['eligible']}",
        f"Success: {summary['person']['success']}",
        f"ASR: {summary['person']['asr_percent']:.2f}%",
        "",
        "## CAR",
        "",
        f"Eligible: {summary['car']['eligible']}",
        f"Success: {summary['car']['success']}",
        f"ASR: {summary['car']['asr_percent']:.2f}%",
        "",
        "## BOTH-VALID N",
        "",
        f"{summary['both_valid_n']}",
        "",
        "## CONFIDENCE DROP",
        "",
        f"Mean: {fmt(summary['mean_confidence_drop'])}",
        f"Median: {fmt(summary['median_confidence_drop'])}",
        "",
        "## IOU DROP",
        "",
        f"Mean: {fmt(summary['mean_iou_drop'])}",
        f"Median: {fmt(summary['median_iou_drop'])}",
        "",
        "Confidence/IoU drops are calculated only over samples where both clean and patched detections remain valid.",
        "",
        "Evaluation threshold:",
        "Confidence >= 0.25",
        "IoU >= 0.50",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_report_md(summary: dict, path: Path) -> None:
    def fmt(x):
        return "N/A" if x is None else f"{x:.6f}"

    p = summary["person"]
    c = summary["car"]
    asr = summary["asr_percent"]
    interpretation = (
        f"On {summary['total_images']} images, {summary['eligible']} targets were eligible "
        f"(clean-valid under conf>=0.25 and IoU>=0.50). Of these, {summary['successful_attacks']} "
        f"became invalid after the Hyper-YOLO patch (ASR={asr:.2f}%). "
        f"Person ASR={p['asr_percent']:.2f}% ({p['success']}/{p['eligible']}); "
        f"car ASR={c['asr_percent']:.2f}% ({c['success']}/{c['eligible']}). "
        f"Among {summary['both_valid_n']} both-valid targets, mean confidence drop="
        f"{fmt(summary['mean_confidence_drop'])} and mean IoU drop={fmt(summary['mean_iou_drop'])}. "
        "These figures describe transfer behavior of existing Hyper-YOLO adversarial images on D-FINE-S only; "
        "they do not claim universal attack effectiveness."
    )
    lines = [
        "# D-FINE-S Transfer Attack Evaluation",
        "",
        "## Model",
        "",
        "D-FINE-S",
        "",
        "## Paper",
        "",
        "D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement",
        "",
        "## Publication",
        "",
        "ICLR 2025 Spotlight",
        "",
        "## Dataset",
        "",
        "MS COCO 2017",
        "",
        "## Attack Source",
        "",
        "Hyper-YOLO-N adversarial images",
        "",
        "## Attack Type",
        "",
        "Transfer attack / black-box evaluation",
        "",
        "## Evaluation Thresholds",
        "",
        "Confidence >= 0.25",
        "IoU >= 0.50",
        "",
        "## Target Classes",
        "",
        "Person",
        "Car",
        "",
        "## Attack Protocol",
        "",
        "Existing Hyper-YOLO clean and patched image pairs are evaluated independently with the same "
        "official COCO-pretrained D-FINE-S weights and identical preprocessing. A target is eligible only "
        "if it is validly detected on the clean image. Attack success requires an eligible clean detection "
        "that becomes invalid/missing on the patched image. Confidence and IoU drops are computed only for "
        "both-valid cases. No new patches were generated and no D-FINE training/fine-tuning was performed.",
        "",
        "## Metrics",
        "",
        "* Eligible",
        "* Successful attacks",
        "* ASR",
        "* Mean confidence drop",
        "* Median confidence drop",
        "* Mean IoU drop",
        "* Median IoU drop",
        "* Both-valid count",
        "",
        "## Results",
        "",
        "| Metric   | Person | Car | Overall |",
        "| -------- | -----: | --: | ------: |",
        f"| Eligible | {p['eligible']} | {c['eligible']} | {summary['eligible']} |",
        f"| Success  | {p['success']} | {c['success']} | {summary['successful_attacks']} |",
        f"| ASR (%)  | {p['asr_percent']:.2f} | {c['asr_percent']:.2f} | {asr:.2f} |",
        "",
        "| Metric                 | Value |",
        "| ---------------------- | ----: |",
        f"| Total images           | {summary['total_images']} |",
        f"| Eligible               | {summary['eligible']} |",
        f"| Success                | {summary['successful_attacks']} |",
        f"| ASR (%)                | {asr:.2f} |",
        f"| Mean confidence drop   | {fmt(summary['mean_confidence_drop'])} |",
        f"| Median confidence drop | {fmt(summary['median_confidence_drop'])} |",
        f"| Mean IoU drop          | {fmt(summary['mean_iou_drop'])} |",
        f"| Median IoU drop        | {fmt(summary['median_iou_drop'])} |",
        f"| Both-valid n           | {summary['both_valid_n']} |",
        "",
        "## Interpretation",
        "",
        interpretation,
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
        "clean_detected",
        "clean_class",
        "clean_confidence",
        "clean_iou",
        "patched_detected",
        "patched_class",
        "patched_confidence",
        "patched_iou",
        "eligible",
        "attack_success",
        "confidence_drop",
        "iou_drop",
        "both_valid",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            out = {}
            for k in fields:
                v = r.get(k)
                out[k] = "" if v is None else v
            w.writerow(out)


def process_attack(
    attack_id: str,
    model,
    transforms,
    device,
    label_to_coco_id,
    coco_id_to_name,
    *,
    filenames_filter: set[str] | None = None,
    write_visuals: bool = True,
    failure_log: list[dict] | None = None,
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
            print(f"  SKIP unresolved {attack_id}/{fn}")
            continue
        clean_path = PROJECT / "source_data" / attack_id / "clean_images" / fn
        patch_path = PROJECT / "source_data" / attack_id / "patched_images" / fn
        if not clean_path.is_file() or not patch_path.is_file():
            msg = f"missing copy {attack_id}/{fn}"
            print(f"  SKIP {msg}")
            if failure_log is not None:
                failure_log.append({"attack_id": attack_id, "filename": fn, "reason": msg})
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
                model, transforms, device, clean_path, label_to_coco_id, coco_id_to_name
            )
            save_detections_json(clean_det_dir / f"{Path(fn).stem}.json", fn, clean_dets, clean_ms)
            clean_match = match_target(clean_dets, target_class, gt)
            eligible = bool(clean_match["detected"])

            patch_dets, patch_ms = run_detector(
                model, transforms, device, patch_path, label_to_coco_id, coco_id_to_name
            )
            save_detections_json(patch_det_dir / f"{Path(fn).stem}.json", fn, patch_dets, patch_ms)
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
                "clean_detected": clean_match["detected"],
                "clean_class": clean_match["class_name"],
                "clean_confidence": clean_match["confidence"],
                "clean_iou": clean_match["iou"],
                "patched_detected": patch_match["detected"],
                "patched_class": patch_match["class_name"],
                "patched_confidence": patch_match["confidence"],
                "patched_iou": patch_match["iou"],
                "eligible": eligible,
                "attack_success": attack_success,
                "confidence_drop": conf_drop,
                "iou_drop": iou_drop,
                "both_valid": both_valid,
                "_clean_ms": clean_ms,
                "_patch_ms": patch_ms,
            }
            rows.append(row)

            if write_visuals:
                clean_img = Image.open(clean_path).convert("RGB")
                patch_img = Image.open(patch_path).convert("RGB")
                ctitle = (
                    f"{fn} | CLEAN | {target_class} | eligible={eligible} | "
                    f"det={clean_match['detected']} conf={clean_match['confidence']} iou={clean_match['iou']}"
                )
                ptitle = (
                    f"{fn} | PATCHED | {target_class} | success={attack_success} | "
                    f"det={patch_match['detected']} conf={patch_match['confidence']} iou={patch_match['iou']}"
                )
                draw_vis(clean_img, clean_dets, gt, clean_match, title=ctitle).save(vis_clean / fn, quality=92)
                draw_vis(patch_img, patch_dets, gt, patch_match, title=ptitle).save(vis_patch / fn, quality=92)

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
                        "reason": err,
                        "traceback": traceback.format_exc(),
                    }
                )
            continue
    return rows


def write_experiment_config(device, meta) -> None:
    cfg = {
        "experiment_name": "Hyper-YOLO-to-D-FINE-S Transfer Attack",
        "date_time_utc": datetime.now(timezone.utc).isoformat(),
        "model": "D-FINE-S",
        "model_variant": "dfine_hgnetv2_s_coco",
        "checkpoint": str(CHECKPOINT_PATH),
        "checkpoint_filename": CHECKPOINT_PATH.name,
        "checkpoint_sha256": meta.get("checkpoint_sha256"),
        "checkpoint_url": meta.get("checkpoint_url"),
        "config": meta.get("config"),
        "confidence_threshold": CONF_THRESHOLD,
        "iou_threshold": IOU_THRESHOLD,
        "target_classes": ["person", "car"],
        "person_contiguous_class_id": 0,
        "car_contiguous_class_id": 2,
        "dataset": "MS COCO 2017",
        "input_resolution": [640, 640],
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "device": str(device),
        "os": platform.platform(),
        "framework": "official Peterande/D-FINE",
        "do_not_regenerate_patches": True,
        "fine_tune": False,
        "create_combined_analysis": False,
    }
    (PROJECT / "logs" / "experiment_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (PROJECT / "results" / "environment_info.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")


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
    print("CUDA version:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    print("D-FINE model:", meta["model"])
    print("Checkpoint path:", meta["checkpoint"])
    print("Device:", device)
    print("================================")


def smoke_test(model, transforms, device, label_to_coco_id, coco_id_to_name, meta) -> bool:
    print("=== SMOKE TEST attack_01 (1 clean + 1 patched pair) ===")
    assert coco_id_to_name[1] == "person"
    assert coco_id_to_name[3] == "car"
    assert label_to_coco_id[0] == 1
    assert label_to_coco_id[2] == 3

    selected = load_selected("attack_01")
    verified = load_pair_verified("attack_01")
    picks = [r["filename"] for r in selected if r["filename"] in verified][:1]
    if len(picks) < 1:
        print("SMOKE FAIL: need 1 verified pair")
        return False

    failures: list[dict] = []
    rows = process_attack(
        "attack_01",
        model,
        transforms,
        device,
        label_to_coco_id,
        coco_id_to_name,
        filenames_filter=set(picks),
        write_visuals=True,
        failure_log=failures,
    )
    smoke_dir = PROJECT / "results" / "smoke_test"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    (smoke_dir / "smoke_rows.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    write_per_image_csv(smoke_dir / "smoke_per_image_results.csv", rows)

    # Print sample detections for class-map verification
    stem = Path(picks[0]).stem
    det_path = PROJECT / "results" / "attack_01" / "clean_detections" / f"{stem}.json"
    if det_path.is_file():
        payload = json.loads(det_path.read_text(encoding="utf-8"))
        print("Sample clean detections (conf>=0.25, person/car):")
        for d in payload["detections"]:
            if d["confidence"] >= CONF_THRESHOLD and d["class_name"] in {"person", "car"}:
                print(f"  {d['class_name']} {d['confidence']:.2f} (class_id={d['class_id']})")

    ok = (
        len(rows) == 1
        and not failures
        and det_path.is_file()
        and (PROJECT / "results" / "attack_01" / "patched_detections" / f"{stem}.json").is_file()
        and (PROJECT / "results" / "attack_01" / "visualizations" / "clean" / picks[0]).is_file()
        and (PROJECT / "results" / "attack_01" / "visualizations" / "patched" / picks[0]).is_file()
    )
    print("SMOKE", "PASS" if ok else "FAIL", picks)
    return ok


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    t0 = time.perf_counter()
    model, transforms, device, label_to_coco_id, coco_id_to_name, meta = load_dfine_s()
    print_startup(device, meta)
    write_experiment_config(device, meta)

    if mode == "smoke":
        return 0 if smoke_test(model, transforms, device, label_to_coco_id, coco_id_to_name, meta) else 1

    all_failures: list[dict] = []
    summaries = []
    for aid in ATTACK_IDS:
        print(f"\n===== {aid} =====")
        rows = process_attack(
            aid,
            model,
            transforms,
            device,
            label_to_coco_id,
            coco_id_to_name,
            write_visuals=True,
            failure_log=all_failures,
        )
        out_root = PROJECT / "results" / aid
        csv_path = out_root / "per_image_results.csv"
        write_per_image_csv(csv_path, rows)
        summary = summarize_rows(rows, aid)
        (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        write_summary_result_txt(summary, out_root / "summary_result.txt")
        write_report_md(summary, out_root / "report.md")
        summaries.append(summary)
        print(
            f"{aid} ASR={summary['asr_percent']:.2f}% "
            f"({summary['successful_attacks']}/{summary['eligible']}) "
            f"both_valid={summary['both_valid_n']} failures_so_far={len(all_failures)}"
        )

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

    # Final aggregate table
    tot_images = sum(s["total_images"] for s in summaries)
    tot_elig = sum(s["eligible"] for s in summaries)
    tot_succ = sum(s["successful_attacks"] for s in summaries)
    tot_person_e = sum(s["person"]["eligible"] for s in summaries)
    tot_person_s = sum(s["person"]["success"] for s in summaries)
    tot_car_e = sum(s["car"]["eligible"] for s in summaries)
    tot_car_s = sum(s["car"]["success"] for s in summaries)

    print("\n========== FINAL TABLE ==========")
    print("| Attack    | Total | Eligible | Success | ASR |")
    print("| --------- | ----: | -------: | ------: | --: |")
    for s in summaries:
        print(
            f"| {s['attack_id']} | {s['total_images']} | {s['eligible']} | "
            f"{s['successful_attacks']} | {s['asr_percent']:.2f}% |"
        )
    overall_asr = (tot_succ / tot_elig * 100.0) if tot_elig else 0.0
    print(f"| TOTAL     | {tot_images} | {tot_elig} | {tot_succ} | {overall_asr:.2f}% |")
    print()
    print("Person total:")
    print(f"Eligible: {tot_person_e}")
    print(f"Success: {tot_person_s}")
    print(f"ASR: {(tot_person_s / tot_person_e * 100.0) if tot_person_e else 0.0:.2f}%")
    print()
    print("Car total:")
    print(f"Eligible: {tot_car_e}")
    print(f"Success: {tot_car_s}")
    print(f"ASR: {(tot_car_s / tot_car_e * 100.0) if tot_car_e else 0.0:.2f}%")
    print()
    print("Overall:")
    print(f"Total: {tot_images}")
    print(f"Eligible: {tot_elig}")
    print(f"Success: {tot_succ}")
    print(f"ASR: {overall_asr:.2f}%")
    print()
    print("Failed images:", len(all_failures))
    for f in all_failures:
        print(f"  {f['attack_id']}/{f['filename']}: {f['reason']}")
    print("Source integrity:", "PASS" if ok else "FAIL")
    print(f"Elapsed sec: {time.perf_counter()-t0:.1f}")
    print("NOTE: combined_analysis was NOT created.")
    return 0 if ok and not all_failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
