"""
Hyper-YOLO-to-RT-DETRv2-S Transfer Attack benchmark.

Reads ONLY from local source_data/ copies.
Writes ONLY under RT-DETRv2-COCO-Attack/results/.
No combined_analysis in this script.
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
from PIL import Image, ImageDraw, ImageFont

PROJECT = Path(r"D:\project CS\RT-DETRv2-COCO-Attack").resolve()
sys.path.insert(0, str(PROJECT / "src"))

from evaluation_protocol import (  # noqa: E402
    CONF_THRESHOLD,
    TARGET_MATCH_IOU_THRESHOLD,
    gt_xywh_to_xyxy,
    match_target,
)
from load_model import CHECKPOINT_PATH, load_rtdetrv2_s  # noqa: E402

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


def run_detector(model, transforms, device, image_path: Path, label_to_coco_id, coco_id_to_name) -> tuple[list[dict], float]:
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
                "class_id": coco_id,
                "class_name": name,
                "confidence": conf,
                "xmin": x1,
                "ymin": y1,
                "xmax": x2,
                "ymax": y2,
                "bbox": [x1, y1, x2, y2],
                "contiguous_label": contig,
            }
        )
    return dets, elapsed_ms


def save_detections_json(path: Path, image_id: str, dets: list[dict], inference_time_ms: float) -> None:
    slim = [
        {
            "class_id": d["class_id"],
            "class_name": d["class_name"],
            "confidence": d["confidence"],
            "xmin": d["xmin"],
            "ymin": d["ymin"],
            "xmax": d["xmax"],
            "ymax": d["ymax"],
        }
        for d in dets
    ]
    path.write_text(
        json.dumps(
            {"image_id": image_id, "inference_time_ms": inference_time_ms, "detections": slim},
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
        x1, y1, x2, y2 = d["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline=(80, 140, 255), width=2)
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
        if r["clean_detected"] is True
        and r["patched_detected"] is True
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
        "model": "RT-DETRv2-S",
        "weights": str(CHECKPOINT_PATH.name),
        "dataset": "COCO 2017",
        "attack_source": "Existing Hyper-YOLO adversarial images",
        "attack_type": "Hyper-YOLO-to-RT-DETRv2-S Transfer Attack",
        "terminology": (
            "The adversarial patches were generated for Hyper-YOLO and evaluated on "
            "RT-DETRv2-S without regeneration, re-optimization, or training against RT-DETRv2-S."
        ),
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
        "mean_iou_drop": statistics.fmean(iou_drops) if iou_drops else None,
        "both_valid_n": len(both_valid),
        "confidence_threshold": CONF_THRESHOLD,
        "target_match_iou_threshold": TARGET_MATCH_IOU_THRESHOLD,
    }


def write_attack_report(summary: dict, path: Path) -> None:
    def fmt(x):
        return "N/A" if x is None else f"{x:.6f}"

    lines = [
        f"# {summary['attack_id']} — Hyper-YOLO-to-RT-DETRv2-S Transfer Attack",
        "",
        "Model: RT-DETRv2-S",
        f"Weights: {summary['weights']}",
        "Dataset: COCO 2017",
        "Attack source: Existing Hyper-YOLO adversarial images",
        "Attack type: Hyper-YOLO-to-RT-DETRv2-S Transfer Attack",
        "",
        summary["terminology"],
        "",
        "This is NOT a white-box RT-DETRv2 attack.",
        "",
        f"Total images: {summary['total_images']}",
        f"Eligible targets: {summary['eligible_targets']}",
        f"Successful attacks: {summary['successful_attacks']}",
        f"ASR: {summary['ASR']:.2f}%",
        "",
        "Person:",
        f"  eligible = {summary['person_eligible']}",
        f"  successful = {summary['person_successful']}",
        f"  ASR = {summary['person_ASR']:.2f}%",
        "",
        "Car:",
        f"  eligible = {summary['car_eligible']}",
        f"  successful = {summary['car_successful']}",
        f"  ASR = {summary['car_ASR']:.2f}%",
        "",
        f"Mean confidence drop: {fmt(summary['mean_confidence_drop'])}",
        f"Median confidence drop: {fmt(summary['median_confidence_drop'])}",
        f"Mean IoU drop: {fmt(summary['mean_iou_drop'])}",
        f"Both-valid n: {summary['both_valid_n']}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
        "clean_bbox",
        "patched_bbox",
        "clean_iou",
        "patched_iou",
        "confidence_drop",
        "iou_drop",
        "eligible",
        "attack_success",
        "clean_prediction_count",
        "patched_prediction_count",
        "inference_time_ms_clean",
        "inference_time_ms_patched",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            out = {}
            for k in fields:
                v = r.get(k)
                if v is None:
                    out[k] = ""
                elif isinstance(v, list):
                    out[k] = json.dumps(v)
                else:
                    out[k] = v
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

        clean_dets, clean_ms = run_detector(
            model, transforms, device, clean_path, label_to_coco_id, coco_id_to_name
        )
        save_detections_json(clean_det_dir / f"{Path(fn).stem}.json", image_id, clean_dets, clean_ms)
        clean_match = match_target(clean_dets, target_class, gt)
        eligible = bool(clean_match["detected"])

        patch_dets, patch_ms = run_detector(
            model, transforms, device, patch_path, label_to_coco_id, coco_id_to_name
        )
        save_detections_json(patch_det_dir / f"{Path(fn).stem}.json", image_id, patch_dets, patch_ms)
        patch_match = match_target(patch_dets, target_class, gt)
        attack_success = bool(eligible and (not patch_match["detected"]))

        if clean_match["detected"] and patch_match["detected"]:
            conf_drop = float(clean_match["confidence"]) - float(patch_match["confidence"])
            iou_drop = float(clean_match["iou"]) - float(patch_match["iou"])
        else:
            conf_drop = None
            iou_drop = None

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
            "clean_bbox": clean_match["bbox"],
            "patched_bbox": patch_match["bbox"],
            "clean_iou": clean_match["iou"],
            "patched_iou": patch_match["iou"],
            "confidence_drop": conf_drop,
            "iou_drop": iou_drop,
            "eligible": eligible,
            "attack_success": attack_success,
            "clean_prediction_count": len(clean_dets),
            "patched_prediction_count": len(patch_dets),
            "inference_time_ms_clean": clean_ms,
            "inference_time_ms_patched": patch_ms,
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


def write_environment(device, meta) -> None:
    info = {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "operating_system": platform.platform(),
        "device": str(device),
        "model_meta": meta,
    }
    (PROJECT / "results" / "environment_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


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


def smoke_test(model, transforms, device, label_to_coco_id, coco_id_to_name, meta) -> bool:
    print("=== SMOKE TEST attack_01 (2 images) ===")
    print("Model:", meta["model_name"])
    print("Checkpoint:", meta["checkpoint"])
    print("Device:", device)
    print("person contiguous label:", meta["person_contiguous_label"], "coco_id:", meta["person_coco_id"])
    print("car contiguous label:", meta["car_contiguous_label"], "coco_id:", meta["car_coco_id"])
    assert coco_id_to_name[1] == "person"
    assert coco_id_to_name[3] == "car"
    assert label_to_coco_id[0] == 1
    assert label_to_coco_id[2] == 3

    selected = load_selected("attack_01")
    verified = load_pair_verified("attack_01")
    picks = [r["filename"] for r in selected if r["filename"] in verified][:2]
    if len(picks) < 2:
        print("SMOKE FAIL: need 2 verified pairs")
        return False
    rows = process_attack(
        "attack_01",
        model,
        transforms,
        device,
        label_to_coco_id,
        coco_id_to_name,
        filenames_filter=set(picks),
        write_visuals=True,
    )
    smoke_dir = PROJECT / "results" / "smoke_test"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    (smoke_dir / "smoke_rows.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    ok = len(rows) == 2
    for r in rows:
        stem = Path(r["clean_image"]).stem
        if not (PROJECT / "results" / "attack_01" / "clean_detections" / f"{stem}.json").is_file():
            ok = False
        if not (PROJECT / "results" / "attack_01" / "patched_detections" / f"{stem}.json").is_file():
            ok = False
        if not (PROJECT / "results" / "attack_01" / "per_image_results.csv").exists():
            pass
    write_per_image_csv(smoke_dir / "smoke_per_image_results.csv", rows)
    print("SMOKE", "PASS" if ok else "FAIL", picks)
    return ok


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    t0 = time.perf_counter()
    model, transforms, device, label_to_coco_id, coco_id_to_name, meta = load_rtdetrv2_s()
    write_environment(device, meta)

    if mode == "smoke":
        return 0 if smoke_test(model, transforms, device, label_to_coco_id, coco_id_to_name, meta) else 1

    summaries = []
    paths = {"report_md": [], "summary_json": [], "per_image_csv": []}
    for aid in ATTACK_IDS:
        print(f"\n===== {aid} =====")
        rows = process_attack(
            aid, model, transforms, device, label_to_coco_id, coco_id_to_name, write_visuals=True
        )
        csv_path = PROJECT / "results" / aid / "per_image_results.csv"
        write_per_image_csv(csv_path, rows)
        summary = summarize_rows(rows, aid)
        json_path = PROJECT / "results" / aid / "summary.json"
        md_path = PROJECT / "results" / aid / "report.md"
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        write_attack_report(summary, md_path)
        summaries.append(summary)
        paths["report_md"].append(str(md_path))
        paths["summary_json"].append(str(json_path))
        paths["per_image_csv"].append(str(csv_path))
        print(f"{aid} ASR={summary['ASR']:.2f}% ({summary['successful_attacks']}/{summary['eligible_targets']})")

    ok = verify_integrity()
    print("\n========== FINAL SUMMARY ==========")
    print("1. Project root:")
    print(str(PROJECT))
    print("2. Model name:")
    print("RT-DETRv2-S")
    print("3. Weight/checkpoint used:")
    print(str(CHECKPOINT_PATH))
    print("4. Device used:")
    print(str(device))
    for i, s in enumerate(summaries, 1):
        print(f"{4+i}. {s['attack_id']} summary:")
        print(
            f"   total={s['total_images']} eligible={s['eligible_targets']} "
            f"success={s['successful_attacks']} ASR={s['ASR']:.2f}% "
            f"person_ASR={s['person_ASR']:.2f}% car_ASR={s['car_ASR']:.2f}% "
            f"mean_conf_drop={s['mean_confidence_drop']} median_conf_drop={s['median_confidence_drop']} "
            f"mean_iou_drop={s['mean_iou_drop']} both_valid_n={s['both_valid_n']}"
        )
    print("10. report.md paths:")
    for p in paths["report_md"]:
        print("   ", p)
    print("11. summary.json paths:")
    for p in paths["summary_json"]:
        print("   ", p)
    print("12. per_image_results.csv paths:")
    for p in paths["per_image_csv"]:
        print("   ", p)
    print("Source integrity:", "PASS" if ok else "FAIL")
    print(f"Elapsed sec: {time.perf_counter()-t0:.1f}")
    print("NOTE: combined_analysis was NOT created.")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
