#!/usr/bin/env python3
"""Structured 6-image SD2 dedicated inpainting pilot (strength=1.00).

Outputs only under mitigation_debug/sd2_inpainting_pilot_100/.
Does not modify attacks, mitigation_01, Hyper-YOLO, or Stable-Diffusion-Patch.
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

EXPERIMENT = Path(__file__).resolve().parents[1]
ATTACK_02 = EXPERIMENT.parent / "Hyper-YOLO" / "attacks" / "attack_02"
MODEL = EXPERIMENT / "mitigation_models" / "stable-diffusion-2-inpainting"
PILOT_ROOT = EXPERIMENT / "mitigation_debug" / "sd2_inpainting_pilot_100"
PER_IMAGE = PILOT_ROOT / "per_image"
RESULTS = PILOT_ROOT / "results"
OLD_PILOT = EXPERIMENT / "mitigation_debug" / "multi_image_pilot_055"
SD_PYTHON = EXPERIMENT.parent / "Stable-Diffusion-Patch" / ".venv" / "Scripts" / "python.exe"
HYPER_PYTHON = EXPERIMENT.parent / "Hyper-YOLO" / ".venv" / "Scripts" / "python.exe"

# Exact same six images as prior Img2Img pilot
PILOT_IMAGES = [
    {"filename": "000000413689.jpg", "role": "success", "target_class": "car", "patch_size": 7},
    {"filename": "000000135410.jpg", "role": "success", "target_class": "car", "patch_size": 28},
    {"filename": "000000407083.jpg", "role": "success", "target_class": "car", "patch_size": 144},
    {"filename": "000000102411.jpg", "role": "control", "target_class": "car", "patch_size": 11},
    {"filename": "000000436617.jpg", "role": "control", "target_class": "person", "patch_size": 59},
    {"filename": "000000271997.jpg", "role": "control", "target_class": "person", "patch_size": 145},
]

STRENGTH = 1.00
STEPS = 30
GUIDANCE = 7.5
SEED = 20260727
RESOLUTION = 512
CONTEXT = 3.0
MIN_SIDE = 128
FEATHER = 4
PATCH_SCALE = 0.30

PROMPT = (
    "realistic natural photograph, restore the original object surface, "
    "natural texture, consistent lighting, no sticker or artificial patch"
)
NEG = (
    "sticker, adversarial patch, logo, text, watermark, abstract pattern, "
    "cartoon, distortion, face hallucination, extra object"
)


def reconstruct_patch(bbox_x, bbox_y, bbox_w, bbox_h, image_w, image_h):
    patch_size = max(1, int(round(PATCH_SCALE * min(bbox_w, bbox_h))))
    cx = bbox_x + bbox_w / 2.0
    cy = bbox_y + bbox_h / 2.0
    px = max(0.0, min(cx - patch_size / 2.0, image_w - patch_size))
    py = max(0.0, min(cy - patch_size / 2.0, image_h - patch_size))
    x1 = int(round(px))
    y1 = int(round(py))
    return x1, y1, x1 + patch_size, y1 + patch_size, patch_size


def compute_context_crop(w, h, x1, y1, x2, y2):
    pw, ph = max(1, x2 - x1), max(1, y2 - y1)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    side = max(MIN_SIDE, int(round(CONTEXT * max(pw, ph))))
    side = min(side, w, h)
    left = max(0, min(int(round(cx - side / 2.0)), w - side))
    top = max(0, min(int(round(cy - side / 2.0)), h - side))
    return left, top, left + side, top + side


def feathered_alpha(size, feather):
    alpha = np.ones((size, size), dtype=np.float32)
    feather = max(0, int(feather))
    if feather <= 0 or size < 2:
        return alpha
    for i in range(feather):
        weight = (i + 1) / (feather + 1)
        alpha[i, :] = np.minimum(alpha[i, :], weight)
        alpha[-(i + 1), :] = np.minimum(alpha[-(i + 1), :], weight)
        alpha[:, i] = np.minimum(alpha[:, i], weight)
        alpha[:, -(i + 1)] = np.minimum(alpha[:, -(i + 1)], weight)
    return alpha


def visual_flags(attacked_patch: np.ndarray, restored_patch: np.ndarray) -> dict:
    atk = attacked_patch.astype(np.float32)
    res = restored_patch.astype(np.float32)
    mae = float(np.mean(np.abs(atk - res)))
    res_chroma = float(np.mean(np.max(res, axis=2) - np.min(res, axis=2)))
    sticker = "YES" if (mae < 25 and res_chroma > 40) else ("LIKELY_PARTIAL" if mae < 40 and res_chroma > 35 else "NO")
    std = float(np.std(res))
    flat = "YES" if std < 22 else "NO"
    gray = np.mean(res, axis=2).astype(np.float32)
    gy, gx = np.gradient(gray)
    edge = float(np.mean(np.sqrt(gx * gx + gy * gy)))
    halluc = "YES" if mae >= 25 and edge > 12 and std > 28 else "NO"
    if res.shape[0] > 8 and res.shape[1] > 8:
        border = np.concatenate(
            [
                res[0:2].reshape(-1, 3),
                res[-2:].reshape(-1, 3),
                res[:, 0:2].reshape(-1, 3),
                res[:, -2:].reshape(-1, 3),
            ],
            axis=0,
        )
        interior = res[2:-2, 2:-2].reshape(-1, 3)
        seam = "YES" if float(np.linalg.norm(border.mean(0) - interior.mean(0))) > 18 else "NO"
    else:
        seam = "UNKNOWN"
    # natural if sticker gone, not flat, not hallucinated, modest seam
    natural = "YES" if sticker == "NO" and flat == "NO" and halluc == "NO" else "NO"
    return {
        "sticker_removed": "YES" if sticker == "NO" else "NO",
        "hallucination": halluc,
        "flat_gray": flat,
        "visible_seam": seam,
        "natural_restoration": natural,
        "mae_attacked_vs_restored": mae,
    }


def panel(img: Image.Image, label: str, size=(280, 220)) -> Image.Image:
    canvas = Image.new("RGB", size, (245, 245, 245))
    thumb = img.copy()
    thumb.thumbnail((size[0] - 8, size[1] - 34), Image.Resampling.LANCZOS)
    canvas.paste(thumb, ((size[0] - thumb.size[0]) // 2, 30 + (size[1] - 34 - thumb.size[1]) // 2))
    d = ImageDraw.Draw(canvas)
    d.rectangle([0, 0, size[0], 26], fill=(20, 20, 20))
    d.text((6, 6), label[:42], fill=(255, 255, 255))
    return canvas


def hstack(images):
    h = max(im.size[1] for im in images)
    canvas = Image.new("RGB", (sum(im.size[0] for im in images) + 4 * (len(images) - 1), h), (255, 255, 255))
    x = 0
    for im in images:
        canvas.paste(im, (x, 0))
        x += im.size[0] + 4
    return canvas


def load_tables():
    attack = {
        r["filename"]: r
        for r in csv.DictReader((ATTACK_02 / "results" / "attack_results.csv").open(encoding="utf-8"))
    }
    selected = {
        r["filename"]: r
        for r in csv.DictReader((ATTACK_02 / "results" / "selected_images.csv").open(encoding="utf-8"))
    }
    return attack, selected


INPAINT_WORKER = r'''
import json, sys
from pathlib import Path
import numpy as np
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image

args = json.loads(sys.argv[1])
model = args["model"]
attacked_path = Path(args["attacked"])
out_dir = Path(args["out_dir"])
x1,y1,x2,y2 = args["x1"], args["y1"], args["x2"], args["y2"]
left,top,right,bottom = args["left"], args["top"], args["right"], args["bottom"]
strength = float(args["strength"])
seed = int(args["seed"])
steps = int(args["steps"])
guidance = float(args["guidance"])
resolution = int(args["resolution"])
feather = int(args["feather"])
prompt = args["prompt"]
neg = args["neg"]

def feathered_alpha(size, feather):
    alpha = np.ones((size, size), dtype=np.float32)
    feather = max(0, int(feather))
    if feather <= 0 or size < 2:
        return alpha
    for i in range(feather):
        weight = (i + 1) / (feather + 1)
        alpha[i, :] = np.minimum(alpha[i, :], weight)
        alpha[-(i + 1), :] = np.minimum(alpha[-(i + 1), :], weight)
        alpha[:, i] = np.minimum(alpha[:, i], weight)
        alpha[:, -(i + 1)] = np.minimum(alpha[:, -(i + 1)], weight)
    return alpha

attacked = Image.open(attacked_path).convert("RGB")
w, h = attacked.size
mask_full = Image.new("L", (w, h), 0)
m = np.array(mask_full); m[y1:y2, x1:x2] = 255; mask_full = Image.fromarray(m)
cw, ch = right - left, bottom - top
rx1, ry1, rx2, ry2 = x1 - left, y1 - top, x2 - left, y2 - top
crop = attacked.crop((left, top, right, bottom))
mask_crop = mask_full.crop((left, top, right, bottom))
crop_512 = crop.resize((resolution, resolution), Image.Resampling.LANCZOS)
mask_512 = mask_crop.resize((resolution, resolution), Image.Resampling.NEAREST)

attacked.save(out_dir / "01_attacked.jpg", quality=95)
mask_full.save(out_dir / "02_mask.png")
crop.save(out_dir / "03_context_crop.jpg", quality=95)
mask_512.save(out_dir / "04_mask_512.png")

device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    model, torch_dtype=torch.float16 if device=="cuda" else torch.float32,
    variant="fp16", use_safetensors=True, local_files_only=True,
)
if hasattr(pipe, "enable_attention_slicing"):
    pipe.enable_attention_slicing()
if device == "cuda" and hasattr(pipe, "enable_model_cpu_offload"):
    pipe.enable_model_cpu_offload()
else:
    pipe = pipe.to(device)

try:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    out = pipe(
        prompt=prompt, negative_prompt=neg, image=crop_512, mask_image=mask_512,
        strength=strength, guidance_scale=guidance, num_inference_steps=steps, generator=gen,
    ).images[0]
    out.save(out_dir / "05_raw_inpaint_512.png")
    resized = out.resize((cw, ch), Image.Resampling.LANCZOS)
    patch = np.array(resized).astype(np.float32)[ry1:ry2, rx1:rx2]
    base = np.array(attacked).astype(np.float32)
    alpha = feathered_alpha(max(y2-y1, x2-x1), feather)[:y2-y1, :x2-x1][:,:,None]
    base[y1:y2, x1:x2] = alpha * patch + (1.0 - alpha) * base[y1:y2, x1:x2]
    Image.fromarray(np.clip(base,0,255).astype(np.uint8)).save(out_dir / "06_restored_full.jpg", quality=95)
    print(json.dumps({"status":"OK","device":device}))
except Exception as e:
    (out_dir / "FAILED.txt").write_text(str(e), encoding="utf-8")
    print(json.dumps({"status":"FAILED","error":str(e),"device":device}))
    raise
'''


def run_inpaint_one(out_dir: Path, attacked: Path, geom: dict) -> dict:
    payload = {
        "model": str(MODEL),
        "attacked": str(attacked),
        "out_dir": str(out_dir),
        **geom,
        "strength": STRENGTH,
        "seed": SEED,
        "steps": STEPS,
        "guidance": GUIDANCE,
        "resolution": RESOLUTION,
        "feather": FEATHER,
        "prompt": PROMPT,
        "neg": NEG,
    }
    result = subprocess.run(
        [str(SD_PYTHON), "-c", INPAINT_WORKER, json.dumps(payload)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "unknown").strip()
        return {"status": "FAILED", "error": detail}
    # last line JSON
    lines = [ln for ln in (result.stdout or "").splitlines() if ln.strip().startswith("{")]
    if not lines:
        return {"status": "FAILED", "error": result.stdout or result.stderr or "no json"}
    return json.loads(lines[-1])


def draw_verification(
    attacked: Image.Image,
    restored: Image.Image,
    yolo_img: Image.Image | None,
    out_path: Path,
    row: dict,
):
    pad = max(40, int(row["patch_size_pixels"]))
    x1, y1, x2, y2 = row["patch_x1"], row["patch_y1"], row["patch_x2"], row["patch_y2"]
    w, h = attacked.size
    zx1, zy1 = max(0, x1 - pad), max(0, y1 - pad)
    zx2, zy2 = min(w, x2 + pad), min(h, y2 + pad)
    label = row.get("row_label", "")
    panels = [
        panel(attacked.crop((zx1, zy1, zx2, zy2)), "ATTACKED"),
        panel(restored.crop((zx1, zy1, zx2, zy2)), "SD2 INPAINT 1.00"),
    ]
    if yolo_img is not None:
        panels.append(panel(yolo_img.crop((zx1, zy1, zx2, zy2)), "HYPER-YOLO"))
    grid = hstack(panels)
    header = Image.new("RGB", (grid.size[0], 40), (255, 255, 255))
    d = ImageDraw.Draw(header)
    d.text(
        (8, 12),
        f"{row['filename']} | {label} | det={row['inpainted_detected']} "
        f"conf={row['inpainted_confidence']:.3f} iou={row['inpainted_iou']:.3f} "
        f"halluc={row['hallucination']}",
        fill=(0, 0, 0),
    )
    canvas = Image.new("RGB", (grid.size[0], grid.size[1] + 40), (255, 255, 255))
    canvas.paste(header, (0, 0))
    canvas.paste(grid, (0, 40))
    canvas.save(out_path, quality=95)


def main():
    for d in [
        PER_IMAGE,
        RESULTS / "recovered_successes",
        RESULTS / "failed_successes",
        RESULTS / "preserved_controls",
        RESULTS / "regressed_controls",
        RESULTS / "hallucination_cases",
    ]:
        d.mkdir(parents=True, exist_ok=True)

    attack, selected = load_tables()
    rows = []
    sd_failures = 0

    # ---- SD restoration for each image ----
    for spec in PILOT_IMAGES:
        fn = spec["filename"]
        stem = Path(fn).stem
        out_dir = PER_IMAGE / stem
        out_dir.mkdir(parents=True, exist_ok=True)
        a = attack[fn]
        s = selected[fn]
        bbox_x, bbox_y = float(s["bbox_x"]), float(s["bbox_y"])
        bbox_w, bbox_h = float(s["bbox_width"]), float(s["bbox_height"])
        iw, ih = int(s["image_width"]), int(s["image_height"])
        x1, y1, x2, y2, patch_size = reconstruct_patch(bbox_x, bbox_y, bbox_w, bbox_h, iw, ih)
        recorded = int(float(a["patch_size_pixels"]))
        if patch_size != recorded or patch_size != spec["patch_size"]:
            raise RuntimeError(
                f"{fn} patch mismatch reconstructed={patch_size} recorded={recorded} expected={spec['patch_size']}"
            )
        left, top, right, bottom = compute_context_crop(iw, ih, x1, y1, x2, y2)
        attacked_path = ATTACK_02 / "patched_images" / fn
        print(f"Inpainting {fn} ...")
        status = run_inpaint_one(
            out_dir,
            attacked_path,
            {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "left": left, "top": top, "right": right, "bottom": bottom},
        )
        restored_path = out_dir / "06_restored_full.jpg"
        if status.get("status") != "OK" or not restored_path.is_file():
            sd_failures += 1
            row = {
                "image_id": stem,
                "filename": fn,
                "role": spec["role"],
                "target_class": spec["target_class"],
                "patch_size_pixels": patch_size,
                "patch_x1": x1,
                "patch_y1": y1,
                "patch_x2": x2,
                "patch_y2": y2,
                "bbox": (bbox_x, bbox_y, bbox_w, bbox_h),
                "image_size": (iw, ih),
                "clean_detected": a["clean_detected"],
                "attacked_detected": a["attacked_detected"],
                "inpainted_detected": "FAILED",
                "inpainted_confidence": 0.0,
                "inpainted_iou": 0.0,
                "sticker_removed": "FAILED",
                "hallucination": "FAILED",
                "flat_gray": "FAILED",
                "visible_seam": "FAILED",
                "natural_restoration": "FAILED",
                "sd_status": "FAILED",
                "sd_error": status.get("error", ""),
            }
            rows.append(row)
            continue

        attacked_img = Image.open(attacked_path).convert("RGB")
        restored_img = Image.open(restored_path).convert("RGB")
        flags = visual_flags(
            np.array(attacked_img)[y1:y2, x1:x2],
            np.array(restored_img)[y1:y2, x1:x2],
        )
        row = {
            "image_id": stem,
            "filename": fn,
            "role": spec["role"],
            "target_class": spec["target_class"],
            "patch_size_pixels": patch_size,
            "patch_x1": x1,
            "patch_y1": y1,
            "patch_x2": x2,
            "patch_y2": y2,
            "bbox": (bbox_x, bbox_y, bbox_w, bbox_h),
            "image_size": (iw, ih),
            "clean_detected": a["clean_detected"],
            "attacked_detected": a["attacked_detected"],
            "inpainted_detected": "PENDING",
            "inpainted_confidence": 0.0,
            "inpainted_iou": 0.0,
            **flags,
            "sd_status": "OK",
            "device": status.get("device", "unknown"),
        }
        rows.append(row)

        # old vs new visual (read-only old pilot restored)
        old_path = OLD_PILOT / "restored_images" / fn
        if old_path.is_file():
            old = Image.open(old_path).convert("RGB")
            pad = max(40, patch_size)
            zx1, zy1 = max(0, x1 - pad), max(0, y1 - pad)
            zx2, zy2 = min(iw, x2 + pad), min(ih, y2 + pad)
            comp = hstack(
                [
                    panel(attacked_img.crop((zx1, zy1, zx2, zy2)), "ATTACKED"),
                    panel(old.crop((zx1, zy1, zx2, zy2)), "OLD IMG2IMG 0.55"),
                    panel(restored_img.crop((zx1, zy1, zx2, zy2)), "NEW INPAINT 1.00"),
                ]
            )
            comp.save(out_dir / "09_old_vs_new.jpg", quality=95)

    # ---- Hyper-YOLO on restored images ----
    print("Running Hyper-YOLO...")
    sys.path.insert(0, str(EXPERIMENT / "scripts"))
    from pipeline_core import (  # noqa: WPS433
        evaluate_target_match,
        load_yolo,
        parse_label_file,
        run_yolo_inference,
    )

    yolo_src = RESULTS / "_yolo_batch_inputs"
    if yolo_src.exists():
        shutil.rmtree(yolo_src)
    yolo_src.mkdir(parents=True)
    for row in rows:
        if row["sd_status"] != "OK":
            continue
        src = PER_IMAGE / row["image_id"] / "06_restored_full.jpg"
        shutil.copy2(src, yolo_src / row["filename"])

    model = load_yolo()
    pred_images = RESULTS / "_yolo_images"
    pred_labels = RESULTS / "_yolo_labels"
    run_yolo_inference(model, yolo_src, pred_images, pred_labels, RESULTS / "_yolo_parent")

    for row in rows:
        if row["sd_status"] != "OK":
            continue
        stem = row["image_id"]
        label = pred_labels / f"{stem}.txt"
        iw, ih = row["image_size"]
        dets = parse_label_file(label, iw, ih, model)
        match = evaluate_target_match(dets, row["target_class"], row["bbox"])
        row["inpainted_detected"] = str(match["detected"])
        row["inpainted_confidence"] = float(match["confidence"])
        row["inpainted_iou"] = float(match["iou"])

        # copy yolo viz
        yolo_viz = pred_images / row["filename"]
        if yolo_viz.is_file():
            shutil.copy2(yolo_viz, PER_IMAGE / stem / "07_hyper_yolo_prediction.jpg")

        recovered = row["role"] == "success" and row["inpainted_detected"] == "Yes"
        preserved = row["role"] == "control" and row["inpainted_detected"] == "Yes"
        regressed = row["role"] == "control" and row["inpainted_detected"] != "Yes"
        row["recovered"] = "Yes" if recovered else "No"
        row["control_preserved"] = "Yes" if preserved else "No"
        row["control_regressed"] = "Yes" if regressed else "No"

        if row["role"] == "success":
            row["row_label"] = "SUCCESS - RECOVERED" if recovered else "SUCCESS - NOT RECOVERED"
        else:
            row["row_label"] = "CONTROL - PRESERVED" if preserved else "CONTROL - REGRESSED"
        if row["hallucination"] == "YES":
            row["row_label"] += " | HALLUCINATION"

        attacked_img = Image.open(PER_IMAGE / stem / "01_attacked.jpg").convert("RGB")
        restored_img = Image.open(PER_IMAGE / stem / "06_restored_full.jpg").convert("RGB")
        yolo_img = None
        yp = PER_IMAGE / stem / "07_hyper_yolo_prediction.jpg"
        if yp.is_file():
            yolo_img = Image.open(yp).convert("RGB")
        draw_verification(
            attacked_img,
            restored_img,
            yolo_img,
            PER_IMAGE / stem / "08_verification.jpg",
            row,
        )

        # category copies of verification only
        verify = PER_IMAGE / stem / "08_verification.jpg"
        name = f"{stem}_verification.jpg"
        if recovered:
            shutil.copy2(verify, RESULTS / "recovered_successes" / name)
        if row["role"] == "success" and not recovered:
            shutil.copy2(verify, RESULTS / "failed_successes" / name)
        if preserved:
            shutil.copy2(verify, RESULTS / "preserved_controls" / name)
        if regressed:
            shutil.copy2(verify, RESULTS / "regressed_controls" / name)
        if row["hallucination"] == "YES":
            shutil.copy2(verify, RESULTS / "hallucination_cases" / name)

        # per-image result.json
        result_json = {
            k: v
            for k, v in row.items()
            if k not in {"bbox", "image_size"}
        }
        (PER_IMAGE / stem / "result.json").write_text(json.dumps(result_json, indent=2), encoding="utf-8")

    # ---- CSV ----
    csv_path = RESULTS / "pilot_results.csv"
    fields = [
        "image_id",
        "filename",
        "role",
        "target_class",
        "patch_size_pixels",
        "clean_detected",
        "attacked_detected",
        "inpainted_detected",
        "inpainted_confidence",
        "inpainted_iou",
        "sticker_removed",
        "hallucination",
        "flat_gray",
        "visible_seam",
        "natural_restoration",
        "recovered",
        "control_preserved",
        "control_regressed",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})

    # ---- summary ----
    successes = [r for r in rows if r["role"] == "success"]
    controls = [r for r in rows if r["role"] == "control"]
    recovered_n = sum(1 for r in successes if r.get("recovered") == "Yes")
    preserved_n = sum(1 for r in controls if r.get("control_preserved") == "Yes")
    regressed_n = sum(1 for r in controls if r.get("control_regressed") == "Yes")
    sticker_n = sum(1 for r in rows if r.get("sticker_removed") == "YES")
    halluc_n = sum(1 for r in rows if r.get("hallucination") == "YES")
    flat_n = sum(1 for r in rows if r.get("flat_gray") == "YES")
    natural_n = sum(1 for r in rows if r.get("natural_restoration") == "YES")
    det_rate = recovered_n / len(successes) * 100 if successes else 0.0
    pres_rate = preserved_n / len(controls) * 100 if controls else 0.0
    regr_rate = regressed_n / len(controls) * 100 if controls else 0.0
    large = next(r for r in rows if r["filename"] == "000000407083.jpg")

    summary_lines = [
        "STRUCTURED SD2 INPAINTING PILOT SUMMARY (strength=1.00)",
        "=" * 72,
        "NOTE: configuration sanity check only. Not full-dataset defense statistics.",
        "Metric name: Detection Recovery Rate (not ASR after defense).",
        "",
        f"total images: {len(rows)}",
        f"success cases: {len(successes)}",
        f"control cases: {len(controls)}",
        f"SD failures: {sd_failures}",
        "",
        f"sticker removed count: {sticker_n}",
        f"hallucination count: {halluc_n}",
        f"flat/gray count: {flat_n}",
        f"visible seam count: {sum(1 for r in rows if r.get('visible_seam')=='YES')}",
        f"natural restoration count: {natural_n}",
        "",
        f"success recovered count: {recovered_n} / {len(successes)}",
        f"Detection Recovery Rate: {det_rate:.2f}%",
        "",
        f"control preserved count: {preserved_n} / {len(controls)}",
        f"Control Preservation Rate: {pres_rate:.2f}%",
        f"control regression count: {regressed_n}",
        f"Control Regression Rate: {regr_rate:.2f}%",
        "",
        "person results:",
    ]
    for r in rows:
        if r["target_class"] == "person":
            summary_lines.append(
                f"  - {r['filename']} | {r['role']} | det={r['inpainted_detected']} "
                f"halluc={r['hallucination']} natural={r['natural_restoration']}"
            )
    summary_lines.append("car results:")
    for r in rows:
        if r["target_class"] == "car":
            summary_lines.append(
                f"  - {r['filename']} | {r['role']} | det={r['inpainted_detected']} "
                f"halluc={r['hallucination']} natural={r['natural_restoration']}"
            )
    summary_lines.extend(
        [
            "",
            "large-patch result for 000000407083.jpg:",
            f"  detected={large['inpainted_detected']} conf={large['inpainted_confidence']:.4f} "
            f"iou={large['inpainted_iou']:.4f}",
            f"  sticker_removed={large['sticker_removed']} hallucination={large['hallucination']} "
            f"flat_gray={large['flat_gray']} natural={large['natural_restoration']}",
            f"  label={large.get('row_label','')}",
            "",
            f"output_root: {PILOT_ROOT}",
        ]
    )
    (RESULTS / "pilot_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    # ---- overview ----
    overview_rows = []
    for row in rows:
        if row["sd_status"] != "OK":
            continue
        stem = row["image_id"]
        attacked_img = Image.open(PER_IMAGE / stem / "01_attacked.jpg").convert("RGB")
        restored_img = Image.open(PER_IMAGE / stem / "06_restored_full.jpg").convert("RGB")
        yolo_path = PER_IMAGE / stem / "07_hyper_yolo_prediction.jpg"
        yolo_img = Image.open(yolo_path).convert("RGB") if yolo_path.is_file() else Image.new("RGB", attacked_img.size, (40, 0, 0))
        x1, y1, x2, y2 = row["patch_x1"], row["patch_y1"], row["patch_x2"], row["patch_y2"]
        pad = max(40, row["patch_size_pixels"])
        iw, ih = attacked_img.size
        zx1, zy1 = max(0, x1 - pad), max(0, y1 - pad)
        zx2, zy2 = min(iw, x2 + pad), min(ih, y2 + pad)
        line = hstack(
            [
                panel(attacked_img.crop((zx1, zy1, zx2, zy2)), "ATTACKED"),
                panel(restored_img.crop((zx1, zy1, zx2, zy2)), "NEW INPAINT 1.00"),
                panel(yolo_img.crop((zx1, zy1, zx2, zy2)), "HYPER-YOLO"),
            ]
        )
        header = Image.new("RGB", (line.size[0], 28), (255, 255, 255))
        ImageDraw.Draw(header).text((6, 7), f"{row['filename']} | {row['row_label']}", fill=(0, 0, 0))
        block = Image.new("RGB", (line.size[0], line.size[1] + 28), (255, 255, 255))
        block.paste(header, (0, 0))
        block.paste(line, (0, 28))
        overview_rows.append(block)
    if overview_rows:
        oh = sum(b.size[1] for b in overview_rows) + 4 * (len(overview_rows) - 1)
        ow = max(b.size[0] for b in overview_rows)
        overview = Image.new("RGB", (ow, oh), (255, 255, 255))
        y = 0
        for b in overview_rows:
            overview.paste(b, (0, y))
            y += b.size[1] + 4
        overview.save(RESULTS / "pilot_overview.jpg", quality=95)

    # ---- old vs new overview ----
    ovn_rows = []
    for row in rows:
        if row["sd_status"] != "OK":
            continue
        stem = row["image_id"]
        attacked_img = Image.open(PER_IMAGE / stem / "01_attacked.jpg").convert("RGB")
        restored_img = Image.open(PER_IMAGE / stem / "06_restored_full.jpg").convert("RGB")
        old_path = OLD_PILOT / "restored_images" / row["filename"]
        if not old_path.is_file():
            continue
        old = Image.open(old_path).convert("RGB")
        x1, y1, x2, y2 = row["patch_x1"], row["patch_y1"], row["patch_x2"], row["patch_y2"]
        pad = max(40, row["patch_size_pixels"])
        iw, ih = attacked_img.size
        zx1, zy1 = max(0, x1 - pad), max(0, y1 - pad)
        zx2, zy2 = min(iw, x2 + pad), min(ih, y2 + pad)
        line = hstack(
            [
                panel(attacked_img.crop((zx1, zy1, zx2, zy2)), "ATTACKED"),
                panel(old.crop((zx1, zy1, zx2, zy2)), "OLD IMG2IMG 0.55"),
                panel(restored_img.crop((zx1, zy1, zx2, zy2)), "NEW INPAINT 1.00"),
            ]
        )
        header = Image.new("RGB", (line.size[0], 28), (255, 255, 255))
        ImageDraw.Draw(header).text((6, 7), f"{row['filename']} | {row['role']} | {row['target_class']}", fill=(0, 0, 0))
        block = Image.new("RGB", (line.size[0], line.size[1] + 28), (255, 255, 255))
        block.paste(header, (0, 0))
        block.paste(line, (0, 28))
        ovn_rows.append(block)
    if ovn_rows:
        oh = sum(b.size[1] for b in ovn_rows) + 4 * (len(ovn_rows) - 1)
        ow = max(b.size[0] for b in ovn_rows)
        overview = Image.new("RGB", (ow, oh), (255, 255, 255))
        y = 0
        for b in ovn_rows:
            overview.paste(b, (0, y))
            y += b.size[1] + 4
        overview.save(RESULTS / "old_vs_new_overview.jpg", quality=95)

    (RESULTS / "README.txt").write_text(
        "\n".join(
            [
                "SD2 inpainting pilot result categories",
                "",
                "recovered_successes =",
                "targets that attack successfully suppressed but new inpainting restored",
                "",
                "failed_successes =",
                "attack-success targets that remain undetected after restoration",
                "",
                "preserved_controls =",
                "targets detectable after attack and still detectable after restoration",
                "",
                "regressed_controls =",
                "targets detectable after attack but lost after restoration",
                "",
                "hallucination_cases =",
                "restorations containing obvious unrelated/generated structures",
                "",
                "Verification images are COPIED into these folders; originals remain under per_image/.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print("\n".join(summary_lines))
    print(f"PILOT_ROOT={PILOT_ROOT}")


if __name__ == "__main__":
    main()
