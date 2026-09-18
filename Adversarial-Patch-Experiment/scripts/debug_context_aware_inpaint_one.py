#!/usr/bin/env python3
"""ONE-IMAGE context-aware SD2 inpainting test (000000407083 only).

Compares context/guidance variants A/B/C. Does not modify attacks,
mitigation_01, Hyper-YOLO, Stable-Diffusion-Patch, or prior debug outputs.
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

EXPERIMENT = Path(__file__).resolve().parents[1]
ATTACK_02 = EXPERIMENT.parent / "Hyper-YOLO" / "attacks" / "attack_02"
MODEL = EXPERIMENT / "mitigation_models" / "stable-diffusion-2-inpainting"
OUT_ROOT = EXPERIMENT / "mitigation_debug" / "context_aware_inpainting_one_image" / "000000407083"
SOURCE = OUT_ROOT / "source"
RESULTS = OUT_ROOT / "results"

SD_PYTHON = EXPERIMENT.parent / "Stable-Diffusion-Patch" / ".venv" / "Scripts" / "python.exe"
HYPER_PYTHON = EXPERIMENT.parent / "Hyper-YOLO" / ".venv" / "Scripts" / "python.exe"

FILENAME = "000000407083.jpg"
STEM = "000000407083"
STRENGTH = 1.00
STEPS = 30
SEED = 20260727
RESOLUTION = 512
FEATHER = 4
MASK_PADDING = 4
PATCH_SCALE = 0.30
MIN_SIDE = 128

PROMPT = (
    "seamless realistic photograph, natural texture consistent with surrounding image"
)
NEG = (
    "text, letters, logo, watermark, sticker, artificial pattern, "
    "unrelated object, geometric shape, scene change"
)

VARIANTS = [
    {
        "key": "A",
        "folder": "variant_A_context3_guidance75",
        "label": "A - Context 3 / Guidance 7.5",
        "context_factor": 3.0,
        "guidance_scale": 7.5,
    },
    {
        "key": "B",
        "folder": "variant_B_context5_guidance45",
        "label": "B - Context 5 / Guidance 4.5",
        "context_factor": 5.0,
        "guidance_scale": 4.5,
    },
    {
        "key": "C",
        "folder": "variant_C_context7_guidance45",
        "label": "C - Context 7 / Guidance 4.5",
        "context_factor": 7.0,
        "guidance_scale": 4.5,
    },
]


def reconstruct_patch(bbox_x, bbox_y, bbox_w, bbox_h, image_w, image_h):
    patch_size = max(1, int(round(PATCH_SCALE * min(bbox_w, bbox_h))))
    cx = bbox_x + bbox_w / 2.0
    cy = bbox_y + bbox_h / 2.0
    px = max(0.0, min(cx - patch_size / 2.0, image_w - patch_size))
    py = max(0.0, min(cy - patch_size / 2.0, image_h - patch_size))
    x1 = int(round(px))
    y1 = int(round(py))
    return x1, y1, x1 + patch_size, y1 + patch_size, patch_size


def expand_mask(x1, y1, x2, y2, w, h, pad):
    return (
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(w, x2 + pad),
        min(h, y2 + pad),
    )


def compute_context_crop(w, h, x1, y1, x2, y2, factor):
    """Centered crop around patch; preserve aspect by independent clamp (no stretch)."""
    pw, ph = max(1, x2 - x1), max(1, y2 - y1)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    des_w = max(MIN_SIDE, int(round(factor * pw)))
    des_h = max(MIN_SIDE, int(round(factor * ph)))
    des_w = min(des_w, w)
    des_h = min(des_h, h)
    left = max(0, min(int(round(cx - des_w / 2.0)), w - des_w))
    top = max(0, min(int(round(cy - des_h / 2.0)), h - des_h))
    return left, top, left + des_w, top + des_h


def feathered_alpha(hh, ww, feather):
    alpha = np.ones((hh, ww), dtype=np.float32)
    feather = max(0, int(feather))
    if feather <= 0 or hh < 2 or ww < 2:
        return alpha
    for i in range(feather):
        weight = (i + 1) / (feather + 1)
        alpha[i, :] = np.minimum(alpha[i, :], weight)
        alpha[-(i + 1), :] = np.minimum(alpha[-(i + 1), :], weight)
        alpha[:, i] = np.minimum(alpha[:, i], weight)
        alpha[:, -(i + 1)] = np.minimum(alpha[:, -(i + 1)], weight)
    return alpha


def letterbox_to_square(rgb: Image.Image, mask: Image.Image, size: int):
    """Pad then resize without geometric distortion. Returns mapping for inverse."""
    cw, ch = rgb.size
    scale = size / float(max(cw, ch))
    new_w = max(1, int(round(cw * scale)))
    new_h = max(1, int(round(ch * scale)))
    rgb_r = rgb.resize((new_w, new_h), Image.Resampling.LANCZOS)
    mask_r = mask.resize((new_w, new_h), Image.Resampling.NEAREST)
    canvas_rgb = Image.new("RGB", (size, size), (0, 0, 0))
    canvas_mask = Image.new("L", (size, size), 0)
    ox = (size - new_w) // 2
    oy = (size - new_h) // 2
    canvas_rgb.paste(rgb_r, (ox, oy))
    canvas_mask.paste(mask_r, (ox, oy))
    mapping = {
        "crop_w": cw,
        "crop_h": ch,
        "scaled_w": new_w,
        "scaled_h": new_h,
        "pad_x": ox,
        "pad_y": oy,
        "resolution": size,
        "scale": scale,
        "method": "letterbox_then_inverse_crop",
    }
    return canvas_rgb, canvas_mask, mapping


def inverse_letterbox(out_512: Image.Image, mapping: dict) -> Image.Image:
    ox, oy = mapping["pad_x"], mapping["pad_y"]
    nw, nh = mapping["scaled_w"], mapping["scaled_h"]
    cw, ch = mapping["crop_w"], mapping["crop_h"]
    region = out_512.crop((ox, oy, ox + nw, oy + nh))
    return region.resize((cw, ch), Image.Resampling.LANCZOS)


def visual_quality(attacked_region: np.ndarray, restored_region: np.ndarray, context_ring: np.ndarray | None):
    atk = attacked_region.astype(np.float32)
    res = restored_region.astype(np.float32)
    mae = float(np.mean(np.abs(atk - res)))
    res_chroma = float(np.mean(np.max(res, axis=2) - np.min(res, axis=2)))
    sticker_remain = mae < 25 and res_chroma > 40
    sticker_removed = "NO" if sticker_remain else "YES"

    std = float(np.std(res))
    flat_gray = "YES" if std < 22 else "NO"

    gray = np.mean(res, axis=2).astype(np.float32)
    gy, gx = np.gradient(gray)
    edge = float(np.mean(np.sqrt(gx * gx + gy * gy)))

    # Text-like: strong horizontal/vertical edge anisotropy
    abs_gx = float(np.mean(np.abs(gx)))
    abs_gy = float(np.mean(np.abs(gy)))
    aniso = abs_gx / (abs_gy + 1e-6)
    text_like = "YES" if mae >= 20 and ((aniso > 1.55) or (aniso < 0.65)) and edge > 10 else "NO"

    # Unrelated / hallucination: large change + high structure, optionally dissimilar to ring
    ring_diff = 0.0
    if context_ring is not None and context_ring.size > 0:
        ring_mean = context_ring.astype(np.float32).reshape(-1, 3).mean(axis=0)
        res_mean = res.reshape(-1, 3).mean(axis=0)
        ring_diff = float(np.linalg.norm(ring_mean - res_mean))
    halluc = "YES" if mae >= 25 and edge > 12 and std > 28 else "NO"
    unrelated = "YES" if (halluc == "YES" and ring_diff > 35) or (mae >= 40 and edge > 14) else "NO"
    if unrelated == "YES":
        halluc = "YES"

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

    natural = (
        "YES"
        if sticker_removed == "YES"
        and flat_gray == "NO"
        and halluc == "NO"
        and unrelated == "NO"
        and text_like == "NO"
        else "NO"
    )
    return {
        "sticker_removed": sticker_removed,
        "obvious_hallucination": halluc,
        "text_like_artifact": text_like,
        "flat_gray_fill": flat_gray,
        "visible_boundary_seam": seam,
        "unrelated_object_or_scene": unrelated,
        "natural_context_match": natural,
        "mae_attacked_vs_restored": mae,
        "restored_std": std,
        "edge_magnitude": edge,
        "ring_color_diff": ring_diff,
    }


def panel(img: Image.Image, label: str, size=(320, 260)) -> Image.Image:
    canvas = Image.new("RGB", size, (245, 245, 245))
    thumb = img.copy()
    thumb.thumbnail((size[0] - 8, size[1] - 34), Image.Resampling.LANCZOS)
    canvas.paste(thumb, ((size[0] - thumb.size[0]) // 2, 30 + (size[1] - 34 - thumb.size[1]) // 2))
    d = ImageDraw.Draw(canvas)
    d.rectangle([0, 0, size[0], 26], fill=(20, 20, 20))
    d.text((6, 6), label[:48], fill=(255, 255, 255))
    return canvas


def vstack(images):
    w = max(im.size[0] for im in images)
    canvas = Image.new("RGB", (w, sum(im.size[1] for im in images) + 4 * (len(images) - 1)), (255, 255, 255))
    y = 0
    for im in images:
        canvas.paste(im, (0, y))
        y += im.size[1] + 4
    return canvas


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
variants = args["variants"]
prompt = args["prompt"]
neg = args["neg"]
strength = float(args["strength"])
seed = int(args["seed"])
steps = int(args["steps"])
resolution = int(args["resolution"])
feather = int(args["feather"])

def feathered_alpha(hh, ww, feather):
    alpha = np.ones((hh, ww), dtype=np.float32)
    feather = max(0, int(feather))
    if feather <= 0 or hh < 2 or ww < 2:
        return alpha
    for i in range(feather):
        weight = (i + 1) / (feather + 1)
        alpha[i, :] = np.minimum(alpha[i, :], weight)
        alpha[-(i + 1), :] = np.minimum(alpha[-(i + 1), :], weight)
        alpha[:, i] = np.minimum(alpha[:, i], weight)
        alpha[:, -(i + 1)] = np.minimum(alpha[:, -(i + 1)], weight)
    return alpha

def letterbox_to_square(rgb, mask, size):
    cw, ch = rgb.size
    scale = size / float(max(cw, ch))
    new_w = max(1, int(round(cw * scale)))
    new_h = max(1, int(round(ch * scale)))
    rgb_r = rgb.resize((new_w, new_h), Image.Resampling.LANCZOS)
    mask_r = mask.resize((new_w, new_h), Image.Resampling.NEAREST)
    canvas_rgb = Image.new("RGB", (size, size), (0, 0, 0))
    canvas_mask = Image.new("L", (size, size), 0)
    ox = (size - new_w) // 2
    oy = (size - new_h) // 2
    canvas_rgb.paste(rgb_r, (ox, oy))
    canvas_mask.paste(mask_r, (ox, oy))
    mapping = {"crop_w": cw, "crop_h": ch, "scaled_w": new_w, "scaled_h": new_h,
               "pad_x": ox, "pad_y": oy, "resolution": size, "scale": scale}
    return canvas_rgb, canvas_mask, mapping

def inverse_letterbox(out_512, mapping):
    ox, oy = mapping["pad_x"], mapping["pad_y"]
    nw, nh = mapping["scaled_w"], mapping["scaled_h"]
    cw, ch = mapping["crop_w"], mapping["crop_h"]
    region = out_512.crop((ox, oy, ox + nw, oy + nh))
    return region.resize((cw, ch), Image.Resampling.LANCZOS)

attacked = Image.open(attacked_path).convert("RGB")
w, h = attacked.size
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    model, torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    variant="fp16", use_safetensors=True, local_files_only=True,
)
if hasattr(pipe, "enable_attention_slicing"):
    pipe.enable_attention_slicing()
if device == "cuda" and hasattr(pipe, "enable_model_cpu_offload"):
    pipe.enable_model_cpu_offload()
else:
    pipe = pipe.to(device)

outputs = []
for v in variants:
    out_dir = Path(v["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ex1, ey1, ex2, ey2 = v["ex1"], v["ey1"], v["ex2"], v["ey2"]
    left, top, right, bottom = v["left"], v["top"], v["right"], v["bottom"]
    guidance = float(v["guidance"])
    mask_full = Image.new("L", (w, h), 0)
    m = np.array(mask_full)
    m[ey1:ey2, ex1:ex2] = 255
    mask_full = Image.fromarray(m)
    crop = attacked.crop((left, top, right, bottom))
    mask_crop = mask_full.crop((left, top, right, bottom))
    crop_512, mask_512, mapping = letterbox_to_square(crop, mask_crop, resolution)
    crop.save(out_dir / "context_crop.jpg", quality=95)
    mask_512.save(out_dir / "mask_512.png")
    try:
        gen = torch.Generator(device="cpu").manual_seed(seed)
        out = pipe(
            prompt=prompt, negative_prompt=neg, image=crop_512, mask_image=mask_512,
            strength=strength, guidance_scale=guidance, num_inference_steps=steps, generator=gen,
        ).images[0]
        out.save(out_dir / "raw_inpaint_512.png")
        restored_crop = inverse_letterbox(out, mapping)
        rx1, ry1 = ex1 - left, ey1 - top
        rx2, ry2 = ex2 - left, ey2 - top
        patch_gen = np.array(restored_crop).astype(np.float32)[ry1:ry2, rx1:rx2]
        base = np.array(attacked).astype(np.float32)
        hh, ww = ey2 - ey1, ex2 - ex1
        if patch_gen.shape[0] != hh or patch_gen.shape[1] != ww:
            raise RuntimeError(f"shape mismatch {patch_gen.shape} vs {(hh, ww)}")
        alpha = feathered_alpha(hh, ww, feather)[:, :, None]
        base[ey1:ey2, ex1:ex2] = alpha * patch_gen + (1.0 - alpha) * base[ey1:ey2, ex1:ex2]
        Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)).save(out_dir / "restored_full.jpg", quality=95)
        (out_dir / "mapping.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")
        outputs.append({"key": v["key"], "status": "OK", "device": device, "mapping": mapping})
        print(f"VARIANT_{v['key']}_OK", flush=True)
    except Exception as e:
        (out_dir / "FAILED.txt").write_text(str(e), encoding="utf-8")
        outputs.append({"key": v["key"], "status": "FAILED", "error": str(e), "device": device})
        print(f"VARIANT_{v['key']}_FAILED {e}", flush=True)

print("RESULT_JSON:" + json.dumps({"device": device, "variants": outputs}))
'''


def load_metadata():
    attack = {
        r["filename"]: r
        for r in csv.DictReader((ATTACK_02 / "results" / "attack_results.csv").open(encoding="utf-8"))
    }
    selected = {
        r["filename"]: r
        for r in csv.DictReader((ATTACK_02 / "results" / "selected_images.csv").open(encoding="utf-8"))
    }
    return attack[FILENAME], selected[FILENAME]


def main():
    for d in [SOURCE, RESULTS] + [OUT_ROOT / v["folder"] for v in VARIANTS]:
        d.mkdir(parents=True, exist_ok=True)

    attack_row, selected = load_metadata()
    bbox = (
        float(selected["bbox_x"]),
        float(selected["bbox_y"]),
        float(selected["bbox_width"]),
        float(selected["bbox_height"]),
    )
    iw, ih = int(selected["image_width"]), int(selected["image_height"])
    x1, y1, x2, y2, patch_size = reconstruct_patch(*bbox, iw, ih)
    recorded = int(float(attack_row["patch_size_pixels"]))
    if patch_size != recorded:
        raise RuntimeError(f"Patch size mismatch reconstructed={patch_size} recorded={recorded}")

    ex1, ey1, ex2, ey2 = expand_mask(x1, y1, x2, y2, iw, ih, MASK_PADDING)
    attacked_path = ATTACK_02 / "patched_images" / FILENAME
    attacked = Image.open(attacked_path).convert("RGB")
    if attacked.size != (iw, ih):
        raise RuntimeError(f"Image size mismatch {attacked.size} vs {(iw, ih)}")

    # Source artifacts
    exact_mask = Image.new("L", (iw, ih), 0)
    em = np.array(exact_mask)
    em[y1:y2, x1:x2] = 255
    exact_mask = Image.fromarray(em)
    expanded_mask = Image.new("L", (iw, ih), 0)
    xm = np.array(expanded_mask)
    xm[ey1:ey2, ex1:ex2] = 255
    expanded_mask = Image.fromarray(xm)

    attacked.save(SOURCE / "attacked.jpg", quality=95)
    exact_mask.save(SOURCE / "exact_patch_mask.png")
    expanded_mask.save(SOURCE / "expanded_inpaint_mask.png")

    variant_payloads = []
    meta_variants = {}
    for v in VARIANTS:
        left, top, right, bottom = compute_context_crop(iw, ih, ex1, ey1, ex2, ey2, v["context_factor"])
        # Use expanded mask region for crop centering already via ex*
        # Recompute with expanded box as the "patch" for context
        left, top, right, bottom = compute_context_crop(iw, ih, ex1, ey1, ex2, ey2, v["context_factor"])
        out_dir = OUT_ROOT / v["folder"]
        mapping_preview = {
            "context_factor": v["context_factor"],
            "guidance_scale": v["guidance_scale"],
            "crop": {"left": left, "top": top, "right": right, "bottom": bottom,
                     "width": right - left, "height": bottom - top},
            "expanded_region_in_crop": {
                "rx1": ex1 - left,
                "ry1": ey1 - top,
                "rx2": ex2 - left,
                "ry2": ey2 - top,
            },
            "letterbox_note": "crop letterboxed to 512 without stretch; inverse maps only content region",
        }
        meta_variants[v["key"]] = mapping_preview
        variant_payloads.append(
            {
                "key": v["key"],
                "out_dir": str(out_dir),
                "ex1": ex1,
                "ey1": ey1,
                "ex2": ex2,
                "ey2": ey2,
                "left": left,
                "top": top,
                "right": right,
                "bottom": bottom,
                "guidance": v["guidance_scale"],
            }
        )

    metadata = {
        "filename": FILENAME,
        "target_class": attack_row["target_class"],
        "role": "success",
        "gt_bbox_xywh": {"x": bbox[0], "y": bbox[1], "w": bbox[2], "h": bbox[3]},
        "image_size": {"width": iw, "height": ih},
        "exact_patch_coordinates": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "expanded_mask_coordinates": {"x1": ex1, "y1": ey1, "x2": ex2, "y2": ey2},
        "mask_padding_pixels": MASK_PADDING,
        "patch_size_pixels": patch_size,
        "clean_detected": attack_row["clean_detected"],
        "attacked_detected": attack_row["attacked_detected"],
        "seed": SEED,
        "strength": STRENGTH,
        "steps": STEPS,
        "prompt": PROMPT,
        "negative_prompt": NEG,
        "clean_pixels_used": False,
        "neutralization_used": False,
        "variants": meta_variants,
    }
    (SOURCE / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Running SD2 inpainting for variants A/B/C ...")
    payload = {
        "model": str(MODEL),
        "attacked": str(attacked_path),
        "variants": variant_payloads,
        "prompt": PROMPT,
        "neg": NEG,
        "strength": STRENGTH,
        "seed": SEED,
        "steps": STEPS,
        "resolution": RESOLUTION,
        "feather": FEATHER,
    }
    proc = subprocess.run(
        [str(SD_PYTHON), "-c", INPAINT_WORKER, json.dumps(payload)],
        capture_output=True,
        text=True,
        check=False,
    )
    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr)
        raise RuntimeError(f"Inpaint worker failed: {(proc.stderr or proc.stdout)[-2000:]}")

    sd_json = None
    for line in (proc.stdout or "").splitlines():
        if line.startswith("RESULT_JSON:"):
            sd_json = json.loads(line[len("RESULT_JSON:") :])
    if sd_json is None:
        raise RuntimeError("No RESULT_JSON from inpaint worker")

    sd_by_key = {v["key"]: v for v in sd_json["variants"]}

    # Zoom crops + patch_zoom
    pad = max(40, patch_size)
    zx1, zy1 = max(0, x1 - pad), max(0, y1 - pad)
    zx2, zy2 = min(iw, x2 + pad), min(ih, y2 + pad)

    rows = []
    for v in VARIANTS:
        out_dir = OUT_ROOT / v["folder"]
        sd = sd_by_key.get(v["key"], {"status": "FAILED", "error": "missing"})
        row = {
            "variant": v["key"],
            "folder": v["folder"],
            "label": v["label"],
            "context_factor": v["context_factor"],
            "guidance_scale": v["guidance_scale"],
            "strength": STRENGTH,
            "mask_padding": MASK_PADDING,
            "sd_status": sd.get("status", "FAILED"),
            "sd_error": sd.get("error", ""),
            "target_detected": "FAILED",
            "confidence": 0.0,
            "iou": 0.0,
            "sticker_removed": "FAILED",
            "hallucination": "FAILED",
            "text_artifact": "FAILED",
            "flat_gray": "FAILED",
            "visible_seam": "FAILED",
            "unrelated_content": "FAILED",
            "natural_context_match": "FAILED",
        }
        if sd.get("status") != "OK" or not (out_dir / "restored_full.jpg").is_file():
            rows.append(row)
            (out_dir / "result.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
            continue

        restored = Image.open(out_dir / "restored_full.jpg").convert("RGB")
        restored.crop((zx1, zy1, zx2, zy2)).save(out_dir / "patch_zoom.jpg", quality=95)

        # quality on expanded region
        atk_np = np.array(attacked)
        res_np = np.array(restored)
        region_atk = atk_np[ey1:ey2, ex1:ex2]
        region_res = res_np[ey1:ey2, ex1:ex2]
        ring = None
        ry0, ry1r = max(0, ey1 - 8), min(ih, ey2 + 8)
        rx0, rx1r = max(0, ex1 - 8), min(iw, ex2 + 8)
        ring_mask = np.zeros((ry1r - ry0, rx1r - rx0), dtype=bool)
        # mark ring pixels outside expanded region
        local = np.ones_like(ring_mask)
        local[(ey1 - ry0) : (ey2 - ry0), (ex1 - rx0) : (ex2 - rx0)] = False
        ring_pixels = atk_np[ry0:ry1r, rx0:rx1r][local]
        if ring_pixels.size > 0:
            ring = ring_pixels

        flags = visual_quality(region_atk, region_res, ring)
        row.update(
            {
                "sticker_removed": flags["sticker_removed"],
                "hallucination": flags["obvious_hallucination"],
                "text_artifact": flags["text_like_artifact"],
                "flat_gray": flags["flat_gray_fill"],
                "visible_seam": flags["visible_boundary_seam"],
                "unrelated_content": flags["unrelated_object_or_scene"],
                "natural_context_match": flags["natural_context_match"],
                "quality_metrics": flags,
                "mapping": sd.get("mapping"),
                "crop": meta_variants[v["key"]]["crop"],
            }
        )
        rows.append(row)

    # Hyper-YOLO on three restored images
    print("Running Hyper-YOLO on three restored images...")
    sys.path.insert(0, str(EXPERIMENT / "scripts"))
    from pipeline_core import evaluate_target_match, load_yolo, parse_label_file, run_yolo_inference

    yolo_src = RESULTS / "_yolo_batch_inputs"
    if yolo_src.exists():
        shutil.rmtree(yolo_src)
    yolo_src.mkdir(parents=True)
    for row in rows:
        if row["sd_status"] != "OK":
            continue
        src = OUT_ROOT / row["folder"] / "restored_full.jpg"
        shutil.copy2(src, yolo_src / f"{row['variant']}_{FILENAME}")

    model = load_yolo()
    pred_images = RESULTS / "_yolo_images"
    pred_labels = RESULTS / "_yolo_labels"
    run_yolo_inference(model, yolo_src, pred_images, pred_labels, RESULTS / "_yolo_parent")

    for row in rows:
        if row["sd_status"] != "OK":
            continue
        stem = f"{row['variant']}_{STEM}"
        label = pred_labels / f"{stem}.txt"
        dets = parse_label_file(label, iw, ih, model)
        match = evaluate_target_match(dets, attack_row["target_class"], bbox)
        row["target_detected"] = "Yes" if match["detected"] in (True, "Yes", "yes") else str(match["detected"])
        if row["target_detected"] in ("True", "true"):
            row["target_detected"] = "Yes"
        if row["target_detected"] in ("False", "false"):
            row["target_detected"] = "No"
        row["confidence"] = float(match["confidence"])
        row["iou"] = float(match["iou"])
        yolo_viz = pred_images / f"{row['variant']}_{FILENAME}"
        if yolo_viz.is_file():
            shutil.copy2(yolo_viz, OUT_ROOT / row["folder"] / "hyper_yolo_prediction.jpg")

        # persist result.json without oversized nested only
        persist = {k: v for k, v in row.items() if k != "quality_metrics"}
        persist["quality_metrics"] = row.get("quality_metrics", {})
        (OUT_ROOT / row["folder"] / "result.json").write_text(json.dumps(persist, indent=2), encoding="utf-8")

    # Manual visual override pass: inspect restored patch zooms if present
    # (heuristics already applied; summary will note automatic classification)

    # CSV
    csv_path = RESULTS / "results.csv"
    fields = [
        "variant",
        "context_factor",
        "guidance_scale",
        "strength",
        "mask_padding",
        "target_detected",
        "confidence",
        "iou",
        "sticker_removed",
        "hallucination",
        "text_artifact",
        "flat_gray",
        "visible_seam",
        "unrelated_content",
        "natural_context_match",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})

    # Comparisons
    panels = [panel(attacked.crop((zx1, zy1, zx2, zy2)), "ATTACKED")]
    zoom_panels = [panel(attacked.crop((ex1, ey1, ex2, ey2)).resize((256, 256), Image.Resampling.NEAREST), "ATTACKED ZOOM")]
    for row in rows:
        out_dir = OUT_ROOT / row["folder"]
        if row["sd_status"] == "OK" and (out_dir / "restored_full.jpg").is_file():
            rest = Image.open(out_dir / "restored_full.jpg").convert("RGB")
            det = row["target_detected"]
            lab = f"{row['label']} | det={det}"
            panels.append(panel(rest.crop((zx1, zy1, zx2, zy2)), lab))
            zoom_panels.append(
                panel(rest.crop((ex1, ey1, ex2, ey2)).resize((256, 256), Image.Resampling.NEAREST), f"{row['variant']} ZOOM")
            )
        else:
            panels.append(panel(Image.new("RGB", (200, 200), (80, 0, 0)), f"{row['label']} FAILED"))
            zoom_panels.append(panel(Image.new("RGB", (200, 200), (80, 0, 0)), f"{row['variant']} FAILED"))

    vstack(panels).save(RESULTS / "comparison.jpg", quality=95)
    # horizontal zoom comparison is easier to inspect
    zw = sum(p.size[0] for p in zoom_panels) + 4 * (len(zoom_panels) - 1)
    zh = max(p.size[1] for p in zoom_panels)
    zcanvas = Image.new("RGB", (zw, zh), (255, 255, 255))
    x = 0
    for p in zoom_panels:
        zcanvas.paste(p, (x, 0))
        x += p.size[0] + 4
    zcanvas.save(RESULTS / "zoom_comparison.jpg", quality=95)

    # Decision
    ok_rows = [r for r in rows if r["sd_status"] == "OK"]

    def score(r):
        # Higher is better; hallucination heavily penalized
        s = 0
        s += 50 if r["sticker_removed"] == "YES" else 0
        s += 40 if r["hallucination"] != "YES" else -40
        s += 30 if r["unrelated_content"] != "YES" else -30
        s += 25 if r["natural_context_match"] == "YES" else 0
        s += 20 if r["target_detected"] == "Yes" else 0
        s += 10 * float(r.get("confidence") or 0)
        s += 5 * float(r.get("iou") or 0)
        s += 5 if r["visible_seam"] != "YES" else -5
        s += 5 if r["text_artifact"] != "YES" else -5
        s += 5 if r["flat_gray"] != "YES" else -5
        return s

    ranked = sorted(ok_rows, key=score, reverse=True)
    best = ranked[0] if ranked else None
    ref_a = next((r for r in rows if r["variant"] == "A"), None)
    better_than_a = []
    if ref_a and ref_a["sd_status"] == "OK":
        for r in ok_rows:
            if r["variant"] in ("B", "C") and score(r) > score(ref_a):
                # clear improvement requires less hallucination or better natural match, not only YOLO
                clear = (
                    (r["hallucination"] != "YES" and ref_a["hallucination"] == "YES")
                    or (r["natural_context_match"] == "YES" and ref_a["natural_context_match"] != "YES")
                    or (
                        r["hallucination"] == ref_a["hallucination"]
                        and r["unrelated_content"] == ref_a["unrelated_content"]
                        and r["natural_context_match"] == ref_a["natural_context_match"]
                        and r["target_detected"] == "Yes"
                        and ref_a["target_detected"] != "Yes"
                    )
                )
                if clear or score(r) >= score(ref_a) + 15:
                    better_than_a.append(r["variant"])

    least_halluc = min(ok_rows, key=lambda r: (r["hallucination"] == "YES", r["unrelated_content"] == "YES", -score(r))) if ok_rows else None
    best_natural = max(ok_rows, key=lambda r: (r["natural_context_match"] == "YES", score(r))) if ok_rows else None
    best_det = max(ok_rows, key=lambda r: (r["target_detected"] == "Yes", float(r["confidence"]), float(r["iou"]))) if ok_rows else None

    recommend = "A (reference)"
    if "C" in better_than_a:
        recommend = "C — context 7.0 / guidance 4.5"
    elif "B" in better_than_a:
        recommend = "B — context 5.0 / guidance 4.5"
    elif best and best["variant"] != "A" and score(best) > score(ref_a):
        recommend = f"{best['variant']} (marginal — review visually before pilot)"
    else:
        recommend = "Keep A for now OR review visuals; B/C not clearly better on auto metrics — WAIT FOR REVIEW"

    summary_lines = [
        "CONTEXT-AWARE SD2 INPAINTING ONE-IMAGE TEST",
        "=" * 72,
        f"image: {FILENAME}",
        f"target_class: {attack_row['target_class']}",
        f"exact_patch: ({x1},{y1})-({x2},{y2}) size={patch_size}",
        f"expanded_mask: ({ex1},{ey1})-({ex2},{ey2}) pad={MASK_PADDING}",
        f"seed={SEED} strength={STRENGTH} steps={STEPS}",
        f"prompt: {PROMPT}",
        "",
        "NOTE: One difficult image only. Not defense statistics.",
        "",
    ]
    for row in rows:
        summary_lines.extend(
            [
                f"VARIANT {row['variant']}: {row['label']}",
                f"  sd_status={row['sd_status']} crop={row.get('crop')}",
                f"  detected={row['target_detected']} conf={row['confidence']:.4f} iou={row['iou']:.4f}",
                f"  sticker_removed={row['sticker_removed']} halluc={row['hallucination']} "
                f"text={row['text_artifact']} flat={row['flat_gray']} seam={row['visible_seam']}",
                f"  unrelated={row['unrelated_content']} natural={row['natural_context_match']}",
                f"  auto_score={score(row) if row['sd_status']=='OK' else 'n/a'}",
                "",
            ]
        )

    summary_lines.extend(
        [
            f"least hallucination (auto): {least_halluc['variant'] if least_halluc else 'n/a'}",
            f"best natural context (auto): {best_natural['variant'] if best_natural else 'n/a'}",
            f"best target detection (auto): {best_det['variant'] if best_det else 'n/a'} "
            f"(det={best_det['target_detected'] if best_det else 'n/a'})",
            f"B or C clearly better than A (auto): {better_than_a if better_than_a else 'NO'}",
            f"recommended for later 6-image pilot: {recommend}",
            "",
            f"output_root: {OUT_ROOT}",
        ]
    )
    (RESULTS / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print("\n".join(summary_lines))
    print(f"OUT_ROOT={OUT_ROOT}")


if __name__ == "__main__":
    main()
