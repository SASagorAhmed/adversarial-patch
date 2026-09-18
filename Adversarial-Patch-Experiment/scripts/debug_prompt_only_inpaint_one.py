#!/usr/bin/env python3
"""PROMPT-ONLY SD2 inpainting test on 000000407083.

Locks geometry/settings to previous context-aware Variant B.
Changes ONLY positive/negative prompts across P1/P2/P3.
Does not modify attacks, mitigation_01, prior debug outputs, or dependencies.
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
ATTACK_02 = EXPERIMENT / "attacks" / "attack_02"
MODEL = EXPERIMENT / "mitigation_models" / "stable-diffusion-2-inpainting"
PREV_B = (
    EXPERIMENT
    / "mitigation_debug"
    / "context_aware_inpainting_one_image"
    / "000000407083"
    / "variant_B_context5_guidance45"
)
OUT_ROOT = EXPERIMENT / "mitigation_debug" / "prompt_only_inpainting_test" / "000000407083"
RESULTS = OUT_ROOT / "results"
SD_PYTHON = EXPERIMENT.parent / "Stable-Diffusion-Patch" / ".venv" / "Scripts" / "python.exe"

FILENAME = "000000407083.jpg"
STEM = "000000407083"
STRENGTH = 1.00
STEPS = 30
GUIDANCE = 4.5
CONTEXT = 5.0
SEED = 20260727
RESOLUTION = 512
FEATHER = 4
MASK_PADDING = 4
PATCH_SCALE = 0.30
MIN_SIDE = 128

# Locked geometry matching previous Variant B
EXACT_PATCH = (168, 241, 312, 385)
EXPANDED = (164, 237, 316, 389)
CROP = (0, 0, 480, 640)  # full image / letterbox as Variant B

PROMPTS = [
    {
        "key": "P1",
        "folder": "P1_local_continuity",
        "label": "P1 LOCAL CONTINUITY",
        "prompt": (
            "photorealistic restoration of the masked region only, seamlessly "
            "continue the exact colors, textures, edges, shapes, lighting and "
            "perspective from the immediately surrounding pixels, preserve the "
            "existing object and scene, restore only the missing local surface, "
            "natural continuous appearance"
        ),
        "neg": (
            "new object, extra object, unrelated object, new scene, landscape, "
            "text, letters, logo, watermark, sticker, adversarial pattern, "
            "geometric shape, abstract pattern, duplicated object, scene "
            "replacement, strong color change"
        ),
    },
    {
        "key": "P2",
        "folder": "P2_car_aware",
        "label": "P2 CAR-AWARE RESTORATION",
        "prompt": (
            "restore only the missing local part of the existing car, continue the "
            "same vehicle body surface, shape, paint, texture, reflections, lighting "
            "and perspective from the surrounding visible car, seamless "
            "photorealistic continuation, preserve the existing scene, no new "
            "content"
        ),
        "neg": (
            "new car, extra object, unrelated object, landscape, person, face, "
            "text, letters, logo, watermark, sticker, adversarial pattern, "
            "geometric object, scene change, artificial texture, strong color "
            "change"
        ),
    },
    {
        "key": "P3",
        "folder": "P3_minimal_generation",
        "label": "P3 STRICT MINIMAL GENERATION",
        "prompt": (
            "minimal local image restoration, continue only the immediately "
            "surrounding visual structure into the masked region, preserve existing "
            "colors, edges, texture and lighting, no semantic change, no added "
            "object, seamless photographic repair"
        ),
        "neg": (
            "new semantic content, new object, extra object, unrelated structure, "
            "landscape, face, text, letters, logo, sticker, watermark, abstract "
            "pattern, scene replacement, dramatic texture, dramatic color, "
            "artificial object"
        ),
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
    pw, ph = max(1, x2 - x1), max(1, y2 - y1)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    des_w = min(max(MIN_SIDE, int(round(factor * pw))), w)
    des_h = min(max(MIN_SIDE, int(round(factor * ph))), h)
    left = max(0, min(int(round(cx - des_w / 2.0)), w - des_w))
    top = max(0, min(int(round(cy - des_h / 2.0)), h - des_h))
    return left, top, left + des_w, top + des_h


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
strength = float(args["strength"])
seed = int(args["seed"])
steps = int(args["steps"])
guidance = float(args["guidance"])
resolution = int(args["resolution"])
feather = int(args["feather"])
ex1, ey1, ex2, ey2 = args["ex1"], args["ey1"], args["ex2"], args["ey2"]
left, top, right, bottom = args["left"], args["top"], args["right"], args["bottom"]

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
mask_full = Image.new("L", (w, h), 0)
m = np.array(mask_full); m[ey1:ey2, ex1:ex2] = 255; mask_full = Image.fromarray(m)
crop = attacked.crop((left, top, right, bottom))
mask_crop = mask_full.crop((left, top, right, bottom))
crop_512, mask_512, mapping = letterbox_to_square(crop, mask_crop, resolution)

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
    try:
        gen = torch.Generator(device="cpu").manual_seed(seed)
        out = pipe(
            prompt=v["prompt"], negative_prompt=v["neg"],
            image=crop_512, mask_image=mask_512,
            strength=strength, guidance_scale=guidance,
            num_inference_steps=steps, generator=gen,
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

print("RESULT_JSON:" + json.dumps({"device": device, "variants": outputs, "mapping": mapping}))
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


def heuristic_flags(atk_region: np.ndarray, res_region: np.ndarray) -> dict:
    atk = atk_region.astype(np.float32)
    res = res_region.astype(np.float32)
    mae = float(np.mean(np.abs(atk - res)))
    res_chroma = float(np.mean(np.max(res, axis=2) - np.min(res, axis=2)))
    sticker_remain = mae < 25 and res_chroma > 40
    sticker_removed = "NO" if sticker_remain else "YES"
    std = float(np.std(res))
    gray = np.mean(res, axis=2).astype(np.float32)
    gy, gx = np.gradient(gray)
    edge = float(np.mean(np.sqrt(gx * gx + gy * gy)))
    # greenish landscape cue
    mean = res.reshape(-1, 3).mean(axis=0)
    greenish = mean[1] > mean[0] + 8 and mean[1] > mean[2] + 5
    landscape = "YES" if mae >= 25 and greenish and edge > 10 else "NO"
    object_hall = "YES" if mae >= 30 and edge > 12 and std > 28 and landscape == "NO" else "NO"
    abs_gx = float(np.mean(np.abs(gx)))
    abs_gy = float(np.mean(np.abs(gy)))
    aniso = abs_gx / (abs_gy + 1e-6)
    text = "YES" if mae >= 20 and ((aniso > 1.55) or (aniso < 0.65)) and edge > 10 else "NO"
    color_shift = float(np.linalg.norm(atk.mean(axis=(0, 1)) - res.mean(axis=(0, 1))))
    strong_color = "YES" if color_shift > 45 else "NO"
    if res.shape[0] > 8 and res.shape[1] > 8:
        border = np.concatenate(
            [res[0:2].reshape(-1, 3), res[-2:].reshape(-1, 3), res[:, 0:2].reshape(-1, 3), res[:, -2:].reshape(-1, 3)],
            axis=0,
        )
        interior = res[2:-2, 2:-2].reshape(-1, 3)
        seam = "YES" if float(np.linalg.norm(border.mean(0) - interior.mean(0))) > 18 else "NO"
    else:
        seam = "UNKNOWN"
    new_sem = "YES" if landscape == "YES" or object_hall == "YES" else "NO"
    natural = "YES" if sticker_removed == "YES" and landscape == "NO" and object_hall == "NO" and text == "NO" else "NO"
    return {
        "sticker_removed": sticker_removed,
        "landscape_hallucination": landscape,
        "object_hallucination": object_hall,
        "text_artifact": text,
        "strong_color_change": strong_color,
        "visible_seam": seam,
        "natural_context_match": natural,
        "new_semantic_content": new_sem,
        "mae": mae,
        "color_shift": color_shift,
        "edge": edge,
        "std": std,
    }


def main():
    for d in [RESULTS] + [OUT_ROOT / p["folder"] for p in PROMPTS]:
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
    if (x1, y1, x2, y2) != EXACT_PATCH or patch_size != recorded:
        raise RuntimeError(f"Patch verify failed: got {(x1,y1,x2,y2,patch_size)} expected {EXACT_PATCH}+{recorded}")
    ex1, ey1, ex2, ey2 = expand_mask(x1, y1, x2, y2, iw, ih, MASK_PADDING)
    if (ex1, ey1, ex2, ey2) != EXPANDED:
        raise RuntimeError(f"Expanded mask verify failed: got {(ex1,ey1,ex2,ey2)} expected {EXPANDED}")
    left, top, right, bottom = compute_context_crop(iw, ih, ex1, ey1, ex2, ey2, CONTEXT)
    if (left, top, right, bottom) != CROP:
        raise RuntimeError(f"Crop verify failed: got {(left,top,right,bottom)} expected {CROP}")
    if attack_row["target_class"] != "car":
        raise RuntimeError(f"Unexpected target_class {attack_row['target_class']}")

    attacked_path = ATTACK_02 / "patched_images" / FILENAME
    attacked = Image.open(attacked_path).convert("RGB")
    if attacked.size != (iw, ih):
        raise RuntimeError(f"size mismatch {attacked.size}")

    meta = {
        "filename": FILENAME,
        "target_class": "car",
        "exact_patch_coordinates": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "expanded_mask_coordinates": {"x1": ex1, "y1": ey1, "x2": ex2, "y2": ey2},
        "patch_size_pixels": patch_size,
        "locked_from_previous_variant_B": {
            "strength": STRENGTH,
            "steps": STEPS,
            "guidance_scale": GUIDANCE,
            "context_crop_factor": CONTEXT,
            "mask_padding": MASK_PADDING,
            "seed": SEED,
            "crop": {"left": left, "top": top, "right": right, "bottom": bottom},
            "feather": FEATHER,
            "resolution": RESOLUTION,
        },
        "only_prompt_changed": True,
        "clean_pixels_used": False,
        "prompts": {p["key"]: {"prompt": p["prompt"], "neg": p["neg"]} for p in PROMPTS},
    }
    (OUT_ROOT / "test_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Running prompt-only SD2 inpainting P1/P2/P3 ...")
    payload = {
        "model": str(MODEL),
        "attacked": str(attacked_path),
        "variants": [
            {"key": p["key"], "out_dir": str(OUT_ROOT / p["folder"]), "prompt": p["prompt"], "neg": p["neg"]}
            for p in PROMPTS
        ],
        "strength": STRENGTH,
        "seed": SEED,
        "steps": STEPS,
        "guidance": GUIDANCE,
        "resolution": RESOLUTION,
        "feather": FEATHER,
        "ex1": ex1,
        "ey1": ey1,
        "ex2": ex2,
        "ey2": ey2,
        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom,
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
        raise RuntimeError(f"Inpaint failed: {(proc.stderr or proc.stdout)[-2000:]}")

    sd_json = None
    for line in (proc.stdout or "").splitlines():
        if line.startswith("RESULT_JSON:"):
            sd_json = json.loads(line[len("RESULT_JSON:") :])
    if sd_json is None:
        raise RuntimeError("No RESULT_JSON")
    sd_by_key = {v["key"]: v for v in sd_json["variants"]}

    pad = max(40, patch_size)
    zx1, zy1 = max(0, x1 - pad), max(0, y1 - pad)
    zx2, zy2 = min(iw, x2 + pad), min(ih, y2 + pad)
    atk_np = np.array(attacked)

    rows = []
    for p in PROMPTS:
        out_dir = OUT_ROOT / p["folder"]
        sd = sd_by_key.get(p["key"], {"status": "FAILED"})
        row = {
            "variant": p["key"],
            "folder": p["folder"],
            "label": p["label"],
            "sd_status": sd.get("status", "FAILED"),
            "detected": "FAILED",
            "confidence": 0.0,
            "iou": 0.0,
            "sticker_removed": "FAILED",
            "landscape_hallucination": "FAILED",
            "object_hallucination": "FAILED",
            "text_artifact": "FAILED",
            "strong_color_change": "FAILED",
            "visible_seam": "FAILED",
            "natural_context_match": "FAILED",
            "new_semantic_content": "FAILED",
            "prompt": p["prompt"],
            "neg": p["neg"],
        }
        if sd.get("status") != "OK" or not (out_dir / "restored_full.jpg").is_file():
            rows.append(row)
            (out_dir / "result.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
            continue
        restored = Image.open(out_dir / "restored_full.jpg").convert("RGB")
        restored.crop((zx1, zy1, zx2, zy2)).save(out_dir / "patch_zoom.jpg", quality=95)
        flags = heuristic_flags(atk_np[ey1:ey2, ex1:ex2], np.array(restored)[ey1:ey2, ex1:ex2])
        row.update(flags)
        row["mapping"] = sd.get("mapping")
        rows.append(row)

    print("Running Hyper-YOLO...")
    sys.path.insert(0, str(EXPERIMENT / "scripts"))
    from pipeline_core import evaluate_target_match, load_yolo, parse_label_file, run_yolo_inference

    yolo_src = RESULTS / "_yolo_batch_inputs"
    if yolo_src.exists():
        shutil.rmtree(yolo_src)
    yolo_src.mkdir(parents=True)
    for row in rows:
        if row["sd_status"] != "OK":
            continue
        shutil.copy2(OUT_ROOT / row["folder"] / "restored_full.jpg", yolo_src / f"{row['variant']}_{FILENAME}")

    model = load_yolo()
    pred_images = RESULTS / "_yolo_images"
    pred_labels = RESULTS / "_yolo_labels"
    run_yolo_inference(model, yolo_src, pred_images, pred_labels, RESULTS / "_yolo_parent")

    for row in rows:
        if row["sd_status"] != "OK":
            continue
        stem = f"{row['variant']}_{STEM}"
        dets = parse_label_file(pred_labels / f"{stem}.txt", iw, ih, model)
        match = evaluate_target_match(dets, "car", bbox)
        row["detected"] = match["detected"]
        row["confidence"] = float(match["confidence"])
        row["iou"] = float(match["iou"])
        yolo_viz = pred_images / f"{row['variant']}_{FILENAME}"
        if yolo_viz.is_file():
            shutil.copy2(yolo_viz, OUT_ROOT / row["folder"] / "hyper_yolo_prediction.jpg")
        (OUT_ROOT / row["folder"] / "result.json").write_text(json.dumps(row, indent=2), encoding="utf-8")

    # Comparisons vs previous B (read-only)
    prev_b_img = Image.open(PREV_B / "restored_full.jpg").convert("RGB")
    panels = [panel(prev_b_img.crop((zx1, zy1, zx2, zy2)), "PREVIOUS B")]
    zoom_panels = [
        panel(prev_b_img.crop((ex1, ey1, ex2, ey2)).resize((256, 256), Image.Resampling.NEAREST), "PREVIOUS B ZOOM")
    ]
    for row in rows:
        out_dir = OUT_ROOT / row["folder"]
        if row["sd_status"] == "OK":
            rest = Image.open(out_dir / "restored_full.jpg").convert("RGB")
            panels.append(panel(rest.crop((zx1, zy1, zx2, zy2)), f"{row['variant']} det={row['detected']}"))
            zoom_panels.append(
                panel(rest.crop((ex1, ey1, ex2, ey2)).resize((256, 256), Image.Resampling.NEAREST), f"{row['variant']} ZOOM")
            )
        else:
            panels.append(panel(Image.new("RGB", (200, 200), (80, 0, 0)), f"{row['variant']} FAILED"))

    vstack(panels).save(RESULTS / "prompt_comparison.jpg", quality=95)
    zw = sum(p.size[0] for p in zoom_panels) + 4 * (len(zoom_panels) - 1)
    zh = max(p.size[1] for p in zoom_panels)
    zc = Image.new("RGB", (zw, zh), (255, 255, 255))
    x = 0
    for p in zoom_panels:
        zc.paste(p, (x, 0))
        x += p.size[0] + 4
    zc.save(RESULTS / "zoom_comparison.jpg", quality=95)

    # CSV + summary (heuristics; visual review filled after image inspect in companion step)
    csv_path = RESULTS / "results.csv"
    fields = [
        "variant",
        "detected",
        "confidence",
        "iou",
        "sticker_removed",
        "landscape_hallucination",
        "object_hallucination",
        "text_artifact",
        "strong_color_change",
        "visible_seam",
        "natural_context_match",
        "new_semantic_content",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})

    lines = [
        "PROMPT-ONLY SD2 INPAINTING TEST",
        "=" * 72,
        f"image: {FILENAME} target=car patch={EXACT_PATCH} expanded={EXPANDED}",
        f"LOCKED: strength={STRENGTH} steps={STEPS} guidance={GUIDANCE} context={CONTEXT} seed={SEED}",
        "ONLY prompt changed. Compared to previous Variant B (read-only).",
        "",
    ]
    for row in rows:
        lines.append(
            f"{row['variant']}: det={row['detected']} conf={row['confidence']:.4f} iou={row['iou']:.4f} "
            f"sticker={row['sticker_removed']} land={row['landscape_hallucination']} "
            f"obj={row['object_hallucination']} natural={row['natural_context_match']} new_sem={row['new_semantic_content']}"
        )
    lines.append("")
    lines.append(f"output_root: {OUT_ROOT}")
    (RESULTS / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"OUT_ROOT={OUT_ROOT}")


if __name__ == "__main__":
    main()
