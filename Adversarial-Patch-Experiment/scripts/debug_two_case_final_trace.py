#!/usr/bin/env python3
"""TWO-CASE restoration trace (debug only). Does not modify mitigation_01 or attacks."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image, ImageDraw, ImageFilter

EXPERIMENT = Path(__file__).resolve().parents[1]
M01 = EXPERIMENT / "mitigations" / "mitigation_01"
OUT = EXPERIMENT / "mitigation_debug" / "two_case_final_trace"
SD_MODEL = EXPERIMENT.parent / "Stable-Diffusion-Patch" / "model"

CASES = {
    "caseA_gray_flat": "000000009590.jpg",
    "caseB_hallucination": "000000532855.jpg",
}

STRENGTH = 0.55
STEPS = 30
GUIDANCE = 7.5
SEED = 20260727
RESOLUTION = 512
CONTEXT = 3.0
MIN_SIDE = 128
FEATHER = 4
PROMPT = (
    "realistic natural photograph, restore the original object surface, "
    "natural texture, consistent lighting, no sticker or artificial patch"
)
NEG = (
    "sticker, adversarial patch, logo, text, watermark, abstract pattern, "
    "cartoon, distortion"
)


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


def neutralize(crop, rx1, ry1, rx2, ry2, feather=4):
    arr = np.array(crop).astype(np.float32)
    blurred = np.array(crop.filter(ImageFilter.GaussianBlur(radius=18))).astype(np.float32)
    pad = max(4, (rx2 - rx1) // 6)
    h, w = arr.shape[:2]
    y0, y1b = max(0, ry1 - pad), min(h, ry2 + pad)
    x0, x1b = max(0, rx1 - pad), min(w, rx2 + pad)
    ring = arr[y0:y1b, x0:x1b].copy()
    local = np.ones(ring.shape[:2], dtype=bool)
    local[ry1 - y0 : ry2 - y0, rx1 - x0 : rx2 - x0] = False
    mean = ring[local].mean(axis=0) if local.any() else arr.reshape(-1, 3).mean(axis=0)
    filled = blurred.copy()
    filled[ry1:ry2, rx1:rx2] = 0.65 * blurred[ry1:ry2, rx1:rx2] + 0.35 * mean
    ph, pw = ry2 - ry1, rx2 - rx1
    alpha = feathered_alpha(max(ph, pw), feather)[:ph, :pw][:, :, None]
    out = arr.copy()
    out[ry1:ry2, rx1:rx2] = alpha * filled[ry1:ry2, rx1:rx2] + (1 - alpha) * out[ry1:ry2, rx1:rx2]
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def load_meta(filename):
    with (M01 / "source_attack_data" / "selected_images_metadata.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["filename"] == filename:
                return row
    raise KeyError(filename)


def panel(img, label, size=(320, 320)):
    canvas = Image.new("RGB", size, (255, 255, 255))
    thumb = img.copy()
    thumb.thumbnail((size[0], size[1] - 28), Image.Resampling.LANCZOS)
    canvas.paste(thumb, ((size[0] - thumb.size[0]) // 2, 28 + (size[1] - 28 - thumb.size[1]) // 2))
    d = ImageDraw.Draw(canvas)
    d.rectangle([0, 0, size[0], 26], fill=(0, 0, 0))
    d.text((6, 5), label, fill=(255, 255, 255))
    return canvas


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    dtype = torch.float16 if use_cuda else torch.float32
    print(f"Loading SD on {device}...")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        str(SD_MODEL), local_files_only=True, torch_dtype=dtype, safety_checker=None
    ).to(device)
    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None

    report = {
        "device": device,
        "compositing_variable": "patch_restored = restored_crop_np[rel_y1:rel_y2, rel_x1:rel_x2] "
        "where restored_crop comes from restored_512=result.images[0] (SD output), "
        "NOT from neutralized init_512",
        "cases": {},
    }

    for case_name, filename in CASES.items():
        print(f"Tracing {case_name}: {filename}")
        case_dir = OUT / case_name
        case_dir.mkdir(parents=True, exist_ok=True)
        meta = load_meta(filename)
        x1, y1, x2, y2 = map(int, [meta["patch_x1"], meta["patch_y1"], meta["patch_x2"], meta["patch_y2"]])

        attacked_src = M01 / "source_attacked_images" / filename
        final_src = M01 / "restored_images" / filename
        neut_saved = M01 / "neutralized_inputs" / f"{Path(filename).stem}_512.png"

        attacked = Image.open(attacked_src).convert("RGB")
        w, h = attacked.size
        left, top, right, bottom = compute_context_crop(w, h, x1, y1, x2, y2)
        cw, ch = right - left, bottom - top
        rx1, ry1, rx2, ry2 = x1 - left, y1 - top, x2 - left, y2 - top

        crop = attacked.crop((left, top, right, bottom))
        neut_crop = neutralize(crop, rx1, ry1, rx2, ry2, FEATHER)
        init_512 = neut_crop.resize((RESOLUTION, RESOLUTION), Image.Resampling.LANCZOS)

        # Save stage images
        attacked.save(case_dir / "01_attacked.jpg", quality=95)
        init_512.save(case_dir / "02_neutralized_input.jpg", quality=95)
        # also keep exact saved mitigation neutralized for reference
        if neut_saved.is_file():
            shutil.copy2(neut_saved, case_dir / "02b_mitigation01_saved_neutralized_512.png")

        gen = torch.Generator(device=device).manual_seed(SEED)
        sd_out = pipe(
            prompt=PROMPT,
            negative_prompt=NEG,
            image=init_512,
            strength=STRENGTH,
            guidance_scale=GUIDANCE,
            num_inference_steps=STEPS,
            generator=gen,
        ).images[0]
        sd_out.save(case_dir / "03_raw_sd_output_512.png")

        resized = sd_out.resize((cw, ch), Image.Resampling.LANCZOS)
        resized.save(case_dir / "04_sd_output_resized.png")

        sd_patch = resized.crop((rx1, ry1, rx2, ry2))
        sd_patch.save(case_dir / "05_extracted_sd_patch.png")

        # Exact compositing path from restore_patch_region.py
        base = np.array(attacked).astype(np.float32)
        restored_crop_np = np.array(resized).astype(np.float32)
        patch_restored = restored_crop_np[ry1:ry2, rx1:rx2]  # FROM SD
        alpha = feathered_alpha(max(y2 - y1, x2 - x1), FEATHER)[: y2 - y1, : x2 - x1][:, :, None]
        region = base[y1:y2, x1:x2]
        base[y1:y2, x1:x2] = alpha * patch_restored + (1.0 - alpha) * region
        composite = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
        composite.save(case_dir / "06_final_restored.jpg", quality=95)

        # Original mitigation_01 final for side-by-side
        shutil.copy2(final_src, case_dir / "06b_mitigation01_saved_restored.jpg")

        # Metrics on ORIGINAL mitigation_01 restored vs intermediates from this replay
        # and vs this replay composite
        orig_rest = np.array(Image.open(final_src).convert("RGB")).astype(np.float32)
        neut_full_resized = np.array(
            Image.open(neut_saved).convert("RGB").resize((cw, ch), Image.Resampling.LANCZOS)
            if neut_saved.is_file()
            else init_512.resize((cw, ch), Image.Resampling.LANCZOS)
        ).astype(np.float32)
        neut_patch = neut_full_resized[ry1:ry2, rx1:rx2]
        sd_patch_np = np.array(sd_patch).astype(np.float32)
        orig_patch = orig_rest[y1:y2, x1:x2]
        replay_patch = np.array(composite)[y1:y2, x1:x2].astype(np.float32)

        mae_neut_vs_orig = float(np.mean(np.abs(neut_patch - orig_patch)))
        mae_sd_vs_orig = float(np.mean(np.abs(sd_patch_np - orig_patch)))
        mae_neut_vs_replay = float(np.mean(np.abs(neut_patch - replay_patch)))
        mae_sd_vs_replay = float(np.mean(np.abs(sd_patch_np - replay_patch)))
        mae_orig_vs_replay = float(np.mean(np.abs(orig_patch - replay_patch)))

        # Outside-patch unchanged check on original
        mask = np.ones(orig_rest.shape[:2], dtype=bool)
        mask[y1:y2, x1:x2] = False
        atk_np = np.array(attacked).astype(np.float32)
        mae_outside = float(np.mean(np.abs(atk_np[mask] - orig_rest[mask])))

        # Comparison grid
        pad = max(40, x2 - x1)
        zx1, zy1 = max(0, x1 - pad), max(0, y1 - pad)
        zx2, zy2 = min(w, x2 + pad), min(h, y2 + pad)
        panels = [
            panel(attacked.crop((zx1, zy1, zx2, zy2)), "ATTACKED ZOOM"),
            panel(Image.fromarray(neut_patch.astype(np.uint8)), "NEUT PATCH"),
            panel(sd_out, "RAW SD 512"),
            panel(sd_patch, "SD PATCH"),
            panel(Image.fromarray(orig_patch.astype(np.uint8)), "M01 FINAL PATCH"),
            panel(Image.open(final_src).convert("RGB").crop((zx1, zy1, zx2, zy2)), "M01 FINAL ZOOM"),
            panel(composite.crop((zx1, zy1, zx2, zy2)), "REPLAY COMPOSITE"),
        ]
        # hstack
        grid = Image.new("RGB", (sum(p.size[0] for p in panels), panels[0].size[1]), (255, 255, 255))
        x = 0
        for p in panels:
            grid.paste(p, (x, 0))
            x += p.size[0]
        grid.save(case_dir / "07_comparison.jpg", quality=95)

        # Verdict helpers
        closer_to_sd = mae_sd_vs_orig < mae_neut_vs_orig
        case_report = {
            "filename": filename,
            "patch": {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "size": x2 - x1},
            "crop": {"left": left, "top": top, "right": right, "bottom": bottom},
            "A_mae_neutralized_patch_vs_final_restored_patch": mae_neut_vs_orig,
            "B_mae_sd_generated_patch_vs_final_restored_patch": mae_sd_vs_orig,
            "mae_neutralized_vs_replay_composite": mae_neut_vs_replay,
            "mae_sd_vs_replay_composite": mae_sd_vs_replay,
            "mae_original_final_vs_replay_composite": mae_orig_vs_replay,
            "mae_outside_patch_attacked_vs_m01_restored": mae_outside,
            "final_patch_closer_to": "SD_output" if closer_to_sd else "neutralized_input",
            "compositing_source_variable": "patch_restored from restored_crop_np (SD resized), not neutralized",
            "orig_patch_std": float(np.std(orig_patch)),
            "neut_patch_std": float(np.std(neut_patch)),
            "sd_patch_std": float(np.std(sd_patch_np)),
        }
        report["cases"][case_name] = case_report
        (case_dir / "trace_metrics.json").write_text(json.dumps(case_report, indent=2), encoding="utf-8")
        print(json.dumps(case_report, indent=2))

    (OUT / "two_case_trace_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"OUT={OUT}")


if __name__ == "__main__":
    main()
