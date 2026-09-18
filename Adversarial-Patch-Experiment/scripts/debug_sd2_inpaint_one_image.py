#!/usr/bin/env python3
"""ONE-IMAGE SD2 dedicated inpainting debug (000000532855 only).

Does not modify attacks, mitigation_01, Hyper-YOLO, or Stable-Diffusion-Patch trees.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image, ImageDraw

EXPERIMENT = Path(__file__).resolve().parents[1]
ATTACK_02 = EXPERIMENT / "attacks" / "attack_02"
MODEL = EXPERIMENT / "mitigation_models" / "stable-diffusion-2-inpainting"
M01_RESTORED = EXPERIMENT / "mitigations" / "mitigation_01" / "restored_images"
OUT = EXPERIMENT / "mitigation_debug" / "sd2_inpainting_one_image" / "000000532855"

FILENAME = "000000532855.jpg"
STRENGTHS = {"075": 0.75, "090": 0.90, "100": 1.00}
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


def load_rows():
    attack = {
        r["filename"]: r
        for r in csv.DictReader((ATTACK_02 / "results" / "attack_results.csv").open(encoding="utf-8"))
    }
    selected = {
        r["filename"]: r
        for r in csv.DictReader((ATTACK_02 / "results" / "selected_images.csv").open(encoding="utf-8"))
    }
    return attack[FILENAME], selected[FILENAME]


def panel(img: Image.Image, label: str, size=(300, 300)) -> Image.Image:
    canvas = Image.new("RGB", size, (255, 255, 255))
    thumb = img.copy()
    thumb.thumbnail((size[0], size[1] - 28), Image.Resampling.LANCZOS)
    canvas.paste(thumb, ((size[0] - thumb.size[0]) // 2, 28 + (size[1] - 28 - thumb.size[1]) // 2))
    d = ImageDraw.Draw(canvas)
    d.rectangle([0, 0, size[0], 26], fill=(0, 0, 0))
    d.text((6, 5), label, fill=(255, 255, 255))
    return canvas


def hstack(images):
    h = max(im.size[1] for im in images)
    canvas = Image.new("RGB", (sum(im.size[0] for im in images), h), (255, 255, 255))
    x = 0
    for im in images:
        canvas.paste(im, (x, 0))
        x += im.size[0]
    return canvas


def visual_flags(attacked_patch: np.ndarray, restored_patch: np.ndarray) -> dict:
    mae = float(np.mean(np.abs(attacked_patch.astype(np.float32) - restored_patch.astype(np.float32))))
    # sticker color remnant heuristic: high chroma in attacked, still high chroma in restored
    atk = attacked_patch.astype(np.float32)
    res = restored_patch.astype(np.float32)
    atk_chroma = float(np.mean(np.max(atk, axis=2) - np.min(atk, axis=2)))
    res_chroma = float(np.mean(np.max(res, axis=2) - np.min(res, axis=2)))
    sticker_remain = "YES" if (mae < 25 and res_chroma > 40) else ("LIKELY_PARTIAL" if mae < 40 and res_chroma > 35 else "NO")
    # flat/gray: low std
    std = float(np.std(res))
    flat_gray = "YES" if std < 22 else "NO"
    # hallucination heuristic: higher structure than attacked patch after change, high edge density
    gray = np.mean(res, axis=2).astype(np.uint8)
    # simple gradient magnitude
    gy, gx = np.gradient(gray.astype(np.float32))
    edge = float(np.mean(np.sqrt(gx * gx + gy * gy)))
    halluc = "YES" if mae >= 25 and edge > 12 and std > 28 else "NO"
    # seam: border vs interior difference
    if res.shape[0] > 8 and res.shape[1] > 8:
        border = np.concatenate(
            [
                res[0:2, :, :].reshape(-1, 3),
                res[-2:, :, :].reshape(-1, 3),
                res[:, 0:2, :].reshape(-1, 3),
                res[:, -2:, :].reshape(-1, 3),
            ],
            axis=0,
        )
        interior = res[2:-2, 2:-2, :].reshape(-1, 3)
        seam = "YES" if float(np.linalg.norm(border.mean(0) - interior.mean(0))) > 18 else "NO"
    else:
        seam = "UNKNOWN"
    return {
        "mae_attacked_vs_restored_patch": mae,
        "sticker_pattern_remain": sticker_remain,
        "flat_gray_restoration": flat_gray,
        "obvious_hallucination": halluc,
        "boundary_seam_visible": seam,
        "restored_patch_std": std,
        "edge_magnitude": edge,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    attack_row, selected = load_rows()
    recorded_size = int(float(attack_row["patch_size_pixels"]))
    bbox_x = float(selected["bbox_x"])
    bbox_y = float(selected["bbox_y"])
    bbox_w = float(selected["bbox_width"])
    bbox_h = float(selected["bbox_height"])
    image_w = int(selected["image_width"])
    image_h = int(selected["image_height"])
    x1, y1, x2, y2, patch_size = reconstruct_patch(bbox_x, bbox_y, bbox_w, bbox_h, image_w, image_h)
    if patch_size != recorded_size:
        raise RuntimeError(f"Patch size mismatch: reconstructed={patch_size} recorded={recorded_size}")

    attacked_path = ATTACK_02 / "patched_images" / FILENAME
    attacked = Image.open(attacked_path).convert("RGB")
    if attacked.size != (image_w, image_h):
        raise RuntimeError(f"Image size mismatch: {attacked.size} vs {(image_w, image_h)}")

    # Exact full-res mask: white=patch, black=context
    mask_full = Image.new("L", (image_w, image_h), 0)
    mask_arr = np.array(mask_full)
    mask_arr[y1:y2, x1:x2] = 255
    mask_full = Image.fromarray(mask_arr)

    left, top, right, bottom = compute_context_crop(image_w, image_h, x1, y1, x2, y2)
    cw, ch = right - left, bottom - top
    rx1, ry1, rx2, ry2 = x1 - left, y1 - top, x2 - left, y2 - top

    crop = attacked.crop((left, top, right, bottom))
    mask_crop = mask_full.crop((left, top, right, bottom))
    crop_512 = crop.resize((RESOLUTION, RESOLUTION), Image.Resampling.LANCZOS)
    mask_512 = mask_crop.resize((RESOLUTION, RESOLUTION), Image.Resampling.NEAREST)

    attacked.save(OUT / "00_attacked.jpg", quality=95)
    mask_full.save(OUT / "01_exact_mask.png")
    crop.save(OUT / "02_context_crop.jpg", quality=95)
    mask_512.save(OUT / "03_mask_512.png")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading StableDiffusionInpaintPipeline on {device} with memory-safe settings...")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        str(MODEL),
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        variant="fp16",
        use_safetensors=True,
        local_files_only=True,
    )
    # RTX 2050 4GB: prefer CPU offload + attention slicing
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    if device == "cuda" and hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)

    attacked_np = np.array(attacked).astype(np.float32)
    attacked_patch = attacked_np[y1:y2, x1:x2]
    results = {
        "filename": FILENAME,
        "target_class": attack_row["target_class"],
        "patch": {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "patch_size_pixels": patch_size},
        "crop": {"left": left, "top": top, "right": right, "bottom": bottom},
        "device": device,
        "seed": SEED,
        "steps": STEPS,
        "guidance_scale": GUIDANCE,
        "clean_pixels_used": False,
        "neutralization_used": False,
        "strengths": {},
    }

    zoom_pad = max(40, patch_size)
    zx1, zy1 = max(0, x1 - zoom_pad), max(0, y1 - zoom_pad)
    zx2, zy2 = min(image_w, x2 + zoom_pad), min(image_h, y2 + zoom_pad)

    restored_by_tag = {}
    for tag, strength in STRENGTHS.items():
        print(f"Inpainting strength={strength} ...")
        sub = OUT / f"strength_{tag}"
        sub.mkdir(parents=True, exist_ok=True)
        try:
            gen = torch.Generator(device="cuda" if device == "cuda" else "cpu").manual_seed(SEED)
            # With CPU offload, generator device may need to be cpu - try cuda first then cpu
            try:
                out_img = pipe(
                    prompt=PROMPT,
                    negative_prompt=NEG,
                    image=crop_512,
                    mask_image=mask_512,
                    strength=strength,
                    guidance_scale=GUIDANCE,
                    num_inference_steps=STEPS,
                    generator=gen,
                ).images[0]
            except Exception:
                gen = torch.Generator(device="cpu").manual_seed(SEED)
                out_img = pipe(
                    prompt=PROMPT,
                    negative_prompt=NEG,
                    image=crop_512,
                    mask_image=mask_512,
                    strength=strength,
                    guidance_scale=GUIDANCE,
                    num_inference_steps=STEPS,
                    generator=gen,
                ).images[0]

            out_img.save(sub / "raw_inpaint_512.png")
            resized = out_img.resize((cw, ch), Image.Resampling.LANCZOS)
            patch_gen = np.array(resized).astype(np.float32)[ry1:ry2, rx1:rx2]
            if patch_gen.shape[0] != (y2 - y1) or patch_gen.shape[1] != (x2 - x1):
                raise RuntimeError(f"Patch shape mismatch: {patch_gen.shape}")

            base = attacked_np.copy()
            alpha = feathered_alpha(max(y2 - y1, x2 - x1), FEATHER)[: y2 - y1, : x2 - x1][:, :, None]
            base[y1:y2, x1:x2] = alpha * patch_gen + (1.0 - alpha) * base[y1:y2, x1:x2]
            composite = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
            composite.save(sub / "restored_full.jpg", quality=95)
            restored_by_tag[tag] = composite

            flags = visual_flags(attacked_patch, np.array(composite)[y1:y2, x1:x2])
            results["strengths"][tag] = {
                "strength": strength,
                "status": "OK",
                "raw_inpaint_512": str(sub / "raw_inpaint_512.png"),
                "restored_full": str(sub / "restored_full.jpg"),
                **flags,
            }
            print(json.dumps(results["strengths"][tag], indent=2))
        except Exception as exc:
            results["strengths"][tag] = {
                "strength": strength,
                "status": "FAILED",
                "error": str(exc),
            }
            print(f"FAILED strength={strength}: {exc}")
            (sub / "FAILED.txt").write_text(str(exc), encoding="utf-8")

    # 04 strength comparison
    panels = [panel(attacked.crop((zx1, zy1, zx2, zy2)), "ATTACKED")]
    for tag, strength in STRENGTHS.items():
        if tag in restored_by_tag:
            panels.append(
                panel(restored_by_tag[tag].crop((zx1, zy1, zx2, zy2)), f"SD2 INPAINT {strength:.2f}")
            )
        else:
            panels.append(panel(Image.new("RGB", (200, 200), (80, 0, 0)), f"FAILED {strength:.2f}"))
    hstack(panels).save(OUT / "04_strength_comparison.jpg", quality=95)

    # 05 old vs new
    old = Image.open(M01_RESTORED / FILENAME).convert("RGB") if (M01_RESTORED / FILENAME).is_file() else None
    panels2 = [panel(attacked.crop((zx1, zy1, zx2, zy2)), "ATTACKED")]
    if old is not None:
        panels2.append(panel(old.crop((zx1, zy1, zx2, zy2)), "OLD SD2.1 IMG2IMG"))
    for tag, strength in STRENGTHS.items():
        if tag in restored_by_tag:
            panels2.append(
                panel(restored_by_tag[tag].crop((zx1, zy1, zx2, zy2)), f"NEW INPAINT {strength:.2f}")
            )
    hstack(panels2).save(OUT / "05_old_vs_new_comparison.jpg", quality=95)

    (OUT / "inpaint_debug_report.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"OUT={OUT}")


if __name__ == "__main__":
    main()
