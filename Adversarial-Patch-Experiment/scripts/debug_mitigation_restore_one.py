#!/usr/bin/env python3
"""ONE-IMAGE mitigation restoration debug (mitigation_debug only).

Does NOT write into attacks/, Hyper-YOLO/, or Stable-Diffusion-Patch/.
Does NOT overwrite mitigations/mitigation_01/ partial outputs.
Uses PIL only (runs in Stable-Diffusion-Patch venv).
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image, ImageDraw, ImageFilter

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEBUG_ROOT = EXPERIMENT_ROOT / "mitigation_debug" / "one_image_restore"
MITIGATION_01 = EXPERIMENT_ROOT / "mitigations" / "mitigation_01"
ATTACK_02 = EXPERIMENT_ROOT / "attacks" / "attack_02"
SD_MODEL_PATH = EXPERIMENT_ROOT.parent / "Stable-Diffusion-Patch" / "model"

PREFERRED_FILENAME = "000000407083.jpg"

RESOLUTION = 512
STEPS = 30
GUIDANCE = 7.5
SEED = 20260727
CONTEXT_FACTOR = 3.0
MIN_CONTEXT_SIDE = 128
FEATHER_PIXELS = 4
STRENGTHS = {"A_055": 0.55, "B_070": 0.70, "C_080": 0.80}

POSITIVE_PROMPT = (
    "realistic natural photograph, restore the original object surface, "
    "natural texture, consistent lighting, no sticker or artificial patch"
)
NEGATIVE_PROMPT = (
    "sticker, adversarial patch, logo, text, watermark, abstract pattern, "
    "cartoon, distortion"
)


def compute_context_crop(
    image_width: int,
    image_height: int,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> tuple[int, int, int, int]:
    patch_w = max(1, x2 - x1)
    patch_h = max(1, y2 - y1)
    patch_cx = (x1 + x2) / 2.0
    patch_cy = (y1 + y2) / 2.0
    side = max(MIN_CONTEXT_SIDE, int(round(CONTEXT_FACTOR * max(patch_w, patch_h))))
    side = min(side, image_width, image_height)
    left = int(round(patch_cx - side / 2.0))
    top = int(round(patch_cy - side / 2.0))
    left = max(0, min(left, image_width - side))
    top = max(0, min(top, image_height - side))
    return left, top, left + side, top + side


def feathered_alpha(size: int, feather_pixels: int) -> np.ndarray:
    alpha = np.ones((size, size), dtype=np.float32)
    feather = max(0, int(feather_pixels))
    if feather <= 0 or size < 2:
        return alpha
    for i in range(feather):
        weight = (i + 1) / (feather + 1)
        alpha[i, :] = np.minimum(alpha[i, :], weight)
        alpha[-(i + 1), :] = np.minimum(alpha[-(i + 1), :], weight)
        alpha[:, i] = np.minimum(alpha[:, i], weight)
        alpha[:, -(i + 1)] = np.minimum(alpha[:, -(i + 1)], weight)
    return alpha


def neutralize_patch_in_crop(
    crop_rgb: Image.Image,
    rel_x1: int,
    rel_y1: int,
    rel_x2: int,
    rel_y2: int,
    feather: int = 4,
) -> Image.Image:
    """Replace exact patch square using only surrounding attacked-crop pixels."""
    crop = np.array(crop_rgb).astype(np.float32)
    blurred = np.array(crop_rgb.filter(ImageFilter.GaussianBlur(radius=18))).astype(np.float32)

    pad = max(4, (rel_x2 - rel_x1) // 6)
    h, w = crop.shape[:2]
    y0 = max(0, rel_y1 - pad)
    y1b = min(h, rel_y2 + pad)
    x0 = max(0, rel_x1 - pad)
    x1b = min(w, rel_x2 + pad)
    ring = crop[y0:y1b, x0:x1b].copy()
    local_mask = np.ones(ring.shape[:2], dtype=bool)
    ly1 = rel_y1 - y0
    ly2 = rel_y2 - y0
    lx1 = rel_x1 - x0
    lx2 = rel_x2 - x0
    local_mask[ly1:ly2, lx1:lx2] = False
    if local_mask.any():
        mean_color = ring[local_mask].mean(axis=0)
    else:
        mean_color = crop.reshape(-1, 3).mean(axis=0)

    filled = blurred.copy()
    filled[rel_y1:rel_y2, rel_x1:rel_x2] = (
        0.65 * blurred[rel_y1:rel_y2, rel_x1:rel_x2] + 0.35 * mean_color
    )

    patch_h = rel_y2 - rel_y1
    patch_w = rel_x2 - rel_x1
    alpha = feathered_alpha(max(patch_h, patch_w), feather)[:patch_h, :patch_w][:, :, None]
    out = crop.copy()
    region = out[rel_y1:rel_y2, rel_x1:rel_x2]
    out[rel_y1:rel_y2, rel_x1:rel_x2] = (
        alpha * filled[rel_y1:rel_y2, rel_x1:rel_x2] + (1.0 - alpha) * region
    )
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def load_metadata(filename: str) -> dict:
    meta_csv = MITIGATION_01 / "source_attack_data" / "selected_images_metadata.csv"
    with meta_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["filename"] == filename:
                return row
    raise FileNotFoundError(f"{filename} not in {meta_csv}")


def choose_debug_image() -> str:
    meta_csv = MITIGATION_01 / "source_attack_data" / "selected_images_metadata.csv"
    names = []
    with meta_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            names.append(row["filename"])
    if PREFERRED_FILENAME in names:
        return PREFERRED_FILENAME
    return names[0]


def panel_label(img: Image.Image, text: str, size: tuple[int, int] = (360, 360)) -> Image.Image:
    canvas = Image.new("RGB", size, (255, 255, 255))
    fitted = img.copy()
    fitted.thumbnail((size[0], size[1] - 28), Image.Resampling.LANCZOS)
    ox = (size[0] - fitted.size[0]) // 2
    oy = 28 + (size[1] - 28 - fitted.size[1]) // 2
    canvas.paste(fitted, (ox, oy))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, size[0], 26], fill=(0, 0, 0))
    draw.text((6, 5), text, fill=(255, 255, 255))
    return canvas


def draw_boxes(img: Image.Image, boxes: list[tuple[tuple[int, int, int, int], tuple[int, int, int], int]]) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    for (x1, y1, x2, y2), color, width in boxes:
        for i in range(width):
            draw.rectangle([x1 - i, y1 - i, x2 + i, y2 + i], outline=color)
    return out


def hstack(images: list[Image.Image]) -> Image.Image:
    h = max(im.size[1] for im in images)
    w = sum(im.size[0] for im in images)
    canvas = Image.new("RGB", (w, h), (255, 255, 255))
    x = 0
    for im in images:
        canvas.paste(im, (x, 0))
        x += im.size[0]
    return canvas


def grid(images: list[Image.Image], cols: int = 4) -> Image.Image:
    if not images:
        return Image.new("RGB", (100, 100), (255, 255, 255))
    w, h = images[0].size
    rows = int(np.ceil(len(images) / cols))
    canvas = Image.new("RGB", (cols * w, rows * h), (255, 255, 255))
    for i, im in enumerate(images):
        r, c = divmod(i, cols)
        canvas.paste(im, (c * w, r * h))
    return canvas


def composite_patch(
    attacked: Image.Image,
    restored_crop: Image.Image,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    rel_x1: int,
    rel_y1: int,
    rel_x2: int,
    rel_y2: int,
) -> Image.Image:
    base = np.array(attacked).astype(np.float32)
    restored_crop_np = np.array(restored_crop).astype(np.float32)
    patch_restored = restored_crop_np[rel_y1:rel_y2, rel_x1:rel_x2]
    if patch_restored.shape[0] != (y2 - y1) or patch_restored.shape[1] != (x2 - x1):
        raise RuntimeError(
            f"Composite shape mismatch: {patch_restored.shape[:2]} vs {(y2 - y1, x2 - x1)}"
        )
    alpha = feathered_alpha(max(y2 - y1, x2 - x1), FEATHER_PIXELS)[: y2 - y1, : x2 - x1][:, :, None]
    region = base[y1:y2, x1:x2]
    base[y1:y2, x1:x2] = alpha * patch_restored + (1.0 - alpha) * region
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


def main() -> None:
    DEBUG_ROOT.mkdir(parents=True, exist_ok=True)
    filename = choose_debug_image()
    meta = load_metadata(filename)
    x1 = int(meta["patch_x1"])
    y1 = int(meta["patch_y1"])
    x2 = int(meta["patch_x2"])
    y2 = int(meta["patch_y2"])
    patch_size = int(meta["patch_size_pixels"])

    attacked_src = MITIGATION_01 / "source_attacked_images" / filename
    if not attacked_src.is_file():
        attacked_src = ATTACK_02 / "patched_images" / filename
    clean_src = MITIGATION_01 / "source_clean_images" / filename
    if not clean_src.is_file():
        clean_src = ATTACK_02 / "clean_images" / filename

    attacked_copy = DEBUG_ROOT / "source_attacked_copy.jpg"
    clean_copy = DEBUG_ROOT / "source_clean_copy_REFERENCE_ONLY.jpg"
    shutil.copy2(attacked_src, attacked_copy)
    if clean_src.is_file():
        shutil.copy2(clean_src, clean_copy)

    attacked = Image.open(attacked_copy).convert("RGB")
    image_width, image_height = attacked.size
    if (x2 - x1) != patch_size or (y2 - y1) != patch_size:
        raise RuntimeError("Patch size metadata inconsistent with coordinates")

    crop_box = compute_context_crop(image_width, image_height, x1, y1, x2, y2)
    crop_left, crop_top, crop_right, crop_bottom = crop_box
    crop_w = crop_right - crop_left
    crop_h = crop_bottom - crop_top
    crop = attacked.crop(crop_box)

    rel_x1 = x1 - crop_left
    rel_y1 = y1 - crop_top
    rel_x2 = x2 - crop_left
    rel_y2 = y2 - crop_top

    sx = RESOLUTION / float(crop_w)
    sy = RESOLUTION / float(crop_h)
    map512 = {
        "rel_x1_512": rel_x1 * sx,
        "rel_y1_512": rel_y1 * sy,
        "rel_x2_512": rel_x2 * sx,
        "rel_y2_512": rel_y2 * sy,
    }

    neutralized = neutralize_patch_in_crop(crop, rel_x1, rel_y1, rel_x2, rel_y2, FEATHER_PIXELS)
    crop.save(DEBUG_ROOT / "01_context_crop_original_attacked.png")
    neutralized.save(DEBUG_ROOT / "02_context_crop_neutralized.png")

    attacked_boxes = draw_boxes(
        attacked,
        [
            ((x1, y1, x2, y2), (255, 255, 0), 3),
            ((crop_left, crop_top, crop_right, crop_bottom), (255, 128, 0), 2),
        ],
    )
    attacked_boxes.save(DEBUG_ROOT / "00_attacked_with_boxes.jpg")
    crop_marked = draw_boxes(crop, [((rel_x1, rel_y1, rel_x2, rel_y2), (255, 255, 0), 2)])
    crop_marked.save(DEBUG_ROOT / "01b_context_crop_with_patch_box.png")

    init_512 = neutralized.resize((RESOLUTION, RESOLUTION), Image.Resampling.LANCZOS)
    init_512.save(DEBUG_ROOT / "03_diffusion_input_512_neutralized.png")
    old_512 = crop.resize((RESOLUTION, RESOLUTION), Image.Resampling.LANCZOS)
    old_512.save(DEBUG_ROOT / "03b_diffusion_input_512_ORIGINAL_NO_NEUTRALIZE.png")

    # Mark patch location on 512 inputs for mapping verification
    init_512_marked = draw_boxes(
        init_512,
        [
            (
                (
                    int(round(map512["rel_x1_512"])),
                    int(round(map512["rel_y1_512"])),
                    int(round(map512["rel_x2_512"])),
                    int(round(map512["rel_y2_512"])),
                ),
                (255, 0, 0),
                2,
            )
        ],
    )
    init_512_marked.save(DEBUG_ROOT / "03c_diffusion_input_512_with_mapped_patch_box.png")

    mapping = {
        "filename": filename,
        "subset_role": meta.get("subset_role"),
        "target_class": meta.get("target_class"),
        "image_size": {"width": image_width, "height": image_height},
        "patch": {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "patch_size_pixels": patch_size},
        "context_crop": {
            "left": crop_left,
            "top": crop_top,
            "right": crop_right,
            "bottom": crop_bottom,
            "width": crop_w,
            "height": crop_h,
        },
        "patch_in_crop": {
            "rel_x1": rel_x1,
            "rel_y1": rel_y1,
            "rel_x2": rel_x2,
            "rel_y2": rel_y2,
        },
        "patch_in_512": map512,
        "back_projection_check": {
            "x1_from_crop": crop_left + rel_x1,
            "y1_from_crop": crop_top + rel_y1,
            "x2_from_crop": crop_left + rel_x2,
            "y2_from_crop": crop_top + rel_y2,
            "matches_original_patch": (
                crop_left + rel_x1 == x1
                and crop_top + rel_y1 == y1
                and crop_left + rel_x2 == x2
                and crop_top + rel_y2 == y2
            ),
        },
        "composite_region": "exact patch square (x1,y1)-(x2,y2) with feather",
        "neutralization": "gaussian_blur_plus_local_ring_mean_from_attacked_crop_only",
        "clean_pixels_used_for_restoration": False,
        "seed": SEED,
        "steps": STEPS,
        "guidance_scale": GUIDANCE,
        "strengths": STRENGTHS,
    }
    (DEBUG_ROOT / "coordinate_mapping.json").write_text(
        json.dumps(mapping, indent=2), encoding="utf-8"
    )
    print("Coordinate mapping OK:", mapping["back_projection_check"]["matches_original_patch"])
    print(json.dumps(mapping["patch"], indent=2))
    print(json.dumps(mapping["context_crop"], indent=2))

    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    dtype = torch.float16 if use_cuda else torch.float32
    print(f"Loading SD Img2Img on {device}...")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        str(SD_MODEL_PATH),
        local_files_only=True,
        torch_dtype=dtype,
        safety_checker=None,
    )
    pipe = pipe.to(device)
    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None

    pad = max(40, patch_size)
    zx1 = max(0, x1 - pad)
    zy1 = max(0, y1 - pad)
    zx2 = min(image_width, x2 + pad)
    zy2 = min(image_height, y2 + pad)

    results_summary: dict = {"device": device, "filename": filename, "variants": {}}
    variant_composites: dict[str, Image.Image] = {}
    variant_raws: dict[str, Image.Image] = {}

    for tag, strength in STRENGTHS.items():
        print(f"Running neutralized Img2Img strength={strength} ({tag})...")
        generator = torch.Generator(device=device).manual_seed(SEED)
        out = pipe(
            prompt=POSITIVE_PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            image=init_512,
            strength=strength,
            guidance_scale=GUIDANCE,
            num_inference_steps=STEPS,
            generator=generator,
        ).images[0]
        raw_path = DEBUG_ROOT / f"04_raw_img2img_{tag}.png"
        out.save(raw_path)
        restored_crop = out.resize((crop_w, crop_h), Image.Resampling.LANCZOS)
        restored_crop.save(DEBUG_ROOT / f"05_resized_crop_{tag}.png")
        composite = composite_patch(
            attacked, restored_crop, x1, y1, x2, y2, rel_x1, rel_y1, rel_x2, rel_y2
        )
        composite_path = DEBUG_ROOT / f"06_final_composite_{tag}.jpg"
        composite.save(composite_path)
        attacked.crop((zx1, zy1, zx2, zy2)).save(DEBUG_ROOT / f"07_attacked_zoom_{tag}.png")
        composite.crop((zx1, zy1, zx2, zy2)).save(DEBUG_ROOT / f"08_composite_zoom_{tag}.png")
        variant_composites[tag] = composite
        variant_raws[tag] = out
        results_summary["variants"][tag] = {
            "strength": strength,
            "raw_img2img": str(raw_path),
            "final_composite": str(composite_path),
        }

    print("Running ORIGINAL (no neutralize) Img2Img strength=0.55 for comparison...")
    generator = torch.Generator(device=device).manual_seed(SEED)
    old_out = pipe(
        prompt=POSITIVE_PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        image=old_512,
        strength=0.55,
        guidance_scale=GUIDANCE,
        num_inference_steps=STEPS,
        generator=generator,
    ).images[0]
    old_out.save(DEBUG_ROOT / "04_raw_img2img_ORIGINAL_NO_NEUTRALIZE_055.png")
    old_crop = old_out.resize((crop_w, crop_h), Image.Resampling.LANCZOS)
    old_composite = composite_patch(
        attacked, old_crop, x1, y1, x2, y2, rel_x1, rel_y1, rel_x2, rel_y2
    )
    old_composite.save(DEBUG_ROOT / "06_final_composite_ORIGINAL_NO_NEUTRALIZE_055.jpg")
    old_composite.crop((zx1, zy1, zx2, zy2)).save(
        DEBUG_ROOT / "08_composite_zoom_ORIGINAL_NO_NEUTRALIZE_055.png"
    )

    panels = [
        panel_label(attacked, "ATTACKED"),
        panel_label(attacked_boxes, "PATCH+CONTEXT BOXES"),
        panel_label(crop_marked, "CONTEXT CROP + PATCH"),
        panel_label(old_512, "512 INPUT (NO NEUTRALIZE)"),
        panel_label(init_512, "512 INPUT (NEUTRALIZED)"),
        panel_label(neutralized, "NEUTRALIZED CROP"),
        panel_label(old_out, "RAW Img2Img OLD 0.55"),
        panel_label(old_composite, "COMPOSITE OLD 0.55"),
    ]
    for tag, strength in STRENGTHS.items():
        panels.append(panel_label(variant_raws[tag], f"RAW {tag} s={strength}"))
        panels.append(panel_label(variant_composites[tag], f"COMPOSITE {tag}"))
        panels.append(
            panel_label(variant_composites[tag].crop((zx1, zy1, zx2, zy2)), f"ZOOM {tag}")
        )
    if clean_copy.is_file():
        panels.append(panel_label(Image.open(clean_copy).convert("RGB"), "CLEAN REF ONLY"))

    master = grid(panels, cols=4)
    master_path = DEBUG_ROOT / "00_master_comparison.jpg"
    master.save(master_path, quality=95)

    zoom_panels = [
        panel_label(attacked.crop((zx1, zy1, zx2, zy2)), "ATTACKED ZOOM", (280, 280)),
        panel_label(
            old_composite.crop((zx1, zy1, zx2, zy2)), "OLD 0.55 NO NEUT", (280, 280)
        ),
    ]
    for tag, strength in STRENGTHS.items():
        zoom_panels.append(
            panel_label(
                variant_composites[tag].crop((zx1, zy1, zx2, zy2)),
                f"{tag} s={strength}",
                (280, 280),
            )
        )
    if clean_copy.is_file():
        zoom_panels.append(
            panel_label(
                Image.open(clean_copy).convert("RGB").crop((zx1, zy1, zx2, zy2)),
                "CLEAN REF",
                (280, 280),
            )
        )
    zoom = hstack(zoom_panels)
    zoom_path = DEBUG_ROOT / "00_zoom_strength_comparison.jpg"
    zoom.save(zoom_path, quality=95)

    results_summary["mapping_matches"] = mapping["back_projection_check"]["matches_original_patch"]
    results_summary["master_comparison"] = str(master_path)
    results_summary["zoom_comparison"] = str(zoom_path)
    results_summary["coordinate_mapping"] = str(DEBUG_ROOT / "coordinate_mapping.json")
    results_summary["clean_pixels_used_for_restoration"] = False
    (DEBUG_ROOT / "debug_summary.json").write_text(
        json.dumps(results_summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(results_summary, indent=2))
    print(f"DEBUG_ROOT={DEBUG_ROOT}")


if __name__ == "__main__":
    main()
