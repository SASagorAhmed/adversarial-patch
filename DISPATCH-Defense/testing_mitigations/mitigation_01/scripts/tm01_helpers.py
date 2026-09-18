#!/usr/bin/env python
"""Shared helpers for testing_mitigations/mitigation_01 (diagnostic only)."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0].keys()) if rows else [])
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def yn(v) -> bool:
    return str(v).strip().lower() in {"true", "1", "yes", "y"}


def reconstruct_patch_xyxy(gt_x, gt_y, gt_w, gt_h, image_width, image_height):
    patch_size = max(1, int(round(0.30 * min(float(gt_w), float(gt_h)))))
    center_x = float(gt_x) + float(gt_w) / 2.0
    center_y = float(gt_y) + float(gt_h) / 2.0
    patch_x = center_x - patch_size / 2.0
    patch_y = center_y - patch_size / 2.0
    patch_x = max(0.0, min(patch_x, float(image_width) - patch_size))
    patch_y = max(0.0, min(patch_y, float(image_height) - patch_size))
    x1 = int(round(patch_x))
    y1 = int(round(patch_y))
    return x1, y1, x1 + patch_size, y1 + patch_size, patch_size


def rasterize_stored(x1, y1, patch_size):
    ix1 = int(round(float(x1)))
    iy1 = int(round(float(y1)))
    ps = int(round(float(patch_size)))
    return ix1, iy1, ix1 + ps, iy1 + ps, ps


def make_rect_mask(h, w, x1, y1, x2, y2) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.uint8)
    m[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = 255
    return m


def expand_rect(x1, y1, x2, y2, expand, w, h):
    return max(0, x1 - expand), max(0, y1 - expand), min(w, x2 + expand), min(h, y2 + expand)


def overlay_mask(rgb, mask, color=(255, 0, 0), alpha=0.45) -> np.ndarray:
    out = rgb.astype(np.float32)
    m = (mask > 127)[..., None]
    col = np.array(color, dtype=np.float32)
    out = np.where(m, out * (1.0 - alpha) + col * alpha, out)
    return np.clip(out, 0, 255).astype(np.uint8)


def save_jpg(arr_or_img, path: Path, quality=95) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(arr_or_img, np.ndarray):
        Image.fromarray(arr_or_img).convert("RGB").save(path, quality=quality)
    else:
        arr_or_img.convert("RGB").save(path, quality=quality)


def save_png(arr, path: Path, mode=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if mode:
        Image.fromarray(arr, mode=mode).save(path)
    else:
        Image.fromarray(arr).save(path)


def patch_similarity(clean_rgb, restored_rgb, mask) -> dict:
    m = mask > 127
    if m.sum() == 0:
        return {"mae": None, "mse": None, "psnr": None, "ssim": "N/A"}
    a = clean_rgb.astype(np.float32)[m]
    b = restored_rgb.astype(np.float32)[m]
    mae = float(np.mean(np.abs(a - b)))
    mse = float(np.mean((a - b) ** 2))
    psnr = float("inf") if mse <= 1e-12 else float(20.0 * math.log10(255.0 / math.sqrt(mse)))
    ys, xs = np.where(m)
    h = int(ys.max() - ys.min() + 1)
    w = int(xs.max() - xs.min() + 1)
    ssim_v = "N/A"
    if h >= 7 and w >= 7:
        y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
        ssim_v = _ssim_rgb(clean_rgb[y0:y1, x0:x1], restored_rgb[y0:y1, x0:x1], mask[y0:y1, x0:x1])
        if ssim_v is None or (isinstance(ssim_v, float) and (math.isnan(ssim_v) or math.isinf(ssim_v))):
            ssim_v = "N/A"
    return {"mae": mae, "mse": mse, "psnr": psnr, "ssim": ssim_v}


def _ssim_rgb(a, b, mask=None) -> float:
    vals = [_ssim_gray(a[..., c].astype(np.float64), b[..., c].astype(np.float64), mask) for c in range(3)]
    return float(np.mean(vals))


def _ssim_gray(x, y, mask=None) -> float:
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    if mask is not None:
        m = mask > 127
        if m.sum() < 16:
            return float("nan")
        x, y = x[m], y[m]
    mu_x, mu_y = float(x.mean()), float(y.mean())
    sig_x, sig_y = float(x.var()), float(y.var())
    sig_xy = float(((x - mu_x) * (y - mu_y)).mean())
    den = (mu_x**2 + mu_y**2 + C1) * (sig_x + sig_y + C2)
    return float(((2 * mu_x * mu_y + C1) * (2 * sig_xy + C2)) / den) if den else 1.0


def restore_known_at_original(adapter, attacked_rgb, mask_orig, steps, resolution=512):
    orig_h, orig_w = attacked_rgb.shape[:2]
    proc_img = np.array(Image.fromarray(attacked_rgb).resize((resolution, resolution), Image.Resampling.LANCZOS))
    proc_mask = np.array(
        Image.fromarray(mask_orig, mode="L").resize((resolution, resolution), Image.Resampling.NEAREST)
    )
    ldm_512 = adapter.inpaint(proc_img, proc_mask, steps=steps)
    ldm_up = np.array(Image.fromarray(ldm_512).resize((orig_w, orig_h), Image.Resampling.LANCZOS))
    m = (mask_orig > 127)[..., None]
    restored = np.where(m, ldm_up, attacked_rgb).astype(np.uint8)
    outside = ~m[..., 0]
    mae = float(np.abs(restored.astype(np.float32) - attacked_rgb.astype(np.float32))[outside].mean()) if outside.any() else 0.0
    return {"ldm_512": ldm_512, "ldm_up": ldm_up, "restored": restored, "outside_mask_mae": mae}


def _font(size: int):
    for p in (
        r"C:\Windows\Fonts\arialbd.ttf",
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
    ):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def labeled_thumb(im: Image.Image, label: str, tw=480) -> Image.Image:
    label_h = 56
    t = im.convert("RGB").copy()
    t.thumbnail((tw, tw))
    canvas = Image.new("RGB", (tw, tw + label_h), (30, 30, 30))
    ox, oy = (tw - t.width) // 2, (tw - t.height) // 2
    canvas.paste(t, (ox, oy))
    dr = ImageDraw.Draw(canvas)
    dr.rectangle([0, tw, tw, tw + label_h], fill=(12, 12, 12))
    font = _font(15)
    # wrap into two lines for long panel titles
    if len(label) > 36:
        cut = label.rfind(" ", 0, 36)
        cut = cut if cut > 12 else 36
        line1, line2 = label[:cut].strip(), label[cut:].strip()
    else:
        line1, line2 = label, ""
    dr.text((8, tw + 6), line1, fill=(255, 230, 80), font=font)
    if line2:
        dr.text((8, tw + 28), line2, fill=(255, 230, 80), font=font)
    return canvas


def make_grid(panels: list[tuple[str, Image.Image]], out_path: Path, title: str, cols=4, tw=480) -> None:
    thumbs = [labeled_thumb(im, lab, tw) for lab, im in panels]
    rows = (len(thumbs) + cols - 1) // cols
    gap, header, label_h = 10, 56, 56
    cell_h = tw + label_h
    W = cols * tw + (cols + 1) * gap
    H = header + rows * cell_h + (rows + 1) * gap
    canvas = Image.new("RGB", (W, H), (245, 245, 245))
    dr = ImageDraw.Draw(canvas)
    dr.rectangle([0, 0, W, header], fill=(25, 40, 70))
    dr.text((12, 16), title, fill=(255, 255, 255), font=_font(18))
    for i, th in enumerate(thumbs):
        r, c = divmod(i, cols)
        x = gap + c * (tw + gap)
        y = header + gap + r * (cell_h + gap)
        canvas.paste(th, (x, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=95)
