"""Visualization helpers. All writes go through safety.assert_writable."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from helpers.safety import assert_writable


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, color=(255, 220, 0), alpha=0.45) -> np.ndarray:
    out = rgb.astype(np.float32)
    m = (mask > 127)[..., None]
    col = np.array(color, dtype=np.float32)
    return np.clip(np.where(m, out * (1 - alpha) + col * alpha, out), 0, 255).astype(np.uint8)


def to_vis_png(d: np.ndarray) -> np.ndarray:
    dmin, dmax = float(d.min()), float(d.max())
    if dmax <= dmin:
        return np.zeros_like(d, dtype=np.uint8)
    return ((d - dmin) / (dmax - dmin) * 255.0).astype(np.uint8)


def _font(size: int = 16):
    for p in (r"C:\Windows\Fonts\arialbd.ttf", r"C:\Windows\Fonts\arial.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def labeled_panel(im: Image.Image, label: str, tw: int = 280) -> Image.Image:
    label_h = 36
    t = im.convert("RGB").copy()
    t.thumbnail((tw, tw))
    canvas = Image.new("RGB", (tw, tw + label_h), (28, 28, 28))
    canvas.paste(t, ((tw - t.width) // 2, (tw - t.height) // 2))
    dr = ImageDraw.Draw(canvas)
    dr.rectangle([0, tw, tw, tw + label_h], fill=(8, 8, 8))
    dr.text((8, tw + 8), label[:42], fill=(240, 240, 240), font=_font(14))
    return canvas


def hstack(panels: list[Image.Image]) -> Image.Image:
    w, h = panels[0].size
    out = Image.new("RGB", (w * len(panels), h), (0, 0, 0))
    for i, p in enumerate(panels):
        out.paste(p, (i * w, 0))
    return out


def save_jpg(arr_or_img, path, quality=95) -> None:
    path = assert_writable(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(arr_or_img, np.ndarray):
        Image.fromarray(arr_or_img).convert("RGB").save(path, quality=quality)
    else:
        arr_or_img.convert("RGB").save(path, quality=quality)


def save_png(arr, path, mode="L") -> None:
    path = assert_writable(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode=mode).save(path)
