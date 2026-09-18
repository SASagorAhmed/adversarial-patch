#!/usr/bin/env python
"""5-image visual Automatic DisPatch test.

Writes ONLY under original_model_5image_test/.
Uses existing paper-faithful implementation. Attacked image is the only restoration input.
"""
from __future__ import annotations

import csv
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

TEST_ROOT = Path(r"D:\project CS\DISPATCH-Paper-Faithful\original_model_5image_test").resolve()
PF_ROOT = Path(r"D:\project CS\DISPATCH-Paper-Faithful").resolve()
if str(PF_ROOT) not in sys.path:
    sys.path.insert(0, str(PF_ROOT))

from configs.reference_config import (  # noqa: E402
    DIFFUSION_STEPS,
    GAUSSIAN_SIGMA,
    LDM_CKPT,
    MORPH_OPEN_KERNEL,
    NUM_GRIDS,
    OFFICIAL_COMMIT,
    RESOLUTION,
    RUN_SEED,
)
from implementation.evaluation import localization_metrics  # noqa: E402
from implementation.ldm_adapter import OfficialLDMAdapter  # noqa: E402
from implementation.localization import detect_adversarial_mask, gaussian_kernel_size  # noqa: E402
from implementation.rectification import mask_to_uint8_orig, rectify, resize_to_original, to_pil_rgb  # noqa: E402
from implementation.regeneration import make_checkerboard_masks, validate_complementary  # noqa: E402

FILENAMES = [
    "000000127263.jpg",
    "000000393093.jpg",
    "000000026926.jpg",
    "000000407083.jpg",
    "000000542776.jpg",
]
SRC_ATT = PF_ROOT / "diagnostic" / "source_attacked_images"
SRC_CLN = PF_ROOT / "diagnostic" / "source_clean_images"
SRC_SEL = PF_ROOT / "diagnostic" / "source_attack_data" / "selected_images.csv"


def assert_test(path: Path | str) -> Path:
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(TEST_ROOT)
    except ValueError as exc:
        raise RuntimeError(f"STOP: write outside original_model_5image_test: {resolved}") from exc
    return resolved


def seed_all(seed: int = RUN_SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def overlay_mask(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = rgb.astype(np.float32)
    m = (mask > 127)[..., None]
    col = np.array([255.0, 220.0, 0.0])
    return np.clip(np.where(m, out * 0.55 + col * 0.45, out), 0, 255).astype(np.uint8)


def labeled_panel(im: Image.Image, label: str, tw: int = 420) -> Image.Image:
    label_h = 40
    t = im.convert("RGB").copy()
    t.thumbnail((tw, tw))
    canvas = Image.new("RGB", (tw, tw + label_h), (24, 24, 24))
    canvas.paste(t, ((tw - t.width) // 2, (tw - t.height) // 2))
    dr = ImageDraw.Draw(canvas)
    dr.rectangle([0, tw, tw, tw + label_h], fill=(10, 10, 10))
    try:
        font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    dr.text((10, tw + 10), label, fill=(240, 240, 240), font=font)
    return canvas


def montage(panels: list[Image.Image]) -> Image.Image:
    w, h = panels[0].size
    out = Image.new("RGB", (w * len(panels), h), (0, 0, 0))
    for i, p in enumerate(panels):
        out.paste(p, (i * w, 0))
    return out


def true_mask(h: int, w: int, meta: dict | None) -> np.ndarray | None:
    if not meta:
        return None
    x1, y1, x2, y2 = int(meta["patch_x1"]), int(meta["patch_y1"]), int(meta["patch_x2"]), int(meta["patch_y2"])
    m = np.zeros((h, w), dtype=np.uint8)
    m[max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)] = 255
    return m


def main() -> int:
    dirs = {
        "att": TEST_ROOT / "source_attacked_images",
        "cln": TEST_ROOT / "source_clean_images",
        "regen": TEST_ROOT / "regenerated_images",
        "masks": TEST_ROOT / "predicted_masks",
        "ov": TEST_ROOT / "mask_overlays",
        "rest": TEST_ROOT / "restored_images",
        "ver": TEST_ROOT / "verification",
        "res": TEST_ROOT / "results",
        "logs": TEST_ROOT / "logs",
    }
    for d in dirs.values():
        assert_test(d).mkdir(parents=True, exist_ok=True)

    missing = [fn for fn in FILENAMES if not (SRC_ATT / fn).is_file()]
    if missing:
        print("STOP: missing attacked images:", missing)
        return 2

    meta_by = {}
    if SRC_SEL.is_file():
        with SRC_SEL.open(encoding="utf-8", newline="") as f:
            meta_by = {r["filename"]: r for r in csv.DictReader(f)}

    sel_rows = []
    for fn in FILENAMES:
        src_a = SRC_ATT / fn
        dst_a = assert_test(dirs["att"] / fn)
        shutil.copy2(src_a, dst_a)
        src_c = SRC_CLN / fn
        dst_c = assert_test(dirs["cln"] / fn)
        clean_ok = src_c.is_file()
        if clean_ok:
            shutil.copy2(src_c, dst_c)
        m = meta_by.get(fn, {})
        sel_rows.append(
            {
                "filename": fn,
                "source_attacked_path": str(src_a),
                "local_attacked_copy": str(dst_a),
                "source_clean_path": str(src_c) if clean_ok else "",
                "local_clean_copy": str(dst_c) if clean_ok else "",
                "target_class_if_known": m.get("target_class", ""),
                "patch_size_if_known": m.get("patch_size", ""),
                "status": "copied",
            }
        )

    copied = list(dirs["att"].glob("*.jpg"))
    ckpt_ok = LDM_CKPT.is_file()
    print("PAPER-FAITHFUL 5-IMAGE RESTORATION PREFLIGHT")
    print("=============================================")
    flags = {
        "Output root safe": str(TEST_ROOT).endswith("original_model_5image_test"),
        "Official implementation available": (PF_ROOT / "implementation" / "ldm_adapter.py").is_file(),
        "Official configuration resolved": True,
        "5 attacked images resolved": len(missing) == 0,
        "5 attacked images copied locally": len(copied) == 5,
        "Original filenames preserved": all((dirs["att"] / fn).is_file() for fn in FILENAMES),
        "Clean images excluded from restoration": True,
        "True patch location excluded from Automatic": True,
        "LDM checkpoint available": ckpt_ok,
        "Existing projects read-only": True,
    }
    for k, v in flags.items():
        print(f"{k}: {'YES' if v else 'NO'}")
    if not all(flags.values()):
        print("STOP: preflight failed")
        return 2

    seed_all()
    mask0, mask1 = make_checkerboard_masks(RESOLUTION, NUM_GRIDS)
    val = validate_complementary(mask0, mask1)
    if not val["sum_allclose_1"]:
        raise RuntimeError(val)

    print("Loading official LDM adapter...")
    adapter = OfficialLDMAdapter(steps=DIFFUSION_STEPS)
    ksize = gaussian_kernel_size(RESOLUTION, NUM_GRIDS)
    per = []
    sample_modes = set()

    for fn in FILENAMES:
        stem = Path(fn).stem
        att_path = dirs["att"] / fn
        print(f"\n=== VISUAL AUTOMATIC {fn} ===")
        t0 = time.perf_counter()
        seed_all()
        regen = adapter.regenerate_pair(att_path, mask0, mask1, size=RESOLUTION)
        sample_modes.add(regen.get("sample_mode"))
        gen = regen["generated_01"]
        inp = regen["input_01"]
        ori_w, ori_h = regen["ori_size"]

        gen_orig = resize_to_original(to_pil_rgb(gen), (ori_w, ori_h))
        regen_path = assert_test(dirs["regen"] / fn)
        gen_orig.save(regen_path, quality=95)

        adv, _dist, km_stats = detect_adversarial_mask(
            inp, gen, size=RESOLUTION, num_grids=NUM_GRIDS, kmeans_random_state=RUN_SEED
        )
        mask_orig = mask_to_uint8_orig(adv, (ori_w, ori_h))
        mask_path = assert_test(dirs["masks"] / f"{stem}.png")
        Image.fromarray(mask_orig, mode="L").save(mask_path)

        attacked = np.array(Image.open(att_path).convert("RGB"))
        ov_path = assert_test(dirs["ov"] / fn)
        Image.fromarray(overlay_mask(attacked, mask_orig)).save(ov_path, quality=95)

        loc = {}
        meta = meta_by.get(fn)
        tmask = true_mask(attacked.shape[0], attacked.shape[1], meta)
        if tmask is not None:
            loc = localization_metrics(mask_orig, tmask)

        rect = rectify(adv, gen, inp)
        rest_path = assert_test(dirs["rest"] / fn)
        resize_to_original(to_pil_rgb(rect), (ori_w, ori_h)).save(rest_path, quality=95)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        clean_path = dirs["cln"] / fn
        panels = []
        if clean_path.is_file():
            panels.append(labeled_panel(Image.open(clean_path), "CLEAN REFERENCE (eval only)"))
        panels.extend(
            [
                labeled_panel(Image.open(att_path), "ATTACKED / PATCHED INPUT"),
                labeled_panel(Image.fromarray(mask_orig).convert("RGB"), "PREDICTED MASK"),
                labeled_panel(Image.open(regen_path), "FULL REGENERATED (NOT FINAL)"),
                labeled_panel(Image.open(rest_path), "FINAL RESTORED"),
            ]
        )
        ver_path = assert_test(dirs["ver"] / fn)
        montage(panels).save(ver_path, quality=95)
        elapsed = time.perf_counter() - t0
        rec = {
            "filename": fn,
            "input_width": ori_w,
            "input_height": ori_h,
            "predicted_mask_area": int((mask_orig > 127).sum()),
            "regenerated_image_path": str(regen_path),
            "predicted_mask_path": str(mask_path),
            "restored_image_path": str(rest_path),
            "verification_path": str(ver_path),
            "inference_time_sec": elapsed,
            "sample_mode": regen.get("sample_mode"),
            "true_patch_area": loc.get("true_patch_area"),
            "mask_iou": loc.get("mask_iou"),
            "precision": loc.get("mask_precision"),
            "recall": loc.get("mask_recall"),
            "status": "completed",
            "error_message": "",
        }
        per.append(rec)
        print(f"  saved restored={rest_path.name} mask_area={rec['predicted_mask_area']} t={elapsed:.1f}s")

    cfg = {
        "official_source_commit": OFFICIAL_COMMIT,
        "ldm_checkpoint_path": str(LDM_CKPT),
        "resolution": RESOLUTION,
        "ddim_steps": DIFFUSION_STEPS,
        "checkerboard_N": NUM_GRIDS,
        "gaussian_kernel": ksize,
        "gaussian_sigma": GAUSSIAN_SIGMA,
        "morphology": f"MORPH_OPEN {MORPH_OPEN_KERNEL}x{MORPH_OPEN_KERNEL}",
        "kmeans": {"k": 2, "n_init": "auto", "random_state": RUN_SEED, "higher_centroid": "adversarial"},
        "sample_mode": sorted(sample_modes),
        "seed": RUN_SEED,
        "input_filenames": FILENAMES,
        "true_patch_supplied_to_automatic": False,
        "clean_pixels_used_for_restoration": False,
        "hyper_yolo_required": False,
        "checkerboard_validation": val,
    }
    (assert_test(dirs["res"] / "run_config.json")).write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    def write_csv(path: Path, rows: list[dict]) -> None:
        path = assert_test(path)
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    write_csv(dirs["res"] / "selected_images.csv", sel_rows)
    write_csv(dirs["res"] / "per_image_results.csv", per)

    n_rest = len(list(dirs["rest"].glob("*.jpg")))
    n_mask = len(list(dirs["masks"].glob("*.png")))
    n_reg = len(list(dirs["regen"].glob("*.jpg")))
    n_ver = len(list(dirs["ver"].glob("*.jpg")))
    summary = f"""OFFICIAL / PAPER-FAITHFUL DISPATCH
5-IMAGE VISUAL RESTORATION TEST

Number of attacked images:
5

True patch location supplied to Automatic:
NO

Clean pixels used for restoration:
NO

Input:
Adversarial-patched images

Automatic localization:
YES

Final output folder:
{dirs['rest']}

Regenerated images:
INTERMEDIATE ONLY  ({n_reg} files)

Restored images:
FINAL OUTPUT  ({n_rest} files)

Predicted masks: {n_mask}
Verification montages: {n_ver}
Official commit: {OFFICIAL_COMMIT}
Gaussian kernel={ksize} sigma={GAUSSIAN_SIGMA} MORPH_OPEN={MORPH_OPEN_KERNEL}
sample_mode={sorted(sample_modes)}
"""
    (assert_test(dirs["res"] / "summary.txt")).write_text(summary, encoding="utf-8")
    print("\n" + summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
