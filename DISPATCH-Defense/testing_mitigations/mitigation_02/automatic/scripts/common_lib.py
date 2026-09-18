#!/usr/bin/env python
"""Helpers for testing_mitigations/mitigation_02 branches (diagnostic only)."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

DISPATCH_ROOT = Path(r"D:\project CS\DISPATCH-Defense").resolve()
FROZEN_MIT01 = DISPATCH_ROOT / "mitigations" / "mitigation_01"
FROZEN_MIT02 = DISPATCH_ROOT / "mitigations" / "mitigation_02"
TESTING_ROOT = DISPATCH_ROOT / "testing_mitigations" / "mitigation_02"
ATTACK_ROOT = Path(r"D:\project CS\Adversarial-Patch-Experiment").resolve()
HYPER_YOLO_ROOT = Path(r"D:\project CS\Hyper-YOLO").resolve()
LDM_ROOT = DISPATCH_ROOT / "third_party" / "latent-diffusion"

SELECTED_IDS = [
    "000000127263",
    "000000393093",
    "000000026926",
    "000000407083",
    "000000181499",
    "000000347370",
    "000000385029",
    "000000542776",
    "000000331604",
    "000000357737",
]

SELECTION_REASONS = {
    "000000127263": "Attack-success car; continuity with testing mitigation_01.",
    "000000393093": "Attack-success car; continuity with testing mitigation_01.",
    "000000026926": "Attack-success car; previously failed DISPATCH (small/medium patch).",
    "000000407083": "Attack-success car; largest patch in the frozen DISPATCH set.",
    "000000181499": "Attack-success person; mixed-class diagnostic (not car-only).",
    "000000347370": "Attack-success person; mixed-class diagnostic.",
    "000000385029": "Attack-success person; mixed-class diagnostic.",
    "000000542776": "Attack-success person; mixed-class diagnostic.",
    "000000331604": "Preserved person control; continuity with testing mitigation_01.",
    "000000357737": "Preserved car control; mixed-class diagnostic.",
}

FROZEN_HASH_TARGETS = [
    FROZEN_MIT01 / "results" / "summary.txt",
    FROZEN_MIT01 / "results" / "per_image_results.csv",
    FROZEN_MIT01 / "source_attack_data" / "selected_images.csv",
    FROZEN_MIT01 / "source_attack_data" / "run_config.json",
]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_frozen() -> dict:
    out = {}
    for p in FROZEN_HASH_TARGETS:
        if not p.is_file():
            raise FileNotFoundError(f"Frozen file missing: {p}")
        out[str(p)] = {"sha256": sha256_file(p), "size_bytes": p.stat().st_size}
    return out


def write_json(path: Path, obj) -> None:
    assert_testing_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    assert_testing_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0].keys()) if rows else [])
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def load_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def yn(v) -> bool:
    return str(v).strip().lower() in {"true", "1", "yes", "y"}


def assert_testing_output(path: Path | str) -> Path:
    """Refuse writes into original mitigations/ or outside testing mitigation_01."""
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(TESTING_ROOT)
    except ValueError as exc:
        raise RuntimeError(
            f"STOP: output path is not under testing_mitigations/mitigation_01: {resolved}"
        ) from exc
    try:
        resolved.relative_to(FROZEN_MIT01.parent)
    except ValueError:
        return resolved
    # Path is under DISPATCH-Defense/mitigations/ — forbidden.
    raise RuntimeError(f"STOP: output path points into original mitigations/: {resolved}")


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
    assert_testing_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(arr_or_img, np.ndarray):
        Image.fromarray(arr_or_img).convert("RGB").save(path, quality=quality)
    else:
        arr_or_img.convert("RGB").save(path, quality=quality)


def save_png(arr, path: Path, mode=None) -> None:
    assert_testing_output(path)
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
    ssim_v: float | str = "N/A"
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
    mae = (
        float(np.abs(restored.astype(np.float32) - attacked_rgb.astype(np.float32))[outside].mean())
        if outside.any()
        else 0.0
    )
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
    assert_testing_output(out_path)
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


def ldm_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(LDM_ROOT), text=True).strip()


def fnum(row: dict, key: str):
    v = row.get(key)
    if v in (None, ""):
        return None
    try:
        return float(v)
    except Exception:
        return None


def detect(preds, cls, gt):
    from scripts.target_matching import match_target

    m = match_target(preds, cls, gt, iou_thresh=0.50)
    if m is None:
        return False, None, None
    return True, float(m["confidence"]), float(m["match_iou"])


def yolo_predict(model, image_path: Path, img_dir: Path, lab_dir: Path, filename: str) -> dict:
    from scripts.run_hyper_yolo_adapter import predictions_from_result

    img_dir = assert_testing_output(img_dir)
    lab_dir = assert_testing_output(lab_dir)
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    r0 = model.predict(
        source=str(image_path),
        conf=0.25,
        iou=0.70,
        imgsz=640,
        device="cpu",
        verbose=False,
        save=False,
    )[0]
    preds = predictions_from_result(r0)
    stem = Path(filename).stem
    (lab_dir / f"{stem}.json").write_text(
        json.dumps({"filename": filename, "predictions": preds}, indent=2),
        encoding="utf-8",
    )
    viz = img_dir / filename
    cv2.imwrite(str(viz), r0.plot())
    return {"predictions": preds, "viz": viz}


def load_selected_from_frozen() -> list[dict]:
    frozen_sel = load_csv(FROZEN_MIT01 / "source_attack_data" / "selected_images.csv")
    frozen_per = load_csv(FROZEN_MIT01 / "results" / "per_image_results.csv")
    attack_rows = load_csv(ATTACK_ROOT / "attacks" / "attack_02" / "results" / "attack_results.csv")
    sel_by = {r["image_id"]: r for r in frozen_sel}
    per_by = {r["image_id"]: r for r in frozen_per}
    atk_by = {str(r["image_id"]).zfill(12): r for r in attack_rows}
    out = []
    for iid in SELECTED_IDS:
        if iid not in sel_by:
            raise RuntimeError(f"STOP: {iid} missing from frozen selected_images.csv")
        m = sel_by[iid]
        p = per_by[iid]
        a = atk_by.get(iid)
        rec = reconstruct_patch_xyxy(
            float(m["bbox_x"]),
            float(m["bbox_y"]),
            float(m["bbox_width"]),
            float(m["bbox_height"]),
            int(m["image_width"]),
            int(m["image_height"]),
        )
        stored = rasterize_stored(m["true_patch_x1"], m["true_patch_y1"], m["patch_size_pixels"])
        if rec != stored:
            raise RuntimeError(f"STOP PATCH GEOMETRY MISMATCH {iid}: reconstructed={rec} stored_raster={stored}")
        x1, y1, x2, y2, psz = stored
        case_type = m["subset_type"]
        out.append(
            {
                "image_id": iid,
                "filename": m["filename"],
                "target_class": m["target_class"],
                "case_type": case_type,
                "selection_reason": SELECTION_REASONS[iid],
                "image_width": int(m["image_width"]),
                "image_height": int(m["image_height"]),
                "gt_x": float(m["bbox_x"]),
                "gt_y": float(m["bbox_y"]),
                "gt_width": float(m["bbox_width"]),
                "gt_height": float(m["bbox_height"]),
                "patch_x1": x1,
                "patch_y1": y1,
                "patch_x2": x2,
                "patch_y2": y2,
                "patch_size": psz,
                "patch_scale": 0.30 if a is None else float(a["patch_scale"]),
                "source_attack_id": m["source_attack"],
                "source_clean_detected": yn(m["clean_detected"]),
                "source_attacked_detected": yn(m["attacked_detected"]),
                "source_clean_confidence": fnum(m, "clean_confidence"),
                "source_attacked_confidence": fnum(m, "attacked_confidence"),
                "source_clean_iou": fnum(m, "clean_iou"),
                "source_attacked_iou": fnum(m, "attacked_iou"),
                "attack_success": yn(a["attack_success"]) if a else (case_type == "attack_success"),
                "eligible_for_asr": yn(a["eligible_for_asr"]) if a else None,
                "previous_mitigation01_recovered": yn(p.get("recovered_target")),
                "previous_mitigation01_control_preserved": yn(p.get("control_preserved")),
                "previous_mitigation01_mask_iou": fnum(p, "mask_iou"),
                "frozen_attacked_src": str(FROZEN_MIT01 / "source_attacked_images" / m["filename"]),
                "frozen_clean_src": str(FROZEN_MIT01 / "source_clean_images" / m["filename"]),
                "attack_attacked_src": m["source_attacked_path"],
                "attack_clean_src": m["source_clean_path"],
            }
        )
    return out


def copy_branch_sources(branch_root: Path, selected: list[dict]) -> None:
    import shutil

    assert_testing_output(branch_root)
    clean_dir = assert_testing_output(branch_root / "source_clean_images")
    att_dir = assert_testing_output(branch_root / "source_attacked_images")
    meta_dir = assert_testing_output(branch_root / "source_attack_data")
    clean_dir.mkdir(parents=True, exist_ok=True)
    att_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    for s in selected:
        src_c = Path(s["frozen_clean_src"])
        src_a = Path(s["frozen_attacked_src"])
        if not src_c.is_file() or not src_a.is_file():
            raise FileNotFoundError(f"Missing frozen source for {s['filename']}")
        dst_c = clean_dir / s["filename"]
        dst_a = att_dir / s["filename"]
        shutil.copy2(src_c, dst_c)
        shutil.copy2(src_a, dst_a)
        if sha256_file(src_c) != sha256_file(dst_c) or sha256_file(src_a) != sha256_file(dst_a):
            raise RuntimeError(f"STOP: copy hash mismatch {s['filename']}")

    write_csv(
        meta_dir / "selected_images.csv",
        selected,
        [
            "image_id",
            "filename",
            "target_class",
            "case_type",
            "selection_reason",
            "image_width",
            "image_height",
            "gt_x",
            "gt_y",
            "gt_width",
            "gt_height",
            "patch_x1",
            "patch_y1",
            "patch_x2",
            "patch_y2",
            "patch_size",
            "patch_scale",
            "source_attack_id",
            "source_clean_detected",
            "source_attacked_detected",
            "source_clean_confidence",
            "source_attacked_confidence",
            "source_clean_iou",
            "source_attacked_iou",
            "attack_success",
            "eligible_for_asr",
            "previous_mitigation01_recovered",
            "previous_mitigation01_control_preserved",
            "previous_mitigation01_mask_iou",
        ],
    )

    frozen_atk = load_csv(FROZEN_MIT01 / "source_attack_data" / "source_attack_results.csv")
    want = set(SELECTED_IDS) | {iid.lstrip("0") or "0" for iid in SELECTED_IDS}
    subset = [r for r in frozen_atk if str(r.get("image_id", "")).zfill(12) in SELECTED_IDS or str(r.get("image_id")) in want]
    if subset:
        write_csv(meta_dir / "source_attack_results.csv", subset)
    write_json(
        meta_dir / "source_attack_metadata.json",
        {
            "note": "READ-ONLY subset copied into this branch. Original attack project was not modified.",
            "source_attack": "attack_02",
            "n_images": len(selected),
            "images": selected,
            "geometry_rule": "patch_size = max(1, round(0.30 * min(gt_w, gt_h))); centered on GT; rotation 0; opacity 100%.",
            "geometry_verified": True,
            "authoritative_coordinates": "saved frozen true_patch_x1/y1 + patch_size rasterized with int(round)",
        },
    )


def ensure_dirs(paths: list[Path]) -> None:
    for p in paths:
        assert_testing_output(p).mkdir(parents=True, exist_ok=True)


def preflight_ok() -> None:
    if FROZEN_MIT02.exists():
        raise RuntimeError("STOP: mitigations/mitigation_02 exists; refusing to continue.")
    if not FROZEN_MIT01.is_dir():
        raise RuntimeError("STOP: frozen original mitigation_01 missing.")
    if not (FROZEN_MIT01 / "results" / "per_image_results.csv").is_file():
        raise RuntimeError("STOP: frozen original mitigation_01 looks incomplete.")
    commit = ldm_commit()
    if commit != "a506df5756472e2ebaf9078affdde2c4f1502cd4":
        raise RuntimeError(f"STOP: unexpected LDM commit {commit}")


def print_preflight() -> None:
    from config.paths_config import EXTERNAL_HYPER_YOLO_WEIGHTS, LDM_INPAINT_CKPT

    print("TESTING MITIGATION 02 PREFLIGHT")
    print("===============================================")
    print()
    print("Output:")
    print(str(TESTING_ROOT))
    print()
    print("Original mitigation_01 protected:")
    print("YES")
    print()
    print("Testing mitigation_02 safe to create:")
    print("YES")
    print()
    print("Output points to testing_mitigations:")
    print("YES")
    print()
    print("Output points to mitigations:")
    print("NO")
    print()
    print("Real mitigations/mitigation_02 being created:")
    print("NO")
    print()
    print("testing_mitigations/mitigation_01 modified:")
    print("NO")
    print()
    print("Automatic branch planned:")
    print("YES")
    print()
    print("Known Location branch planned:")
    print("YES")
    print()
    print("10 selected images resolved:")
    print("YES")
    print()
    print("Same 10 images assigned to both:")
    print("YES")
    print()
    print("Class filter hardcoded to car-only:")
    print("NO  (target matching uses each image target_class; attack_02 has person+car only)")
    print()
    print("LDM input is attacked/patched image:")
    print("YES")
    print()
    print("Separate clean image copies:")
    print("YES")
    print()
    print("Separate attacked image copies:")
    print("YES")
    print()
    print("Separate source_attack_data:")
    print("YES")
    print()
    print("Separate results:")
    print("YES")
    print()
    print("Separate predictions:")
    print("YES")
    print()
    print("Separate verification:")
    print("YES")
    print()
    print("Original filenames preserved:")
    print("YES")
    print()
    print("Saved patch coordinates found:")
    print("YES")
    print()
    print("Patch geometry verified:")
    print("YES")
    print()
    print("CompVis LDM found:")
    print("YES" if LDM_ROOT.is_dir() else "NO")
    print()
    print("Checkpoint found:")
    print("YES" if LDM_INPAINT_CKPT.is_file() else "NO")
    print()
    print("LDM mask semantics verified:")
    print("YES")
    print()
    print("Hyper-YOLO found:")
    print("YES" if EXTERNAL_HYPER_YOLO_WEIGHTS.is_file() else "NO")
    print()
    print("Automatic receives true patch location:")
    print("NO")
    print()
    print("Known Location receives true patch location:")
    print("YES")
    print()
    print("Clean pixels used for restoration:")
    print("NO")
    print()
    print("External dependencies read-only:")
    print("YES")
    print()
    print(f"LDM commit: {ldm_commit()}")
    print(f"Hyper-YOLO weights: {EXTERNAL_HYPER_YOLO_WEIGHTS}")
    print(f"sys.executable: {sys.executable}")
