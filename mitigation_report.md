# Mitigation Analysis Report

**Project:** A Survey on Adversarial Patch Attacks on Target Detection  
**Scope:** Analysis only — mitigation planning for known-location diffusion restoration  
**Date of inspection:** 2026-07-27  
**Experiment root:** `D:\project CS\Adversarial-Patch-Experiment\`  
**Read-only dependencies:** `Hyper-YOLO\`, `Stable-Diffusion-Patch\`  

**Attack phase status:** PERMANENTLY FROZEN (attack_01–attack_05 treated as read-only source data)

---

## A. Current mitigation feasibility: READY

**Verdict: READY** (with scientific and technical caveats listed in section K).

| Check | Status |
|-------|--------|
| Attack CSVs contain required fields | YES |
| GT bbox available to reconstruct patch square | YES (`selected_images.csv`) |
| Patch geometry formula reproducible | YES (matches `pipeline_core.compute_patch_geometry`) |
| Attacked images available under each attack folder | YES (`patched_images\`) |
| Local SD 2.1 Base present | YES |
| `StableDiffusionImg2ImgPipeline` importable | YES |
| CUDA available in SD venv | YES (NVIDIA GeForce RTX 2050) |
| No new model download required for planned Img2Img load path | YES (`local_files_only=True` intended) |
| Proposed subset pools large enough | YES |

This is a **simplified known-location diffusion-based restoration baseline**, not a reproduction of DIFFender / DiffPAD / dedicated inpainting defenses. Stable Diffusion 2.1 Base is **not** a dedicated inpainting checkpoint.

---

## B. Exact source files / CSV fields available for mitigation

### Primary per-image master table (recommended join key: `filename`)

**File:**  
`attacks\attack_XX\results\attack_results.csv`

**Relevant columns already present:**

| Need | Column(s) |
|------|-----------|
| image_id | `image_id` |
| filename | `filename` |
| target_class | `target_class` |
| clean_detected | `clean_detected` |
| attacked_detected | `attacked_detected` |
| attack_success | `attack_success` |
| eligible_for_asr | `eligible_for_asr` |
| clean confidence | `clean_confidence` |
| attacked confidence | `attacked_confidence` |
| clean IoU | `clean_iou` |
| attacked IoU | `attacked_iou` |
| patch size | `patch_size_pixels` |
| patch scale | `patch_scale` (0.3) |
| patch center | `patch_center_x`, `patch_center_y` |
| patch filename | `patch_filename` |

**Not stored as columns:** `patch_x`, `patch_y` (top-left). These are **reconstructible** (section C).

### GT bounding box + image size (required for exact geometry)

**File:**  
`attacks\attack_XX\results\selected_images.csv`

**Columns:**  
`image_id`, `filename`, `target_class`, `class_id`,  
`bbox_x`, `bbox_y`, `bbox_width`, `bbox_height`,  
`image_width`, `image_height`, `random_seed`

### Comparison-focused export (subset of same metrics)

**File:**  
`attacks\attack_XX\results\comparison\clean_vs_attacked.csv`  

Useful, but **less complete** than `attack_results.csv` (no patch centers). Prefer `attack_results.csv` + `selected_images.csv`.

### Attacked images to restore (do not modify originals)

**Folder:**  
`attacks\attack_XX\patched_images\*.jpg`

**Rule:** copy into mitigation folders or read-only open; never move/overwrite attack files.

### Inspected success / control pool sizes

| Attack | Success (`attack_success=Yes`) | Control pool (eligible, attacked still Yes) |
|--------|--------------------------------|---------------------------------------------|
| attack_02 | **15** (person 4, car 11) | **188** (person 102, car 86) |
| attack_05 | **20** (person 4, car 16) | **310** (person 170, car 140) |

Success cases are **car-heavy**. Exact 50/50 balance among successes is impossible; control selection can still be approximately class-balanced.

---

## C. Exact way patch coordinates can be reconstructed

Existing attack code in `scripts\pipeline_core.py`:

```text
PATCH_SCALE = 0.30

patch_size = max(1, int(round(0.30 * min(gt_w, gt_h))))
center_x = gt_x + gt_w / 2.0
center_y = gt_y + gt_h / 2.0
patch_x = center_x - patch_size / 2.0
patch_y = center_y - patch_size / 2.0
# clamp to image bounds:
patch_x = max(0.0, min(patch_x, image_width - patch_size))
patch_y = max(0.0, min(patch_y, image_height - patch_size))

# applied with:
x1 = int(round(patch_x)); y1 = int(round(patch_y))
x2 = x1 + patch_size;     y2 = y1 + patch_size
```

**Reconstruction recipe for mitigation (deterministic):**

1. Load GT box + image size from `selected_images.csv`.
2. Re-run the same `compute_patch_geometry` logic (prefer importing/reusing a **copied** helper or a new mitigation-local function that mirrors the frozen formula; do **not** change attack scripts).
3. Optionally cross-check `patch_size_pixels` and `patch_center_x/y` against `attack_results.csv` (inspection verified size match on sampled rows).
4. Build a binary mask over `[x1:x2, y1:y2]` for documentation.

**Conclusion:** Mitigation can reproduce the **exact patch square** without guessing, using only saved attack metadata + the frozen placement formula.

---

## D. Whether existing Stable Diffusion 2.1 Base supports the proposed Img2Img workflow

### Installed SD environment (inspected, no install/upgrade)

| Package | Version |
|---------|---------|
| torch | 2.6.0+cu124 |
| CUDA available | **True** |
| GPU | NVIDIA GeForce RTX 2050 |
| diffusers | 0.39.0 |
| transformers | 5.14.1 |
| accelerate | 1.14.0 |

### Pipeline availability

- `from diffusers import StableDiffusionImg2ImgPipeline` — **available**
- Local model dir: `D:\project CS\Stable-Diffusion-Patch\model` — **present**
- `model_index.json` `_class_name`: `StableDiffusionPipeline` (text-to-image save format)

**Interpretation:** The local weights are a standard SD 2.1 Base checkpoint. `StableDiffusionImg2ImgPipeline.from_pretrained(..., local_files_only=True)` is the intended load path and is the normal way to reuse the same UNet/VAE/text encoder for Img2Img **without downloading another model**. This is supported by the installed `diffusers` package.

**Wording requirement for thesis/report:** call this **Img2Img local restoration**, not “dedicated SD inpainting.”

### Recommended process architecture

Same pattern already used for candidate-patch generation:

1. Main orchestration runs under **Hyper-YOLO** Python (YOLO + OpenCV + CSV/plots).
2. Diffusion restoration runs as a **subprocess** under:

   `D:\project CS\Stable-Diffusion-Patch\.venv\Scripts\python.exe`

3. Pass crop paths / parameters via CLI; write restored crop / restored full image only into `mitigations\...`.
4. Never write into `Stable-Diffusion-Patch\` or `Hyper-YOLO\`.

---

## E. Recommended mitigation subset design

### Proposed design — **scientifically reasonable**

| Mitigation | Source attack | Success cases | Controls | Approx. total |
|------------|---------------|---------------|----------|---------------|
| mitigation_01 | attack_02 | all **15** successes | **15** eligible non-success (attacked still detected) | ~30 |
| mitigation_02 | attack_05 | all **20** successes | **20** eligible non-success | ~40 |

### Why this is defensible

- Focuses compute on the cases where the attack actually suppressed the target (primary recovery question).
- Adds **controls** to measure whether restoration harms targets that were still detected after attack (preservation / over-restoration risk).
- Uses **observed strongest normalized effects** (attack_02 / attack_05) without claiming those patches are universally stronger (different random samples).

### Selection rules (exact)

**Success set (mandatory include-all):**

```text
eligible_for_asr = Yes
AND clean_detected = Yes
AND attacked_detected = No
AND attack_success = Yes
```

**Control pool:**

```text
eligible_for_asr = Yes
AND attack_success = No
AND attacked_detected = Yes
```

**Control sampling:**

- Deterministic `random.Random(mitigation_seed)` (save seed).
- Aim for class balance among controls (≈ half person / half car) when pool sizes allow (pools are large enough).
- Do **not** require class balance among successes (only 4 person successes in each of attack_02 and attack_05).

### Reporting honesty

- Call rates **subset recovery / subset preservation**, not full-dataset “ASR after defense.”
- Always report sample sizes and person/car breakdowns.
- Note attack_02 / attack_05 used different seeds/images; mitigation_01 vs mitigation_02 compare **restoration effectiveness on those subsets**, not absolute patch strength.

---

## F. Recommended restoration algorithm (step-by-step)

**Name for papers/slides:**  
*Simplified known-location diffusion-based restoration* / *known-location diffusion restoration baseline*

**Critical validity rule:** restoration input = **attacked image** + known patch location. Clean pixels may be used **only** for evaluation reference, never pasted into the restored region.

1. Load subset row from `attack_results.csv` + GT from `selected_images.csv`.
2. Copy attacked image from `attacks\attack_XX\patched_images\<filename>` into `mitigations\mitigation_YY\source_attacked_images\` (copy only).
3. Reconstruct `patch_x, patch_y, patch_size` with the frozen geometry formula; verify against `patch_size_pixels` / centers.
4. Save documentation mask (`masks\<stem>_patch_mask.png`) covering the exact patch square.
5. Build a **context crop** around the patch:
   - Recommended: square crop centered on patch center with side length  
     `max(128, min(image_min_side, int(round(3.0 * patch_size))))`  
     then clamp to image bounds (adjust if near edge).
6. Resize crop to **512×512** for Img2Img.
7. Run `StableDiffusionImg2ImgPipeline` with fixed prompt/negative/params/seed (section G).
8. Resize restored 512 crop back to original crop size.
9. Composite **only the patch square** (or patch square with small feathered alpha, e.g. 3–5 px) into a full-resolution copy of the attacked image. Outside the patch region must remain identical to the attacked image.
10. Save `restored_images\<filename>`.
11. Run Hyper-YOLO-N with frozen settings (`conf=0.25`, `iou=0.7`, `imgsz=640`, `device=cpu`) on restored images only; write predictions under `mitigations\...\results\mitigated_predictions\`.
12. Match target with **same class + IoU ≥ 0.50**.
13. Build verification / CSV / summary / global mitigation comparison.

Use the **same** restoration configuration for mitigation_01 and mitigation_02.

---

## G. Recommended fixed Img2Img parameters

| Parameter | Recommended fixed value | Rationale |
|-----------|-------------------------|-----------|
| Diffusion crop resolution | 512×512 | Native SD 2.1 operating size |
| `num_inference_steps` | **30** | Reasonable quality/time on RTX 2050 |
| `guidance_scale` | **7.5** | Standard SD guidance |
| `strength` | **0.55** | Mid-band: enough to weaken sticker texture; lower risk than 0.60–0.70 of destroying object geometry |
| Context crop | **~3× patch_size**, min side 128 px, clamped | Provides surrounding texture without restoring whole image |
| Feather | **3–5 px** soft alpha at patch edge | Reduces hard seams; still local |
| Generator seed | Fixed per mitigation run (save it) | Reproducibility |
| `local_files_only` | **True** | No download |
| Device | CUDA if available, else CPU | Inspected CUDA True |

**Prompt (generic, fixed):**

```text
realistic natural photograph, restore the original object surface, natural texture, consistent lighting, no sticker or artificial patch
```

**Negative prompt (fixed):**

```text
sticker, adversarial patch, logo, text, watermark, abstract pattern, cartoon, distortion
```

**Notes:**

- Do not specialize prompts per class unless a controlled ablation is added later.
- If early visual checks show object damage, lower `strength` to **0.50** (still keep one fixed value for both mitigations).
- If sticker residue remains, raise to **0.60** once, then freeze.

---

## H. Recommended metrics and exact formulas

Let:

- \(S\) = set of originally successful attack cases in the mitigation subset  
- \(C\) = set of control cases in the mitigation subset  
- For image \(i\):  
  - \(D^{mit}_i\) = mitigated target detection (Yes/No under class + IoU≥0.50)  
  - \(c^{cl}_i, c^{at}_i, c^{mi}_i\) = clean / attacked / mitigated confidences (0 if not detected)  
  - \(u^{cl}_i, u^{at}_i, u^{mi}_i\) = clean / attacked / mitigated IoUs (0 if not detected)

### Core counts / rates

1. **source_attack_success_count**  
   \(= |S|\)

2. **recovered_target_count**  
   \(= |\{ i \in S : D^{mit}_i = \text{Yes} \}|\)

3. **detection_recovery_rate**  
   \(= \dfrac{\text{recovered_target_count}}{\text{source_attack_success_count}} \times 100\%\)  
   (define as 0% if \(|S|=0\))

4. **remaining_failed_targets**  
   \(= |S| - \text{recovered_target_count}\)

5. **control_preserved_count**  
   \(= |\{ i \in C : D^{mit}_i = \text{Yes} \}|\)

6. **control_preservation_rate**  
   \(= \dfrac{\text{control_preserved_count}}{|C|} \times 100\%\)

### Confidence / IoU (report means over \(S\) and over \(C\) separately)

7. **clean_confidence** — \(c^{cl}_i\) (and mean over set)  
8. **attacked_confidence** — \(c^{at}_i\)  
9. **mitigated_confidence** — \(c^{mi}_i\)  

10. **confidence_recovery** (per success image; then mean over recovered or over all \(S\) — report both if helpful)  
    \(= c^{mi}_i - c^{at}_i\)

11–13. **clean_iou / attacked_iou / mitigated_iou** — analogous  

14. **iou_recovery**  
    \(= u^{mi}_i - u^{at}_i\)

15. **clean-to-mitigated confidence gap**  
    \(= c^{cl}_i - c^{mi}_i\)

16. **clean-to-mitigated IoU gap**  
    \(= u^{cl}_i - u^{mi}_i\)

17. **person_recovery_rate**  
    recovery rate restricted to \(S_{person}\)

18. **car_recovery_rate**  
    recovery rate restricted to \(S_{car}\)

### Additional useful (keep small)

19. **control_regression_count**  
    controls where attacked was Yes but mitigated is No  
    \(= |C| - \text{control_preserved_count}\)

20. **control_regression_rate**  
    \(= \dfrac{\text{control_regression_count}}{|C|} \times 100\%\)

### Naming caution

Do **not** call subset recovery “ASR after defense” unless the denominator is the full ASR-eligible set of the original attack. Prefer:

- `detection_recovery_rate` (on success subset)  
- `control_preservation_rate` (on controls)

---

## I. Recommended final folder structure

Proposed structure is **appropriate**. Keep attacks frozen; add parallel `mitigations\`.

```text
D:\project CS\Adversarial-Patch-Experiment\
|
+-- attacks\                    # READ-ONLY (frozen)
|   +-- attack_01\ ... attack_05\
|
+-- mitigations\                # NEW (create only when implementing)
|   +-- mitigation_01\          # source: attack_02
|   |   +-- source_attacked_images\
|   |   +-- masks\
|   |   +-- restored_images\
|   |   +-- results\
|   |       +-- mitigated_predictions\images\
|   |       +-- mitigated_predictions\labels\
|   |       +-- verification\
|   |       +-- comparison\
|   |       +-- selected_subset.csv
|   |       +-- mitigation_results.csv
|   |       +-- mitigation_summary.txt
|   |       +-- mitigation_overview.jpg
|   |       +-- restoration_config.json
|   +-- mitigation_02\          # source: attack_05
|       +-- (same structure)
|
+-- scripts\                    # add mitigation scripts only
|
+-- results\
    +-- all_attacks_comparison.csv / .png     # DO NOT MODIFY
    +-- all_mitigations_comparison.csv / .png # NEW later
```

Also save: subset seed, Img2Img params, prompts, and source attack ID in `restoration_config.json`.

---

## J. Exact new scripts that would need to be created later

Prefer **new** scripts under `Adversarial-Patch-Experiment\scripts\`. Do not change attack pipeline behavior.

| New script | Role |
|------------|------|
| `mitigation_config.py` | Fixed Img2Img params, prompts, seeds, crop factor, feather, mappings mitigation→attack |
| `select_mitigation_subset.py` | Build deterministic success+control subset CSVs from frozen attack results |
| `restore_patch_region.py` | SD-venv Img2Img worker: crop → restore → composite → write restored image/mask |
| `run_mitigation_pipeline.py` | Orchestrator (Hyper-YOLO venv): subset → restore subprocess → YOLO → match → reports |
| `mitigation_reporting.py` | Metrics, CSVs, plots, overview, global `all_mitigations_comparison.*` |

**Reuse (read-only import / copy of formulas):**  
IoU matching (`TARGET_MATCH_IOU_THRESHOLD=0.50`), YOLO predict helpers, verification drawing patterns from existing modules — without altering attack outputs or attack scripts’ behavior.

**Subprocess split:**  
`run_mitigation_pipeline.py` (Hyper-YOLO Python) → calls `restore_patch_region.py` (SD Python).

---

## K. Technical / scientific risks and limitations

1. **Known-location assumption:** defender is assumed to know the patch square. This is a **baseline**, not a blind detection+removal defense.
2. **Not dedicated inpainting:** Img2Img may alter non-patch pixels inside the context crop; compositing must restrict visible change to the patch region.
3. **Strength trade-off:** too low → sticker remains; too high → object texture/geometry damaged → recovery fails or controls regress.
4. **Success class imbalance:** mostly cars among successes; person recovery rates will have high variance / small \(n\).
5. **Different attack samples:** mitigation_01 vs mitigation_02 are not a pure patch-vs-patch bake-off.
6. **Subset metrics ≠ dataset ASR:** do not over-claim defense success rate.
7. **GPU memory:** RTX 2050 is modest; 512 Img2Img with fp16 should be planned; fall back to CPU if OOM (much slower).
8. **Seam / feathering artifacts** may remain even after restoration.
9. **Safety wording:** keep “simplified known-location diffusion restoration baseline” terminology.
10. **Frozen attacks:** any future bugfix must not rewrite attack_01–05 artifacts.

---

## L. Confirmation of safety constraints (this analysis step)

| Constraint | Confirmed |
|------------|-----------|
| attack_01 through attack_05 untouched | YES |
| attack_06 not created | YES |
| mitigation folders not created | YES |
| no model downloaded | YES |
| no Hyper-YOLO inference run for mitigation | YES |
| no Stable Diffusion generation/restoration run | YES |
| Hyper-YOLO / Stable-Diffusion-Patch not modified | YES |
| Existing attack CSVs / summaries / plots not updated | YES |
| Only new file intended: this report | YES (`mitigation_report.md`) |

---

## Summary recommendation

Proceed later (after your review) with:

- **mitigation_01 ← attack_02** (15 successes + 15 controls)  
- **mitigation_02 ← attack_05** (20 successes + 20 controls)  
- **Known-location Img2Img restoration** via existing SD 2.1 Base + `StableDiffusionImg2ImgPipeline`  
- **Fixed params** (strength 0.55, steps 30, guidance 7.5, 512 crop, ~3× context)  
- **Subset recovery / preservation metrics** (not full ASR-after-defense)  
- **Parallel `mitigations\` tree**; attacks remain read-only  

---

**ANALYSIS COMPLETE — SAFE TO REVIEW BEFORE IMPLEMENTATION**
