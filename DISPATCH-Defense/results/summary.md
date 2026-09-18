# DISPATCH Defense - Detailed Results Summary

**Project root:** `D:\project CS\DISPATCH-Defense\`  
**This file:** `D:\project CS\DISPATCH-Defense\results\summary.md`  
**Purpose:** Consolidate measured Automatic DISPATCH and Known-Location diagnostic results (including Teacher Notebook outputs) in one place.

**Important metric rule**

| Result type | What it is |
| --- | --- |
| Official Detection Recovery Rate (DRR) | Only `mitigations/mitigation_01` (30-image Automatic DISPATCH) |
| Diagnostic subset recovery | Testing mitigations + Teacher Notebook — **NOT** final DISPATCH DRR, **NOT** paper mAP@0.5 |

All numbers below are copied verbatim from existing result artifacts. No new defense runs were performed for this summary.

---

## 1. Official Automatic DISPATCH (`mitigation_01`)

| Field | Value |
| --- | --- |
| Mitigation ID | `mitigation_01` |
| Source attack | Hyper-YOLO `attack_02` |
| Total selected images | 30 |
| Successfully processed | 30 |
| Failed images | 0 |
| Attack-success cases | 15 |
| Control cases | 15 |
| **Recovered targets** | **2** |
| **Detection Recovery Rate (DRR)** | **13.33%** (2/15) |
| Remaining failed targets | 13 |
| Control preserved | 13 |
| **Control Preservation Rate (CPR)** | **86.67%** (13/15) |
| Control regression | 2 (13.33%) |
| Person recovery | 0/4 (0.00%) |
| Car recovery | 2/11 (18.18%) |
| Mean confidence recovery | −0.098558 |
| Mean IoU recovery | −0.017907 |
| Mean adversarial-mask IoU | 0.069325 |
| Mask precision | 0.090624 |
| Mask recall | 0.775688 |
| Mean processing time | 114.24 sec |
| Diffusion | CompVis LDM `inpainting_big` |
| Sampler / steps | DDIM / 5 |
| Grid N | 32 |
| Seed | 20260817 |
| `clean_pixels_used_for_restoration` | **false** |
| `known_patch_location_used_by_defense` | **false** |

**Path:** `D:\project CS\DISPATCH-Defense\mitigations\mitigation_01\`  
**Source files:**

- `D:\project CS\DISPATCH-Defense\mitigations\mitigation_01\results\summary.txt`
- `D:\project CS\DISPATCH-Defense\mitigations\mitigation_01\results\summary.json`

---

## 2. Automatic location — method and results

### Method

1. Checkerboard regeneration over the attacked image  
2. L2 difference map  
3. Smoothing  
4. KMeans clustering (`k=2`) to form the predicted suspicious mask  
5. LDM inpainting on the **attacked / patched** image using that mask  

**True patch coordinates are NOT supplied** to Automatic localization.  
Clean pixels are **not** composited into the restored image.

### Measured Automatic results

| Experiment | Attack-success recovered | Mask IoU (mean) | Controls preserved | Notes |
| --- | ---: | ---: | ---: | --- |
| Official `mitigation_01` (30 img) | **2 / 15** → DRR **13.33%** | 0.069325 | 13 / 15 | Official Automatic DISPATCH |
| Testing M01 branch (`automatic/results/summary.txt`) | **1 / 4** | mean 0.055087 (branch) | YES | Diagnostic |
| Testing M01 combined (`testing_mitigations/mitigation_01/results/summary.txt`) | **2 / 4** | mean 0.049885 | YES | Diagnostic (combined write-up) |
| Testing M02 Automatic | **1 / 8** (person 0/4, car 1/4) | **0.085892** (median 0.040831; min/max 0.003886 / 0.409134) | **2 / 2** | Diagnostic |
| Teacher Notebook comparison | **0 / 4** | **0.0494** (precision 0.0681, recall 0.7942) | **1 / 1** | Diagnostic |

> When Testing M01 branch vs combined counts differ, both are reported with their source labeled. Prefer branch files for per-branch provenance; prefer comparison CSV for side-by-side.

### Testing M02 — Automatic per-image results

Source: `testing_mitigations/mitigation_02/automatic/results/summary.txt`

| Image | Class | Case | Patch | Mask IoU | Restored | Conf | IoU | Recovered |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: | --- |
| 000000127263.jpg | car | attack_success | 20 | 0.0177 | True | 0.3057 | 0.7512 | **True** |
| 000000393093.jpg | car | attack_success | 14 | 0.0039 | False | — | — | False |
| 000000026926.jpg | car | attack_success | 22 | 0.0236 | False | — | — | False |
| 000000407083.jpg | car | attack_success | 144 | 0.1516 | False | — | — | False |
| 000000181499.jpg | person | attack_success | 36 | 0.0569 | False | — | — | False |
| 000000347370.jpg | person | attack_success | 28 | 0.0247 | False | — | — | False |
| 000000385029.jpg | person | attack_success | 51 | 0.0796 | False | — | — | False |
| 000000542776.jpg | person | attack_success | 67 | 0.4091 | False | — | — | False |
| 000000331604.jpg | person | control | 21 | 0.0702 | True | 0.2961 | 0.8191 | False |
| 000000357737.jpg | car | control | 43 | 0.0215 | True | 0.8770 | 0.8828 | False |

### Automatic paths

| Role | Absolute path |
| --- | --- |
| Official mitigation | `D:\project CS\DISPATCH-Defense\mitigations\mitigation_01\` |
| Testing M01 automatic | `D:\project CS\DISPATCH-Defense\testing_mitigations\mitigation_01\automatic\` |
| Testing M02 automatic | `D:\project CS\DISPATCH-Defense\testing_mitigations\mitigation_02\automatic\` |
| Teacher Notebook automatic | `D:\project CS\DISPATCH-Teacher-Notebook\automatic\` |
| Teacher auto summary | `D:\project CS\DISPATCH-Teacher-Notebook\automatic\results\summary.txt` |
| Predicted masks (M02) | `...\mitigation_02\automatic\masks\predicted_masks\` |
| Final auto restored (M02) | `...\mitigation_02\automatic\restored_images\` |

---

## 3. Known location — method and results

### Method

- **True saved patch coordinates ARE supplied** (oracle / known-location branch).  
- Automatic L2/KMeans localization is **bypassed**.  
- Two mask variants: **Exact** and **Expanded +4px**.  
- LDM input = attacked image + binary mask only.  
- Clean pixels never composited. Outside-mask integrity expected MAE ≈ 0.

### Measured Known-Location results

| Experiment | Exact recovered | +4px recovered | Controls | Outside-mask integrity |
| --- | ---: | ---: | ---: | --- |
| Testing M01 Known | **1 / 4** | **3 / 4** | Exact YES / +4 YES | PASS |
| Testing M02 Known | **3 / 8** (person 2/4, car 1/4) | **6 / 8** (person 3/4, car 3/4) | Exact 2/2 · +4 2/2 | **PASS** |
| Teacher Notebook Known | **2 / 4** | **2 / 4** | Exact 1/1 · +4 1/1 | mean outside MAE exact=0, +4=0 |

### Testing M02 — Known per-image results

Source: `testing_mitigations/mitigation_02/known_location/results/summary.txt`

| Image | Class | Patch | Exact det | Exact conf | Exact IoU | Exact rec | +4 det | +4 conf | +4 IoU | +4 rec |
| --- | --- | ---: | --- | ---: | ---: | --- | --- | ---: | ---: | --- |
| 000000127263.jpg | car | 20 | False | — | — | False | False | — | — | False |
| 000000393093.jpg | car | 14 | True | 0.5273 | 0.9289 | **True** | True | 0.5527 | 0.9388 | **True** |
| 000000026926.jpg | car | 22 | False | — | — | False | True | 0.2937 | 0.7453 | **True** |
| 000000407083.jpg | car | 144 | False | — | — | False | True | 0.3955 | 0.8321 | **True** |
| 000000181499.jpg | person | 36 | True | 0.3464 | 0.9242 | **True** | True | 0.3911 | 0.9204 | **True** |
| 000000347370.jpg | person | 28 | False | — | — | False | False | — | — | False |
| 000000385029.jpg | person | 51 | False | — | — | False | True | 0.3472 | 0.9933 | **True** |
| 000000542776.jpg | person | 67 | True | 0.2842 | 0.9795 | **True** | True | 0.4695 | 0.9771 | **True** |
| 000000331604.jpg | person | 21 | True | 0.7808 | 0.8500 | False (control) | True | 0.6792 | 0.6648 | False |
| 000000357737.jpg | car | 43 | True | 0.8755 | 0.9314 | False (control) | True | 0.8784 | 0.9061 | False |

### Known-Location paths

| Role | Absolute path |
| --- | --- |
| Testing M01 known | `D:\project CS\DISPATCH-Defense\testing_mitigations\mitigation_01\known_location\` |
| Testing M02 known | `D:\project CS\DISPATCH-Defense\testing_mitigations\mitigation_02\known_location\` |
| Teacher Notebook known | `D:\project CS\DISPATCH-Teacher-Notebook\known_location\` |
| Exact masks (M02) | `...\known_location\masks\exact_mask\` |
| +4px masks (M02) | `...\known_location\masks\expanded_4px\` |
| Final exact restored (M02) | `...\known_location\restored_images\exact_mask\` |
| Final +4 restored (M02) | `...\known_location\restored_images\expanded_4px\` |

---

## 4. Automatic vs Known comparison

### Headline comparison

| Experiment | Automatic | Known Exact | Known +4px | Known improved over Automatic? |
| --- | ---: | ---: | ---: | --- |
| Testing M01 (comparison summary) | **1 / 4** | **1 / 4** | **3 / 4** | **MIXED** |
| Testing M01 combined write-up | **2 / 4** | **2 / 4** | **2 / 4** | MIXED |
| Testing M02 | **1 / 8** | **3 / 8** | **6 / 8** | **YES** |
| Teacher Notebook | **0 / 4** | **2 / 4** | **2 / 4** | YES (count) |

**Evidence-supported primary bottleneck (M02):** **MIXED** — known location recovers additional cases (localization helps), but several cases still fail even with oracle / +4 masks (LDM restoration also limits recovery).

### Testing M02 — per-image Automatic vs Known

Source: `testing_mitigations/mitigation_02/known_location/results/comparison/summary.txt`

| Image | Auto rec | Exact rec | +4 rec | Auto mask IoU | Interpretation |
| --- | --- | --- | --- | ---: | --- |
| 000000127263.jpg | True | False | False | 0.017694 | Automatic recovered but known did not — mixed / incidental auto regeneration |
| 000000393093.jpg | False | True | True | 0.003886 | Case A: auto fail / known succeed — localization likely bottleneck |
| 000000026926.jpg | False | False | True | 0.023633 | Case C: both fail exact — localization alone does not explain failure |
| 000000407083.jpg | False | False | True | 0.151611 | Case C: both fail exact — localization alone does not explain failure |
| 000000181499.jpg | False | True | True | 0.056930 | Case A: auto fail / known succeed — localization likely bottleneck |
| 000000347370.jpg | False | False | False | 0.024733 | Case C: both fail — localization alone does not explain failure |
| 000000385029.jpg | False | False | True | 0.079622 | Case C: both fail exact — localization alone does not explain failure |
| 000000542776.jpg | False | True | True | 0.409134 | Case A: auto fail / known succeed — localization likely bottleneck |
| 000000331604.jpg | False | False | False | 0.070167 | control |
| 000000357737.jpg | False | False | False | 0.021506 | control |

`manual_visual_review_required = true`  
Source copies byte-equivalent across Automatic/Known branches: **YES**

### Comparison CSV paths

- `D:\project CS\DISPATCH-Defense\testing_mitigations\mitigation_01\known_location\results\comparison\automatic_vs_known.csv`
- `D:\project CS\DISPATCH-Defense\testing_mitigations\mitigation_02\known_location\results\comparison\automatic_vs_known.csv`
- `D:\project CS\DISPATCH-Defense\testing_mitigations\mitigation_02\automatic\results\comparison\automatic_vs_known.csv`
- `D:\project CS\DISPATCH-Teacher-Notebook\comparison\automatic_vs_known.csv`
- `D:\project CS\DISPATCH-Teacher-Notebook\comparison\summary.txt`

---

## 5. Teacher Notebooks — locations and diagnostic outputs

| Item | Absolute path |
| --- | --- |
| Main teacher notebook | `D:\project CS\DISPATCH-Teacher-Notebook\dispatch_defense_experiment.ipynb` |
| Test teacher notebook | `D:\project CS\DISPATCH-Teacher-Notebook\dispatch_defense_experiment_test.ipynb` |
| Automatic results | `D:\project CS\DISPATCH-Teacher-Notebook\automatic\results\summary.txt` |
| Known-location results | `D:\project CS\DISPATCH-Teacher-Notebook\known_location\results\summary.txt` |
| Comparison summary | `D:\project CS\DISPATCH-Teacher-Notebook\comparison\summary.txt` |

**Teacher Notebook comparison summary (verbatim):**

```
DIAGNOSTIC SUBSET RECOVERY RATE (5-image notebook — NOT DISPATCH DRR, NOT paper mAP)
Automatic recovered: 0 / 4
Known Exact recovered: 2 / 4
Known +4px recovered: 2 / 4
Automatic mean mask IoU: 0.0494
Automatic controls preserved: 1 / 1
Known exact controls preserved: 1 / 1
Known +4 controls preserved: 1 / 1
```

Automatic notebook settings note: `N=32 steps=5 Gaussian k=15 sigma=1.0 MORPH_OPEN=3x3`.

---

## 6. Full folder / location map

### DISPATCH-Defense

| Path | Contents |
| --- | --- |
| `D:\project CS\DISPATCH-Defense\` | Defense project root |
| `D:\project CS\DISPATCH-Defense\results\summary.md` | **This consolidated summary** |
| `D:\project CS\DISPATCH-Defense\mitigations\mitigation_01\` | Official Automatic DISPATCH (30 images) |
| `D:\project CS\DISPATCH-Defense\testing_mitigations\mitigation_01\` | 5-image Automatic vs Known diagnostic |
| `D:\project CS\DISPATCH-Defense\testing_mitigations\mitigation_02\automatic\` | 10-image Automatic branch |
| `D:\project CS\DISPATCH-Defense\testing_mitigations\mitigation_02\known_location\` | 10-image Known Exact / +4px branch |

### Typical artifact folders (both Automatic and Known trees)

| Folder | Meaning |
| --- | --- |
| `source_clean_images\` | Clean evaluation references only |
| `source_attacked_images\` | Actual LDM inputs (patched images) |
| `masks\predicted_masks\` | Automatic localized masks |
| `masks\exact_mask\` | Known exact true patch masks |
| `masks\expanded_4px\` | Known location expanded by 4 px |
| `regenerated_images\` / `ldm_outputs\` | Intermediate LDM output — **not final** |
| `restored_images\` | **Final** restored images used for YOLO evaluation |
| `results\restored_predictions\` | Hyper-YOLO predictions on restored images |

### Shared settings across diagnostics

| Setting | Value |
| --- | --- |
| Victim detector | Hyper-YOLO-N |
| Confidence | 0.25 |
| Match IoU (diagnostic) | typically 0.50–0.70 per experiment write-up |
| Diffusion | CompVis LDM inpainting |
| Sampler | DDIM |
| Steps | 5 |
| Seed (Defense runs) | 20260817 |
| LDM input | Attacked image only (+ mask for known branch) |
| Clean pixels for restoration | **NO** |

---

## 7. Safety / protocol notes

- Hyper-YOLO / Adversarial-Patch-Experiment projects were **not** modified for these defenses (read-only sources).  
- Official Automatic DISPATCH does **not** receive true patch coordinates.  
- Known-location branches are **oracle diagnostics** to separate localization failure from LDM failure.  
- Diagnostic recoveries must **not** be reported as final paper DRR.  
- Only official DRR from this tree: **13.33%** on `mitigation_01` (2 recovered / 15 attack-success cases).

---

## 8. One-page numeric snapshot

| Layer | Automatic | Known Exact | Known +4px | Official DRR? |
| --- | ---: | ---: | ---: | --- |
| Official mitigation_01 | 2/15 (**13.33%**) | — | — | **YES** |
| Testing M02 (best auto-vs-known diagnostic) | 1/8 | 3/8 | 6/8 | No |
| Testing M01 comparison | 1/4 | 1/4 | 3/4 | No |
| Teacher Notebook | 0/4 | 2/4 | 2/4 | No |

**Bottom line (measured):** Automatic DISPATCH on the 30-image official run recovers **13.33%** of attack-success targets. On diagnostic subsets, supplying the true patch location (especially with a +4 px expansion) recovers more cases than Automatic localization alone, but recovery remains incomplete — evidence supports a **mixed** bottleneck of localization quality and LDM restoration capability.
