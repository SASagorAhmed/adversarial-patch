# Project CS — Command Cheat Sheet

Workspace root: `D:\project CS`

Use this file to copy-paste everyday commands for **Hyper-YOLO**, **Stable Diffusion**, and the **Adversarial Patch Experiment**.

---

## Quick start — next attack (most used)

After existing attacks finish, the next run creates the next `attack_XX` automatically (ID, patch, image count).

### PowerShell

```powershell
cd "D:\project CS\Adversarial-Patch-Experiment"
& "D:\project CS\Hyper-YOLO\.venv\Scripts\python.exe" scripts\run_attack_pipeline.py
```

### Git Bash

```bash
cd "/d/project CS/Adversarial-Patch-Experiment"
"/d/project CS/Hyper-YOLO/.venv/Scripts/python.exe" scripts/run_attack_pipeline.py
```

---

## 1. Paths and Python interpreters

| Role | Path |
|------|------|
| Workspace | `D:\project CS` |
| Hyper-YOLO project | `D:\project CS\Hyper-YOLO` |
| Stable Diffusion project | `D:\project CS\Stable-Diffusion-Patch` |
| Adversarial experiment | `D:\project CS\Adversarial-Patch-Experiment` |
| Hyper-YOLO Python | `D:\project CS\Hyper-YOLO\.venv\Scripts\python.exe` |
| Stable Diffusion Python | `D:\project CS\Stable-Diffusion-Patch\.venv\Scripts\python.exe` |
| Hyper-YOLO weights | `D:\project CS\Hyper-YOLO\weights\hyper-yolon.pt` |
| SD local model | `D:\project CS\Stable-Diffusion-Patch\model` |

**Important:** Launch the adversarial pipeline with the **Hyper-YOLO** Python. If a candidate patch is missing, the pipeline calls the **Stable Diffusion** Python as a subprocess. Do not require `diffusers` inside the Hyper-YOLO venv.

Treat these as **read-only** during adversarial experiments:

- `D:\project CS\Hyper-YOLO\`
- `D:\project CS\Stable-Diffusion-Patch\`

---

## 2. Shell note (PowerShell vs Git Bash)

| Shell | How to run a Python path |
|-------|---------------------------|
| **PowerShell** | `& "D:\path\to\python.exe" script.py` |
| **Git Bash** | `"/d/project CS/Hyper-YOLO/.venv/Scripts/python.exe" script.py` |

Do **not** use PowerShell `&` inside Git Bash — you will get `syntax error near unexpected token '&'`.

In Git Bash, prefer `/d/project CS/...` paths (forward slashes).

---

## 3. Hyper-YOLO

### Settings used by the adversarial pipeline

| Setting | Value |
|---------|-------|
| Model | `weights\hyper-yolon.pt` |
| conf | `0.25` |
| iou | `0.7` |
| imgsz | `640` |
| device | `cpu` |
| Target-match IoU (experiment) | `0.50` |

### Check / use Hyper-YOLO Python

**PowerShell**

```powershell
& "D:\project CS\Hyper-YOLO\.venv\Scripts\python.exe" --version
```

**Git Bash**

```bash
"/d/project CS/Hyper-YOLO/.venv/Scripts/python.exe" --version
```

### Analyze previous val_500 prediction labels (inside Hyper-YOLO)

Runs against `Hyper-YOLO\runs\val_500\labels` and writes `Hyper-YOLO\runs\analysis_500\`.

**PowerShell**

```powershell
cd "D:\project CS\Hyper-YOLO"
& "D:\project CS\Hyper-YOLO\.venv\Scripts\python.exe" analyze_results.py
```

**Git Bash**

```bash
cd "/d/project CS/Hyper-YOLO"
"/d/project CS/Hyper-YOLO/.venv/Scripts/python.exe" analyze_results.py
```

For adversarial attacks, prefer the automatic pipeline in section 5 (it runs Hyper-YOLO itself and keeps outputs under `attacks\attack_XX\`).

---

## 4. Stable Diffusion (candidate patches)

### Model and output locations

| Item | Path |
|------|------|
| Local SD 2.1 model | `D:\project CS\Stable-Diffusion-Patch\model` |
| Generated patches | `D:\project CS\Adversarial-Patch-Experiment\diffusion_patches\` |
| Generator script | `Adversarial-Patch-Experiment\scripts\generate_candidate_patch.py` |

Normal attacks **auto-generate** a missing patch. Manual generation is optional.

### Patch schedule (attack 01–10)

| Attack | Patch file | Seed |
|--------|------------|------|
| attack_01 | `diffusion_patch_01.png` | 101 |
| attack_02 | `diffusion_patch_02.png` | 102 |
| attack_03 | `diffusion_patch_03.png` | 103 |
| attack_04 | `diffusion_patch_04.png` | 104 |
| attack_05 | `diffusion_patch_05.png` | 105 |
| attack_06 | `diffusion_patch_06.png` | 106 |
| attack_07 | `diffusion_patch_07.png` | 107 |
| attack_08 | `diffusion_patch_08.png` | 108 |
| attack_09 | `diffusion_patch_09.png` | 109 |
| attack_10 | `diffusion_patch_10.png` | 110 |

Prompts live in `Adversarial-Patch-Experiment\scripts\patch_config.py`.

### Manual generate one patch (example: patch 06)

Must use the **Stable Diffusion** Python.

**PowerShell**

```powershell
cd "D:\project CS\Adversarial-Patch-Experiment"
& "D:\project CS\Stable-Diffusion-Patch\.venv\Scripts\python.exe" `
  scripts\generate_candidate_patch.py `
  --output "D:\project CS\Adversarial-Patch-Experiment\diffusion_patches\diffusion_patch_06.png" `
  --seed 106 `
  --prompt "A colorful abstract square sticker with fractal-like shapes, intense contrast, realistic printed texture, centered composition" `
  --model-path "D:\project CS\Stable-Diffusion-Patch\model"
```

**Git Bash**

```bash
cd "/d/project CS/Adversarial-Patch-Experiment"
"/d/project CS/Stable-Diffusion-Patch/.venv/Scripts/python.exe" \
  scripts/generate_candidate_patch.py \
  --output "/d/project CS/Adversarial-Patch-Experiment/diffusion_patches/diffusion_patch_06.png" \
  --seed 106 \
  --prompt "A colorful abstract square sticker with fractal-like shapes, intense contrast, realistic printed texture, centered composition" \
  --model-path "/d/project CS/Stable-Diffusion-Patch/model"
```

Generator args: `--output`, `--seed`, `--prompt`, optional `--model-path`.

---

## 5. Adversarial-Patch-Experiment (main workflow)

Working directory for these commands:

```text
D:\project CS\Adversarial-Patch-Experiment
```

Always launch with **Hyper-YOLO** Python.

### 5.1 Validate only (no attack folder created)

**PowerShell**

```powershell
cd "D:\project CS\Adversarial-Patch-Experiment"
& "D:\project CS\Hyper-YOLO\.venv\Scripts\python.exe" scripts\run_attack_pipeline.py --validate-only
```

**Git Bash**

```bash
cd "/d/project CS/Adversarial-Patch-Experiment"
"/d/project CS/Hyper-YOLO/.venv/Scripts/python.exe" scripts/run_attack_pipeline.py --validate-only
```

### 5.2 Run next attack (automatic)

Determines next attack ID, diffusion patch, seed/prompt, image count, and random selection seed.

**PowerShell**

```powershell
cd "D:\project CS\Adversarial-Patch-Experiment"
& "D:\project CS\Hyper-YOLO\.venv\Scripts\python.exe" scripts\run_attack_pipeline.py
```

**Git Bash**

```bash
cd "/d/project CS/Adversarial-Patch-Experiment"
"/d/project CS/Hyper-YOLO/.venv/Scripts/python.exe" scripts/run_attack_pipeline.py
```

### 5.3 Automatic image-count schedule

Formula: `num_images = 200 + ((attack_number - 1) * 50)`

| Attack | Default images | Default patch |
|--------|----------------|---------------|
| attack_01 | 200 | `diffusion_patch_01.png` |
| attack_02 | 250 | `diffusion_patch_02.png` |
| attack_03 | 300 | `diffusion_patch_03.png` |
| attack_04 | 350 | `diffusion_patch_04.png` |
| attack_05 | 400 | `diffusion_patch_05.png` |
| attack_06 | 450 | `diffusion_patch_06.png` |
| attack_07 | 500 | `diffusion_patch_07.png` |
| attack_08 | 550 | `diffusion_patch_08.png` |
| attack_09 | 600 | `diffusion_patch_09.png` |
| attack_10 | 650 | `diffusion_patch_10.png` |

Balanced person/car sampling when the count is even (e.g. 400 → 200 + 200).

Target matching: **same class AND IoU ≥ 0.50**. ASR denominator = valid clean detections for that attack only.

### 5.4 Optional overrides

**PowerShell**

```powershell
# Override image count
& "D:\project CS\Hyper-YOLO\.venv\Scripts\python.exe" scripts\run_attack_pipeline.py --num-images 300

# Specific patch
& "D:\project CS\Hyper-YOLO\.venv\Scripts\python.exe" scripts\run_attack_pipeline.py --patch diffusion_patch_05.png

# Reproducible image selection
& "D:\project CS\Hyper-YOLO\.venv\Scripts\python.exe" scripts\run_attack_pipeline.py --seed 12345

# Combine
& "D:\project CS\Hyper-YOLO\.venv\Scripts\python.exe" scripts\run_attack_pipeline.py --num-images 400 --seed 12345 --patch diffusion_patch_05.png
```

**Git Bash**

```bash
"/d/project CS/Hyper-YOLO/.venv/Scripts/python.exe" scripts/run_attack_pipeline.py --num-images 300
"/d/project CS/Hyper-YOLO/.venv/Scripts/python.exe" scripts/run_attack_pipeline.py --patch diffusion_patch_05.png
"/d/project CS/Hyper-YOLO/.venv/Scripts/python.exe" scripts/run_attack_pipeline.py --seed 12345
```

### 5.5 Recalculate evaluation (no YOLO / no SD rerun)

Uses saved prediction labels only. Does not create a new attack.

**PowerShell**

```powershell
cd "D:\project CS\Adversarial-Patch-Experiment"
& "D:\project CS\Hyper-YOLO\.venv\Scripts\python.exe" scripts\recalculate_attack_evaluation.py --attack-id attack_01
```

**Git Bash**

```bash
cd "/d/project CS/Adversarial-Patch-Experiment"
"/d/project CS/Hyper-YOLO/.venv/Scripts/python.exe" scripts/recalculate_attack_evaluation.py --attack-id attack_01
```

### 5.6 Older / optional helpers

Usually not needed once `run_attack_pipeline.py` is used.

| Script | Purpose |
|--------|---------|
| `scripts\select_coco_images.py` | Early fixed 20-image selection into `clean_images\` |
| `scripts\run_clean_baseline.py` | Early clean Hyper-YOLO baseline on global `clean_images\` |
| `scripts\run_attack.py` | Older single-attack runner (pre-pipeline layout) |
| `scripts\cleanup_empty_attacks.py` | Remove empty pre-created attack folders |
| `scripts\setup_attack_folders.py` | Older attack-folder setup utility |

---

## 6. Where results live

### Per attack (`attacks\attack_XX\`)

```text
attacks\attack_XX\
  clean_images\
  patch\
  patched_images\
  results\
    selected_images.csv
    clean_baseline.csv
    clean_baseline_summary.txt
    attack_results.csv
    attack_summary.txt
    attack_overview.jpg
    clean_predictions\images\
    clean_predictions\labels\
    attacked_predictions\images\
    attacked_predictions\labels\
    verification\
    comparison\
      clean_vs_attacked.csv
      comparison_summary.txt
      confidence_comparison.png
      confidence_drop_distribution.png
      iou_comparison.png
      detection_comparison.png
      person_vs_car_comparison.png
      attack_success_overview.png
```

### Global experiment folder

| Path | Contents |
|------|----------|
| `Adversarial-Patch-Experiment\diffusion_patches\` | Candidate patch PNGs |
| `Adversarial-Patch-Experiment\results\all_attacks_comparison.csv` | One row per completed attack |
| `Adversarial-Patch-Experiment\results\all_attacks_comparison.png` | Cross-attack comparison chart |
| `Adversarial-Patch-Experiment\results\clean_baseline.csv` | Early global 20-image baseline (legacy) |

---

## 7. Everyday checklist

1. Prefer **Git Bash** or **PowerShell** consistently (do not mix syntax).
2. From experiment folder, run `run_attack_pipeline.py` (no `--num-images` needed).
3. Wait until the run prints `Automatic attack pipeline complete`.
4. Check `attacks\attack_XX\results\attack_summary.txt` and `results\comparison\`.
5. Compare attacks via `results\all_attacks_comparison.csv` / `.png`.

---

*Last updated for the automatic pipeline with increasing sample size, IoU ≥ 0.50 target matching, and automatic diffusion patch generation.*
