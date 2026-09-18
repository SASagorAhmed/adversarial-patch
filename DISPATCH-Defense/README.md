# DISPATCH-Defense

Independent implementation and evaluation of:

**DISPATCH: Disarming Adversarial Patches in Object Detection with Diffusion Models**  
Paper: https://arxiv.org/abs/2509.04597

## Project purpose

Research mitigation of adversarial patches on object detectors using the DISPATCH
**Regenerate and Rectify** pipeline with the official CompVis Latent Diffusion
inpainting model. This project is fully isolated under:

`D:\project CS\DISPATCH-Defense\`

## Method

```text
Attacked Image
↓
Checkerboard Inpainting Pass 0
+
Inverse Checkerboard Inpainting Pass 1
↓
Full Regenerated Image
↓
Original vs Regenerated L2 Difference
↓
Smoothing
↓
KMeans k=2
↓
Automatic Adversarial Mask
↓
Selective Rectification
↓
Rectified Image
↓
Hyper-YOLO Evaluation
```

## Important statement

**DISPATCH does not receive the true patch location.**

True patch geometry may be used **only after** rectification, and only for
localization evaluation metrics.

## External read-only resources

These may be **read** but never modified:

- `D:\project CS\Adversarial-Patch-Experiment\` (frozen attacks / COCO resources)
- `D:\project CS\Hyper-YOLO\` (detector + `weights\hyper-yolon.pt`)
- `D:\project CS\Stable-Diffusion-Patch\` (not used by DISPATCH LDM path)

Previous mitigation folders (`mitigation_01`, `mitigation_02`, pilots, etc.) are
**not** used as experiment sources.

## Storage

All new files remain under:

`D:\project CS\DISPATCH-Defense\`

## Environment

```powershell
cd "D:\project CS\DISPATCH-Defense"
.\.venv_dispatch\Scripts\Activate.ps1
$env:PYTHONPATH = "D:\project CS\DISPATCH-Defense\third_party\latent-diffusion;D:\project CS\DISPATCH-Defense\third_party\taming-transformers;D:\project CS\DISPATCH-Defense"
```

Or call Python directly (with the same `PYTHONPATH`):

```powershell
& "D:\project CS\DISPATCH-Defense\.venv_dispatch\Scripts\python.exe" ...
```

Compatibility note: project path contains a space; set `PYTHONPATH` as above so CompVis `ldm` / `taming` imports resolve reliably.

## Commands

### Verify environment

```powershell
cd "D:\project CS\DISPATCH-Defense"
.\.venv_dispatch\Scripts\python.exe scripts\verify_environment.py
.\.venv_dispatch\Scripts\python.exe scripts\verify_checkpoint.py
.\.venv_dispatch\Scripts\python.exe scripts\verify_external_resources.py
```

### Validate-only (no experiment folders)

```powershell
.\.venv_dispatch\Scripts\python.exe scripts\run_dispatch_pipeline.py --validate-only
```

### Official CompVis LDM smoke test

```powershell
.\.venv_dispatch\Scripts\python.exe scripts\run_dispatch_pipeline.py --smoke-ldm --steps 5
```

### One-image DISPATCH smoke test

```powershell
.\.venv_dispatch\Scripts\python.exe scripts\run_dispatch_pipeline.py --smoke-dispatch --steps 5
```

### Future manifest creation (no full run)

```powershell
.\.venv_dispatch\Scripts\python.exe scripts\build_evaluation_manifest.py --attack attack_02
```

### Future real experiment

Not enabled until explicitly approved. Pipeline refuses large runs without
`--run --i-approve-full-run`.

### Real mitigation run (mitigation-style folders)

```powershell
cd "D:\project CS\DISPATCH-Defense"
.\.venv_dispatch\Scripts\Activate.ps1
$env:PYTHONPATH = "D:\project CS\DISPATCH-Defense\third_party\latent-diffusion;D:\project CS\DISPATCH-Defense\third_party\taming-transformers;D:\project CS\DISPATCH-Defense"

# Creates next mitigations\mitigation_XX\ (never overwrites)
.\.venv_dispatch\Scripts\python.exe scripts\run_mitigation.py `
  --attack attack_02 --n-success 15 --n-control 15 --steps 5 --i-approve-full-run
```

Outputs follow the previous mitigation folder style under:

`D:\project CS\DISPATCH-Defense\mitigations\mitigation_XX\`

## Official LDM dependency

- Repo: https://github.com/CompVis/latent-diffusion.git  
  Cloned to: `third_party\latent-diffusion\`
- Checkpoint: `models\ldm\inpainting_big\last.ckpt`
- Official entry: `scripts\inpaint.py` (DDIM; **no text prompt**)

Paper default for DISPATCH sampling steps: **5** (configured in `config\dispatch_config.py`).

## Seed

Central seed: `DISPATCH_SEED = 20260817` in `config\dispatch_config.py`.
