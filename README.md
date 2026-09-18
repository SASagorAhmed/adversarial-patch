# adversarial-patch (Project CS)

Monorepo for Hyper-YOLO adversarial patch attacks, DISPATCH defense experiments, and black-box transfer evaluation across multiple COCO detectors.

**GitHub:** https://github.com/SASagorAhmed/adversarial-patch

## Major subprojects

| Folder | Role |
| --- | --- |
| `Adversarial-Patch-Experiment/` | Attack/mitigation pipelines + mitigation results (scripts point at Hyper-YOLO attacks) |
| `DISPATCH-Defense/` | DISPATCH mitigation + Automatic vs Known-Location diagnostics |
| `DISPATCH-Teacher-Notebook/` | Teacher demonstration notebooks |
| `DISPATCH-Paper-Faithful/` | Paper-faithful DISPATCH notes/code |
| `Faster-RCNN-COCO-Attack/` | Transfer eval: Faster R-CNN |
| `RT-DETRv2-COCO-Attack/` | Transfer eval: RT-DETRv2-S |
| `D-FINE-S-COCO-Attack/` | Transfer eval: D-FINE-S |
| `YOLOv12-S-COCO-Attack/` | Transfer eval: YOLOv12-S |
| `Deformable-DETR-R50-COCO-Attack/` | Transfer eval: Deformable DETR R50 |
| `Hyper-YOLO/` | Victim detector + Hyper-YOLO-N attack artifacts (`attacks/attack_01`–`05`) |
| `Stable-Diffusion-Patch/` | SD-related utilities (large models not in git) |

## What is in git vs ignored

**Tracked:** code, configs, notebooks, attack/results artifacts (summaries, CSV/JSON, visualizations), DISPATCH reports (including `DISPATCH-Defense/results/summary.md`), and research **model weights via Git LFS** (`Hyper-YOLO/weights/`, transfer `*/weights/`, `Adversarial-Patch-Experiment/mitigation_models/`).

**Ignored:** `.venv`, COCO datasets (`val2017` / `Hyper-YOLO/coco` / annotations), `third_party` clones, duplicate transfer `source_data` images, and the large local dump `Stable-Diffusion-Patch/model/` (~34 GB; re-download from Hugging Face if needed).

### Clone with weights

```powershell
git clone https://github.com/SASagorAhmed/adversarial-patch.git
cd adversarial-patch
git lfs install
git lfs pull
```

COCO images are not in the repo; place them under the paths documented in each project README.

## Note on Hyper-YOLO

`Hyper-YOLO/` is tracked in this monorepo (nested `.git` detached). Attack folders live at `Hyper-YOLO/attacks/`. Detector weights are under `Hyper-YOLO/weights/` (Git LFS). Local COCO/annotations/venvs stay ignored.

## Git hooks

Versioned hooks live in `.githooks/`. After clone, copy into the local repo hooks dir:

```powershell
Copy-Item .githooks\commit-msg .git\hooks\commit-msg -Force
```
