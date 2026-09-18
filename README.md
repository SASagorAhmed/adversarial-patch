# adversarial-patch (Project CS)

Monorepo for Hyper-YOLO adversarial patch attacks, DISPATCH defense experiments, and black-box transfer evaluation across multiple COCO detectors.

**GitHub:** https://github.com/SASagorAhmed/adversarial-patch

## Major subprojects

| Folder | Role |
| --- | --- |
| `Adversarial-Patch-Experiment/` | Hyper-YOLO-N patch attacks (attack_01–05) |
| `DISPATCH-Defense/` | DISPATCH mitigation + Automatic vs Known-Location diagnostics |
| `DISPATCH-Teacher-Notebook/` | Teacher demonstration notebooks |
| `DISPATCH-Paper-Faithful/` | Paper-faithful DISPATCH notes/code |
| `Faster-RCNN-COCO-Attack/` | Transfer eval: Faster R-CNN |
| `RT-DETRv2-COCO-Attack/` | Transfer eval: RT-DETRv2-S |
| `D-FINE-S-COCO-Attack/` | Transfer eval: D-FINE-S |
| `YOLOv12-S-COCO-Attack/` | Transfer eval: YOLOv12-S |
| `Deformable-DETR-R50-COCO-Attack/` | Transfer eval: Deformable DETR R50 |
| `Hyper-YOLO/` | Victim detector codebase (weights/COCO/annotations not in git) |
| `Stable-Diffusion-Patch/` | SD-related utilities (large models not in git) |

## What is in git vs ignored

**Tracked:** code, configs, notebooks, attack/results artifacts (summaries, CSV/JSON, visualizations), DISPATCH reports (including `DISPATCH-Defense/results/summary.md`).

**Ignored:** `.venv`, COCO datasets (`val2017` etc.), model weights (`.pt`/`.pth`/`.ckpt`/`.safetensors`), `third_party` clones, duplicate `source_data` image copies, Stable Diffusion model dumps.

Re-download detector/LDM checkpoints from the URLs documented in each project README.

## Note on Hyper-YOLO

`Hyper-YOLO/` is tracked in this monorepo (nested `.git` detached). Local COCO/annotations/weights/venvs stay ignored.

## Git hooks

Versioned hooks live in `.githooks/`. After clone, copy into the local repo hooks dir:

```powershell
Copy-Item .githooks\commit-msg .git\hooks\commit-msg -Force
```
