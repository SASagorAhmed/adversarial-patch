# Deformable DETR R50 — Hyper-YOLO Transfer Attack

Transferability evaluation of **existing** Hyper-YOLO adversarial images against **standard/base Deformable DETR R50** (ResNet-50, multi-scale).

## Model

- Paper: [Deformable DETR](https://arxiv.org/abs/2010.04159)
- Official GitHub: https://github.com/fundamentalvision/Deformable-DETR
- Config: `configs/r50_deformable_detr.sh`
- Official COCO box AP: **44.5**
- Variant: standard/base (`two_stage=False`, `with_box_refine=False`, `num_feature_levels=4`)

## Checkpoint

SenseTime official R50 weights via Hugging Face:

`SenseTime/deformable-detr`

Cached locally under `weights/SenseTime_deformable-detr/`.

### Windows compatibility note

The official repo requires compiling CUDA `ms_deform_attn` operators (Linux + nvcc). This host has **no MSVC/nvcc**. Therefore inference uses Transformers’ Deformable DETR implementation with the **same SenseTime R50 COCO weights and standard architecture**. Architecture was verified at load time (not single-scale, not box-refine, not two-stage).

## Protocol (same as Faster R-CNN / RT-DETRv2 / D-FINE / YOLOv12)

- Confidence ≥ 0.25
- IoU ≥ 0.50
- Targets: person, car
- Eligible = clean valid
- Success = clean valid ∧ patched invalid
- Conf/IoU drops = both-valid only
- No patch regeneration / training / fine-tuning

## Reproduce

```powershell
cd "D:\project CS\Deformable-DETR-R50-COCO-Attack"
.\.venv\Scripts\Activate.ps1
python scripts\prepare_attack_data.py --attack all
python scripts\run_attack.py --attack smoke
python scripts\run_attack.py --attack attack_01
python scripts\run_attack.py --attack all
```

## GT / COCO

- Selected GT: each attack’s `selected_images.csv`
- Annotations available at: `D:\project CS\Hyper-YOLO\coco\annotations\instances_val2017.json`
- Val images: `D:\project CS\Hyper-YOLO\coco\val2017\val2017`

## Outputs

`results/attack_XX/` contains `clean_predictions/`, `patched_predictions/`, `visualizations/`, `per_image_results.csv`, `summary.json`, `summary_result.txt`, `report.md`.

**No `combined_analysis/` is created by this project.**
