# Deformable DETR R50 Transfer Attack Evaluation

## 1. Model information

- Model: Deformable DETR R50 (standard/base multi-scale)
- Backbone: ResNet-50
- Config correspondence: `configs/r50_deformable_detr.sh`
- Official COCO box AP: 44.5
- Paper: https://arxiv.org/abs/2010.04159
- GitHub: https://github.com/fundamentalvision/Deformable-DETR
- Checkpoint: SenseTime/deformable-detr
- Variant flags: two_stage=False, with_box_refine=False, num_feature_levels=4

## 2. Dataset information

- MS COCO 2017
- Target classes: person, car
- Selected GT boxes from Hyper-YOLO attack `selected_images.csv`

## 3. Transfer setup

The attack images were generated previously for Hyper-YOLO and are being evaluated here as transferred adversarial examples against Deformable DETR R50. These patches were **not** optimized for Deformable DETR.

## 4. Evaluation protocol

- Confidence >= 0.25
- IoU >= 0.50 vs selected GT (highest-IoU same-class match; confidence tie-break)
- Eligible = clean target valid
- Success = clean valid AND patched invalid
- Confidence/IoU drops = both-valid only

## 5. Attack-wise results

| Metric | Value |
| ------ | ----: |
| Total images | 350 |
| Eligible | 304 |
| Success | 11 |
| ASR (%) | 3.62 |

## 6. Person vs Car

| Class | Eligible | Success | ASR (%) |
| ----- | -------: | ------: | ------: |
| Person | 151 | 3 | 1.99 |
| Car | 153 | 8 | 5.23 |

## 7. Confidence analysis

- Both-valid n: 293
- Mean confidence drop: 0.026969
- Median confidence drop: 0.012452

## 8. IoU analysis

- Mean IoU drop: 0.003450
- Median IoU drop: 0.000380

## 9. Important observations

ASR=3.62% on 304 eligible targets (11 successes). Car ASR=5.23%, person ASR=1.99%.

## 10. Limitations

- Transfer evaluation only; no white-box attack on Deformable DETR.
- Official CUDA ms_deform_attn was not compiled on this Windows host; SenseTime weights were loaded via Transformers (same standard R50 architecture).

## 11. Reproducibility

- Device: cuda
- Python: 3.10.11
- PyTorch: 2.5.1+cu121
- Torchvision: 0.20.1+cu121
- Note: Hugging Face Transformers DeformableDetrForObjectDetection (SenseTime weights). Official CUDA ops not compiled: no nvcc/MSVC on this Windows host.
