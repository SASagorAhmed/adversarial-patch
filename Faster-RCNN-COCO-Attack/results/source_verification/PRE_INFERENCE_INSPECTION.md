# Faster R-CNN Transfer Attack — Pre-Inference Inspection Report
# Generated from read-only inspection of Adversarial-Patch-Experiment

## Image counts (filename-paired clean ↔ patched)

| Attack | Clean | Patched | Verified pairs |
|--------|------:|--------:|---------------:|
| attack_01 | 200 | 200 | 200 |
| attack_02 | 250 | 250 | 250 |
| attack_03 | 300 | 300 | 300 |
| attack_04 | 350 | 350 | 350 |
| attack_05 | 400 | 400 | 400 |

All selected_images.csv rows have matching clean and patched filenames.
No unresolved pairs found during inspection.

## Paths

Clean source path:
`D:\project CS\Adversarial-Patch-Experiment\attacks\attack_XX\clean_images\`

Patched source path:
`D:\project CS\Adversarial-Patch-Experiment\attacks\attack_XX\patched_images\`

Metadata path (GT / target):
`D:\project CS\Adversarial-Patch-Experiment\attacks\attack_XX\results\selected_images.csv`

Hyper-YOLO result reference (read-only ASR):
`D:\project CS\Adversarial-Patch-Experiment\attacks\attack_XX\results\attack_summary.txt`
`D:\project CS\Adversarial-Patch-Experiment\attacks\attack_XX\results\attack_results.csv`

## Evaluation protocol (from pipeline_core.py — NOT invented)

Confidence threshold (detector): **0.25**  
(`CONF_THRESHOLD` in `scripts/pipeline_core.py` and `scripts/mitigation_config.py`)

NMS IoU threshold (Hyper-YOLO predict): **0.7**  
(`IOU_THRESHOLD` — used for YOLO NMS; not the GT matching threshold)

Target-match IoU threshold: **0.50**  
(`TARGET_MATCH_IOU_THRESHOLD`)

Matching rule:
- predicted class name == selected `target_class`
- among same-class detections, choose **highest IoU** with the selected COCO GT bbox
- valid detection iff that IoU **>= 0.50**

Eligible-target rule:
- `eligible_for_asr = Yes` iff **clean** target is validly detected
- patched image is NOT used to establish eligibility

Attack-success rule:
- `attack_success = Yes` iff eligible AND patched target is **NOT** validly detected

ASR denominator:
- number of eligible targets (clean valid detections)

GT bbox:
- from `selected_images.csv` (`bbox_x, bbox_y, bbox_width, bbox_height`) — COCO GT, not Faster R-CNN boxes

## Hyper-YOLO reference ASR (existing, read-only)

| Attack | Eligible | Success | ASR | Person ASR | Car ASR |
|--------|--------:|--------:|----:|-----------:|--------:|
| attack_01 | 169 | 9 | 5.33% | 4.49% | 6.25% |
| attack_02 | 203 | 15 | 7.39% | 3.77% | 11.34% |
| attack_03 | 240 | 12 | 5.00% | 1.57% | 8.85% |
| attack_04 | 287 | 13 | 4.53% | 2.74% | 6.38% |
| attack_05 | 330 | 20 | 6.06% | 2.30% | 10.26% |
