# Hyper-YOLO-N Evaluation Summary (First 500 COCO val2017 Images)

## Setup
- Model: weights/hyper-yolon.pt (Hyper-YOLO-N, pretrained)
- Dataset subset: first 500 images from COCO val2017 (sorted by filename)
- Image size: 640
- Device: CPU (torch 2.0.1+cpu)
- Confidence threshold for COCO mAP: 0.001
- NMS IoU: 0.7
- No training was performed

## Report Table

| Metric | Value |
|---|---:|
| Images processed | 500 |
| Images with >=1 detection | 500 |
| Ground-truth instances | 3504 |
| Detected objects (conf>=0.001) | 76456 |
| Detected objects (conf>=0.25) | 2965 |
| Average confidence (conf>=0.001) | 0.0382 |
| Average confidence (conf>=0.25) | 0.5990 |
| Preprocess time (ms/image) | 1.0 |
| Inference time (ms/image) | 152.5 |
| Postprocess time (ms/image) | 2.5 |
| Total time (ms/image) | 156.0 |
| Precision | 0.661 |
| Recall | 0.569 |
| mAP@0.5 (Ultralytics) | 0.604 |
| mAP@0.5:0.95 (Ultralytics) | 0.445 |
| mAP@0.5 (pycocotools) | 0.607 |
| mAP@0.5:0.95 (pycocotools) | 0.446 |

## Comparison with Hyper-YOLO Paper (HyperYOLO-N)

| Metric | This run (500 images) | Paper (full val2017, 5000 images) |
|---|---:|---:|
| mAP@0.5 | 60.7 | 58.3 |
| mAP@0.5:0.95 | 44.6 | 41.8 |

### Notes
- Paper numbers are from the Hyper-YOLO README/TPAMI paper for HyperYOLO-N at 640 on the **full** COCO val2017 set.
- This experiment uses a **500-image subset**, so scores are an approximation and can be higher or lower than the full-set result due to sampling.
- Prefer **pycocotools** mAP columns for paper-style reporting (AP and AP50).
- Precision/Recall above are Ultralytics DetMetrics values (best-F1 operating point), not COCO AR.

## Metric Definitions
- **Precision**: TP / (TP + FP) — fraction of detections that are correct.
- **Recall**: TP / (TP + FN) — fraction of ground-truth objects that were found.
- **mAP@0.5**: mean Average Precision at IoU threshold 0.50.
- **mAP@0.5:0.95**: COCO primary metric; mean AP averaged over IoU thresholds 0.50, 0.55, ..., 0.95.
- **Inference time per image**: model forward-pass time only (excludes preprocess/postprocess).

## Files to Include in Your Report
- 
uns/val_500/summary.md (this file)
- 
uns/val_500/summary.csv
- 
uns/val_500/predictions.json
- 
uns/val_500/PR_curve.png
- 
uns/val_500/P_curve.png
- 
uns/val_500/R_curve.png
- 
uns/val_500/F1_curve.png
- 
uns/val_500/confusion_matrix.png
- 
uns/val_500/val_batch0_pred.jpg (and other al_batch*_pred.jpg samples)
