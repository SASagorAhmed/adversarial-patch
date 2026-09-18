# YOLOv12-S Transfer Attack Evaluation

## Model

YOLOv12-S

## Paper

YOLOv12: Attention-Centric Real-Time Object Detectors

## Publication

NeurIPS 2025

## Dataset

MS COCO 2017

## Attack Source

Hyper-YOLO-N adversarial images

## Attack Type

Transfer attack / black-box evaluation

## Thresholds

Confidence >= 0.25
IoU >= 0.50

## Target Classes

Person
Car

## Protocol

Existing Hyper-YOLO clean and patched image pairs are evaluated independently with the same official COCO-pretrained YOLOv12-S weights and identical official preprocessing (imgsz=640). A target is eligible only if it is validly detected on the clean image (same class, conf>=0.25, IoU>=0.50 vs selected COCO GT). Attack success requires an eligible clean detection that becomes invalid/missing on the patched image. Confidence and IoU drops are computed only for both-valid cases. No new patches were generated and no YOLOv12 training/fine-tuning was performed.

## Results

| Metric   | Person | Car | Overall |
| -------- | -----: | --: | ------: |
| Eligible | 89 | 83 | 172 |
| Success  | 1 | 6 | 7 |
| ASR (%)  | 1.12 | 7.23 | 4.07 |

| Metric                 | Value |
| ---------------------- | ----: |
| Total images           | 200 |
| Eligible               | 172 |
| Success                | 7 |
| ASR (%)                | 4.07 |
| Mean confidence drop   | 0.021915 |
| Median confidence drop | 0.004022 |
| Mean IoU drop          | 0.009103 |
| Median IoU drop        | 0.000986 |
| Both-valid n           | 165 |

## Interpretation

On 200 images, 172 targets were eligible (clean-valid under conf>=0.25 and IoU>=0.50). Of these, 7 became invalid after the Hyper-YOLO patch (ASR=4.07%). Person ASR=1.12% (1/89); car ASR=7.23% (6/83). Among 165 both-valid targets, mean confidence drop=0.021915 and mean IoU drop=0.009103. These figures measure transfer behavior of existing Hyper-YOLO-N adversarial images on YOLOv12-S only; they do not establish that YOLOv12 is inherently vulnerable, nor claim universal attack effectiveness.
