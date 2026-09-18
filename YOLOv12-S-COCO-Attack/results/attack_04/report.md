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
| Eligible | 148 | 149 | 297 |
| Success  | 2 | 14 | 16 |
| ASR (%)  | 1.35 | 9.40 | 5.39 |

| Metric                 | Value |
| ---------------------- | ----: |
| Total images           | 350 |
| Eligible               | 297 |
| Success                | 16 |
| ASR (%)                | 5.39 |
| Mean confidence drop   | 0.016757 |
| Median confidence drop | 0.004864 |
| Mean IoU drop          | 0.000630 |
| Median IoU drop        | 0.000186 |
| Both-valid n           | 281 |

## Interpretation

On 350 images, 297 targets were eligible (clean-valid under conf>=0.25 and IoU>=0.50). Of these, 16 became invalid after the Hyper-YOLO patch (ASR=5.39%). Person ASR=1.35% (2/148); car ASR=9.40% (14/149). Among 281 both-valid targets, mean confidence drop=0.016757 and mean IoU drop=0.000630. These figures measure transfer behavior of existing Hyper-YOLO-N adversarial images on YOLOv12-S only; they do not establish that YOLOv12 is inherently vulnerable, nor claim universal attack effectiveness.
