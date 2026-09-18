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
| Eligible | 109 | 98 | 207 |
| Success  | 4 | 4 | 8 |
| ASR (%)  | 3.67 | 4.08 | 3.86 |

| Metric                 | Value |
| ---------------------- | ----: |
| Total images           | 250 |
| Eligible               | 207 |
| Success                | 8 |
| ASR (%)                | 3.86 |
| Mean confidence drop   | 0.027582 |
| Median confidence drop | 0.006212 |
| Mean IoU drop          | -0.001438 |
| Median IoU drop        | -0.000343 |
| Both-valid n           | 199 |

## Interpretation

On 250 images, 207 targets were eligible (clean-valid under conf>=0.25 and IoU>=0.50). Of these, 8 became invalid after the Hyper-YOLO patch (ASR=3.86%). Person ASR=3.67% (4/109); car ASR=4.08% (4/98). Among 199 both-valid targets, mean confidence drop=0.027582 and mean IoU drop=-0.001438. These figures measure transfer behavior of existing Hyper-YOLO-N adversarial images on YOLOv12-S only; they do not establish that YOLOv12 is inherently vulnerable, nor claim universal attack effectiveness.
