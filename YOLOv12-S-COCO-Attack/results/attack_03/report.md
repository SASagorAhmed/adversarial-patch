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
| Eligible | 128 | 122 | 250 |
| Success  | 0 | 3 | 3 |
| ASR (%)  | 0.00 | 2.46 | 1.20 |

| Metric                 | Value |
| ---------------------- | ----: |
| Total images           | 300 |
| Eligible               | 250 |
| Success                | 3 |
| ASR (%)                | 1.20 |
| Mean confidence drop   | 0.025300 |
| Median confidence drop | 0.004718 |
| Mean IoU drop          | -0.000169 |
| Median IoU drop        | -0.000086 |
| Both-valid n           | 247 |

## Interpretation

On 300 images, 250 targets were eligible (clean-valid under conf>=0.25 and IoU>=0.50). Of these, 3 became invalid after the Hyper-YOLO patch (ASR=1.20%). Person ASR=0.00% (0/128); car ASR=2.46% (3/122). Among 247 both-valid targets, mean confidence drop=0.025300 and mean IoU drop=-0.000169. These figures measure transfer behavior of existing Hyper-YOLO-N adversarial images on YOLOv12-S only; they do not establish that YOLOv12 is inherently vulnerable, nor claim universal attack effectiveness.
