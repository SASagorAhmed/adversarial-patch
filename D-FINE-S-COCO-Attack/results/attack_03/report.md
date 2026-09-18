# D-FINE-S Transfer Attack Evaluation

## Model

D-FINE-S

## Paper

D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement

## Publication

ICLR 2025 Spotlight

## Dataset

MS COCO 2017

## Attack Source

Hyper-YOLO-N adversarial images

## Attack Type

Transfer attack / black-box evaluation

## Evaluation Thresholds

Confidence >= 0.25
IoU >= 0.50

## Target Classes

Person
Car

## Attack Protocol

Existing Hyper-YOLO clean and patched image pairs are evaluated independently with the same official COCO-pretrained D-FINE-S weights and identical preprocessing. A target is eligible only if it is validly detected on the clean image. Attack success requires an eligible clean detection that becomes invalid/missing on the patched image. Confidence and IoU drops are computed only for both-valid cases. No new patches were generated and no D-FINE training/fine-tuning was performed.

## Metrics

* Eligible
* Successful attacks
* ASR
* Mean confidence drop
* Median confidence drop
* Mean IoU drop
* Median IoU drop
* Both-valid count

## Results

| Metric   | Person | Car | Overall |
| -------- | -----: | --: | ------: |
| Eligible | 135 | 141 | 276 |
| Success  | 1 | 5 | 6 |
| ASR (%)  | 0.74 | 3.55 | 2.17 |

| Metric                 | Value |
| ---------------------- | ----: |
| Total images           | 300 |
| Eligible               | 276 |
| Success                | 6 |
| ASR (%)                | 2.17 |
| Mean confidence drop   | 0.020004 |
| Median confidence drop | 0.003787 |
| Mean IoU drop          | 0.002793 |
| Median IoU drop        | 0.000239 |
| Both-valid n           | 270 |

## Interpretation

On 300 images, 276 targets were eligible (clean-valid under conf>=0.25 and IoU>=0.50). Of these, 6 became invalid after the Hyper-YOLO patch (ASR=2.17%). Person ASR=0.74% (1/135); car ASR=3.55% (5/141). Among 270 both-valid targets, mean confidence drop=0.020004 and mean IoU drop=0.002793. These figures describe transfer behavior of existing Hyper-YOLO adversarial images on D-FINE-S only; they do not claim universal attack effectiveness.
