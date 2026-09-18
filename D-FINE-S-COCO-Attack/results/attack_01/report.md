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
| Eligible | 91 | 94 | 185 |
| Success  | 0 | 1 | 1 |
| ASR (%)  | 0.00 | 1.06 | 0.54 |

| Metric                 | Value |
| ---------------------- | ----: |
| Total images           | 200 |
| Eligible               | 185 |
| Success                | 1 |
| ASR (%)                | 0.54 |
| Mean confidence drop   | 0.026847 |
| Median confidence drop | 0.002828 |
| Mean IoU drop          | 0.004872 |
| Median IoU drop        | 0.001127 |
| Both-valid n           | 184 |

## Interpretation

On 200 images, 185 targets were eligible (clean-valid under conf>=0.25 and IoU>=0.50). Of these, 1 became invalid after the Hyper-YOLO patch (ASR=0.54%). Person ASR=0.00% (0/91); car ASR=1.06% (1/94). Among 184 both-valid targets, mean confidence drop=0.026847 and mean IoU drop=0.004872. These figures describe transfer behavior of existing Hyper-YOLO adversarial images on D-FINE-S only; they do not claim universal attack effectiveness.
