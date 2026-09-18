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
| Eligible | 179 | 184 | 363 |
| Success  | 2 | 3 | 5 |
| ASR (%)  | 1.12 | 1.63 | 1.38 |

| Metric                 | Value |
| ---------------------- | ----: |
| Total images           | 400 |
| Eligible               | 363 |
| Success                | 5 |
| ASR (%)                | 1.38 |
| Mean confidence drop   | 0.021184 |
| Median confidence drop | 0.003131 |
| Mean IoU drop          | 0.003974 |
| Median IoU drop        | 0.000695 |
| Both-valid n           | 358 |

## Interpretation

On 400 images, 363 targets were eligible (clean-valid under conf>=0.25 and IoU>=0.50). Of these, 5 became invalid after the Hyper-YOLO patch (ASR=1.38%). Person ASR=1.12% (2/179); car ASR=1.63% (3/184). Among 358 both-valid targets, mean confidence drop=0.021184 and mean IoU drop=0.003974. These figures describe transfer behavior of existing Hyper-YOLO adversarial images on D-FINE-S only; they do not claim universal attack effectiveness.
