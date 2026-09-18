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
| Eligible | 153 | 161 | 314 |
| Success  | 4 | 1 | 5 |
| ASR (%)  | 2.61 | 0.62 | 1.59 |

| Metric                 | Value |
| ---------------------- | ----: |
| Total images           | 350 |
| Eligible               | 314 |
| Success                | 5 |
| ASR (%)                | 1.59 |
| Mean confidence drop   | 0.016473 |
| Median confidence drop | 0.002877 |
| Mean IoU drop          | 0.005923 |
| Median IoU drop        | 0.000391 |
| Both-valid n           | 309 |

## Interpretation

On 350 images, 314 targets were eligible (clean-valid under conf>=0.25 and IoU>=0.50). Of these, 5 became invalid after the Hyper-YOLO patch (ASR=1.59%). Person ASR=2.61% (4/153); car ASR=0.62% (1/161). Among 309 both-valid targets, mean confidence drop=0.016473 and mean IoU drop=0.005923. These figures describe transfer behavior of existing Hyper-YOLO adversarial images on D-FINE-S only; they do not claim universal attack effectiveness.
