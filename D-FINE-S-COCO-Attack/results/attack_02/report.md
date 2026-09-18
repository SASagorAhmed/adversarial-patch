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
| Eligible | 114 | 115 | 229 |
| Success  | 2 | 4 | 6 |
| ASR (%)  | 1.75 | 3.48 | 2.62 |

| Metric                 | Value |
| ---------------------- | ----: |
| Total images           | 250 |
| Eligible               | 229 |
| Success                | 6 |
| ASR (%)                | 2.62 |
| Mean confidence drop   | 0.016758 |
| Median confidence drop | 0.004334 |
| Mean IoU drop          | 0.006173 |
| Median IoU drop        | 0.000324 |
| Both-valid n           | 223 |

## Interpretation

On 250 images, 229 targets were eligible (clean-valid under conf>=0.25 and IoU>=0.50). Of these, 6 became invalid after the Hyper-YOLO patch (ASR=2.62%). Person ASR=1.75% (2/114); car ASR=3.48% (4/115). Among 223 both-valid targets, mean confidence drop=0.016758 and mean IoU drop=0.006173. These figures describe transfer behavior of existing Hyper-YOLO adversarial images on D-FINE-S only; they do not claim universal attack effectiveness.
