# attack_02 — Hyper-YOLO-to-RT-DETRv2-S Transfer Attack

Model: RT-DETRv2-S
Weights: rtdetrv2_r18vd_120e_coco_rerun_48.1.pth
Dataset: COCO 2017
Attack source: Existing Hyper-YOLO adversarial images
Attack type: Hyper-YOLO-to-RT-DETRv2-S Transfer Attack

The adversarial patches were generated for Hyper-YOLO and evaluated on RT-DETRv2-S without regeneration, re-optimization, or training against RT-DETRv2-S.

This is NOT a white-box RT-DETRv2 attack.

Total images: 250
Eligible targets: 226
Successful attacks: 6
ASR: 2.65%

Person:
  eligible = 115
  successful = 4
  ASR = 3.48%

Car:
  eligible = 111
  successful = 2
  ASR = 1.80%

Mean confidence drop: 0.041074
Median confidence drop: 0.002234
Mean IoU drop: 0.008820
Both-valid n: 220
