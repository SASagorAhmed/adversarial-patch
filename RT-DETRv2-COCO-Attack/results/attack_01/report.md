# attack_01 — Hyper-YOLO-to-RT-DETRv2-S Transfer Attack

Model: RT-DETRv2-S
Weights: rtdetrv2_r18vd_120e_coco_rerun_48.1.pth
Dataset: COCO 2017
Attack source: Existing Hyper-YOLO adversarial images
Attack type: Hyper-YOLO-to-RT-DETRv2-S Transfer Attack

The adversarial patches were generated for Hyper-YOLO and evaluated on RT-DETRv2-S without regeneration, re-optimization, or training against RT-DETRv2-S.

This is NOT a white-box RT-DETRv2 attack.

Total images: 200
Eligible targets: 184
Successful attacks: 3
ASR: 1.63%

Person:
  eligible = 91
  successful = 2
  ASR = 2.20%

Car:
  eligible = 93
  successful = 1
  ASR = 1.08%

Mean confidence drop: 0.013282
Median confidence drop: 0.001600
Mean IoU drop: 0.005105
Both-valid n: 181
