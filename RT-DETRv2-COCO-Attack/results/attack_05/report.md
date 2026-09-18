# attack_05 — Hyper-YOLO-to-RT-DETRv2-S Transfer Attack

Model: RT-DETRv2-S
Weights: rtdetrv2_r18vd_120e_coco_rerun_48.1.pth
Dataset: COCO 2017
Attack source: Existing Hyper-YOLO adversarial images
Attack type: Hyper-YOLO-to-RT-DETRv2-S Transfer Attack

The adversarial patches were generated for Hyper-YOLO and evaluated on RT-DETRv2-S without regeneration, re-optimization, or training against RT-DETRv2-S.

This is NOT a white-box RT-DETRv2 attack.

Total images: 400
Eligible targets: 359
Successful attacks: 10
ASR: 2.79%

Person:
  eligible = 178
  successful = 4
  ASR = 2.25%

Car:
  eligible = 181
  successful = 6
  ASR = 3.31%

Mean confidence drop: 0.025238
Median confidence drop: 0.003124
Mean IoU drop: 0.003591
Both-valid n: 349
