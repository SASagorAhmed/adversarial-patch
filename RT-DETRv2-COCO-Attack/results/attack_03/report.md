# attack_03 — Hyper-YOLO-to-RT-DETRv2-S Transfer Attack

Model: RT-DETRv2-S
Weights: rtdetrv2_r18vd_120e_coco_rerun_48.1.pth
Dataset: COCO 2017
Attack source: Existing Hyper-YOLO adversarial images
Attack type: Hyper-YOLO-to-RT-DETRv2-S Transfer Attack

The adversarial patches were generated for Hyper-YOLO and evaluated on RT-DETRv2-S without regeneration, re-optimization, or training against RT-DETRv2-S.

This is NOT a white-box RT-DETRv2 attack.

Total images: 300
Eligible targets: 264
Successful attacks: 4
ASR: 1.52%

Person:
  eligible = 130
  successful = 2
  ASR = 1.54%

Car:
  eligible = 134
  successful = 2
  ASR = 1.49%

Mean confidence drop: 0.013776
Median confidence drop: 0.002238
Mean IoU drop: 0.005740
Both-valid n: 260
