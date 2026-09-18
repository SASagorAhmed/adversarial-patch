# attack_04 — Transfer Attack Report

Model:
Faster R-CNN ResNet-50 FPN V2

Weights:
FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1

Dataset:
COCO 2017

Attack source:
Existing Hyper-YOLO adversarial images

Attack type:
Transfer Attack

Total images:
350

Eligible targets:
314

Successful attacks:
4

ASR:
1.27%

Person:
eligible = 151
successful = 1
ASR = 0.66%

Car:
eligible = 163
successful = 3
ASR = 1.84%

Mean confidence drop:
0.020942

Median confidence drop:
0.000204

Mean IoU drop:
0.001895

Note: mean/median over eligible where BOTH clean and patched matches are valid; misses excluded
(n=310 both-valid eligible pairs)
