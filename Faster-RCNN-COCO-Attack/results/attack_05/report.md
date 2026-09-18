# attack_05 — Transfer Attack Report

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
400

Eligible targets:
362

Successful attacks:
7

ASR:
1.93%

Person:
eligible = 178
successful = 0
ASR = 0.00%

Car:
eligible = 184
successful = 7
ASR = 3.80%

Mean confidence drop:
0.013950

Median confidence drop:
0.000221

Mean IoU drop:
0.001469

Note: mean/median over eligible where BOTH clean and patched matches are valid; misses excluded
(n=355 both-valid eligible pairs)
