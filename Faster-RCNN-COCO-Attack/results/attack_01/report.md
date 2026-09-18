# attack_01 — Transfer Attack Report

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
200

Eligible targets:
187

Successful attacks:
2

ASR:
1.07%

Person:
eligible = 92
successful = 1
ASR = 1.09%

Car:
eligible = 95
successful = 1
ASR = 1.05%

Mean confidence drop:
0.006666

Median confidence drop:
0.000119

Mean IoU drop:
0.003081

Note: mean/median over eligible where BOTH clean and patched matches are valid; misses excluded
(n=185 both-valid eligible pairs)
