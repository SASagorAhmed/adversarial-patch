# attack_02 — Transfer Attack Report

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
250

Eligible targets:
225

Successful attacks:
2

ASR:
0.89%

Person:
eligible = 113
successful = 1
ASR = 0.88%

Car:
eligible = 112
successful = 1
ASR = 0.89%

Mean confidence drop:
0.009843

Median confidence drop:
0.000140

Mean IoU drop:
0.001531

Note: mean/median over eligible where BOTH clean and patched matches are valid; misses excluded
(n=223 both-valid eligible pairs)
