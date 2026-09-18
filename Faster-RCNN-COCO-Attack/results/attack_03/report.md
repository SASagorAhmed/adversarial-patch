# attack_03 — Transfer Attack Report

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
300

Eligible targets:
277

Successful attacks:
1

ASR:
0.36%

Person:
eligible = 134
successful = 0
ASR = 0.00%

Car:
eligible = 143
successful = 1
ASR = 0.70%

Mean confidence drop:
0.007127

Median confidence drop:
0.000130

Mean IoU drop:
-0.001163

Note: mean/median over eligible where BOTH clean and patched matches are valid; misses excluded
(n=276 both-valid eligible pairs)
