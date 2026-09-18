# Faster R-CNN Transfer Attack — Combined Report

Attack type: Hyper-YOLO-to-Faster-R-CNN Transfer Attack
Model: Faster R-CNN ResNet-50 FPN V2
Weights: FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1

Total images: 1500
Eligible targets: 1365
Successful attacks: 16
Overall ASR: 1.17%
Person ASR: 0.45%
Car ASR: 1.87%
Mean confidence drop (avg of attack means, both-valid only): 0.01170547427097467
Mean IoU drop (avg of attack means, both-valid only): 0.0013626396713162486

## Per attack
- attack_01: ASR=1.07% (success 2/187) person=1.09% car=1.05%
- attack_02: ASR=0.89% (success 2/225) person=0.88% car=0.89%
- attack_03: ASR=0.36% (success 1/277) person=0.00% car=0.70%
- attack_04: ASR=1.27% (success 4/314) person=0.66% car=1.84%
- attack_05: ASR=1.93% (success 7/362) person=0.00% car=3.80%
