# Weights

This project uses the official Torchvision pretrained checkpoint:

- Model builder: `torchvision.models.detection.fasterrcnn_resnet50_fpn_v2`
- Weights enum: `FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1`
- Official download URL (Torchvision):  
  `https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth`

Weights are downloaded automatically by Torchvision into the local Torch hub / Torchvision cache.
Do **not** place unofficial checkpoints here.

After `python -m src.verify_setup`, see:

- `results/verification/model_verification.json` for the resolved cache path, file size, and hash metadata
- Torchvision `weights.meta` for official COCO val2017 box mAP and parameter count
