# Faster R-CNN ResNet-50 FPN V2 — Model Verification

1. **Model name:** Faster R-CNN ResNet-50 FPN V2 (`fasterrcnn_resnet50_fpn_v2`)
2. **Torchvision version:** 0.20.1+cu121
3. **Exact weights:** `FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1`
4. **COCO status:** pretrained COCO (FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1)
5. **COCO category count:** 91 (including `__background__`); 90 excluding background
6. **Official COCO val2017 box mAP:** 46.7 (expected 46.7)
7. **Parameters:** measured=43712278, meta=43712278 (expected 43712278)
8. **Weight size:** 175221657 bytes (meta file size MB: 167.104)
9. **Weight/cache path:** `C:\Users\Sagor Ahmed\.cache\torch\hub\checkpoints\fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth`
   - URL: https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth
   - SHA256: `dd69338a24b8d7381807e247652bdc356325bcbaf1cd3e092e00e0a1a58706bf`
10. **Device:** cuda
11. **GPU:** NVIDIA GeForce RTX 2050
12. **CUDA availability:** True (CUDA 12.1)
13. **Test image:** `D:\project CS\Faster-RCNN-COCO-Attack\data\coco_val2017\000000407083.jpg` (id=000000407083)
14. **Number of person detections** (vis conf≥0.25): 3 (raw=8)
15. **Number of car detections** (vis conf≥0.25): 3 (raw=4)
16. **Inference time:** 0.9099 sec
17. **PASS/FAIL:**
    - COCO: PASS
    - WEIGHTS: PASS
    - CLEAN INFERENCE: PASS
    - PERSON DETECTION: PASS
    - CAR DETECTION: PASS

## Notes

- Pretrained weights are frozen (`requires_grad=False`, `model.eval()`).
- Full COCO category set retained; not reduced to person/car.
- Preprocessing uses official `weights.transforms()`.
- Visualization uses confidence ≥ 0.25; raw JSON keeps all detections.
- No adversarial attack code in this setup step.

Source: extracted COCO val2017 image from existing zip (read-only), copied only into this project.
