# Faster-RCNN-COCO-Attack

Independent research detector project for an adversarial-patch benchmark.

## Exact model

- **Builder:** `torchvision.models.detection.fasterrcnn_resnet50_fpn_v2`
- **Weights:** `FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1`
- **Official repo:** https://github.com/pytorch/vision
- **Official COCO val2017 box mAP (Torchvision meta):** 46.7
- **Parameters (Torchvision meta):** 43,712,278 (~43.7M)

The model stays **pretrained and frozen**. It is **not** fine-tuned. The full COCO category set is retained.

## Environment

Isolated venv only:

```text
D:\project CS\Faster-RCNN-COCO-Attack\.venv\
```

Activate / run:

```powershell
cd "D:\project CS\Faster-RCNN-COCO-Attack"
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
# (PyTorch CUDA wheels were installed from download.pytorch.org)
.\.venv\Scripts\python.exe src\verify_setup.py
```

## Clean COCO inference

Test image: COCO val2017 `000000407083.jpg`  
(extracted from existing `Hyper-YOLO\coco\val2017.zip` **into this project only**; existing projects were not modified).

```powershell
.\.venv\Scripts\python.exe src\infer_image.py
.\.venv\Scripts\python.exe src\verify_setup.py
```

Outputs:

- `results\clean_inference\`
- `results\verification\model_verification.json`
- `results\verification\model_verification.md`

## Safety

Do **not** modify Hyper-YOLO, DISPATCH-*, Teacher Notebook, Adversarial-Patch-Experiment, or Stable-Diffusion-Patch from this project.

No adversarial attack generation is included in this setup step.
