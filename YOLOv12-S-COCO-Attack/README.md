# YOLOv12-S COCO Transfer Attack

Black-box transfer evaluation of existing Hyper-YOLO adversarial images on official YOLOv12-S (COCO).

## Important

- Does **not** generate or optimize patches
- Does **not** train or fine-tune YOLOv12
- Reads only from local `source_data/` copies
- Does **not** create `combined_analysis/`

## Setup

```powershell
cd "D:\project CS\YOLOv12-S-COCO-Attack"
.\.venv\Scripts\Activate.ps1
```

## Run

```powershell
python scripts\prepare_source_data.py
python scripts\run_transfer_attack.py smoke
python scripts\run_transfer_attack.py full
```

Checkpoint: official `yolov12s.pt` from https://github.com/sunsmarterjie/yolov12/releases/download/v1.0/yolov12s.pt
