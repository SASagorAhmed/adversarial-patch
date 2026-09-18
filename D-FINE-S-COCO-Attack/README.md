# D-FINE-S COCO Transfer Attack

Black-box transfer evaluation of existing Hyper-YOLO adversarial images on official D-FINE-S (COCO).

## Important

- Does **not** generate or optimize patches
- Does **not** train or fine-tune D-FINE
- Reads only from local `source_data/` copies
- Does **not** create `combined_analysis/`

## Setup

```powershell
cd "D:\project CS\D-FINE-S-COCO-Attack"
.\.venv\Scripts\Activate.ps1
```

## Run

```powershell
python scripts\prepare_source_data.py
python scripts\run_transfer_attack.py smoke
python scripts\run_transfer_attack.py full
```
