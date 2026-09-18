# RT-DETRv2-COCO-Attack

Independent transfer-attack evaluation:

**Hyper-YOLO-to-RT-DETRv2-S Transfer Attack**

## Model

- RT-DETRv2-S (PResNet-18 / `rtdetrv2_r18vd_120e_coco`)
- Official repo: https://github.com/lyuwenyu/RT-DETR
- Paper: https://arxiv.org/abs/2407.17140
- Checkpoint: `weights/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth`
- Input: 640×640

## Important

Patches were generated for Hyper-YOLO. This project only evaluates existing
clean/patched pairs on RT-DETRv2-S. No patch regeneration, no fine-tuning.

## Run

```powershell
cd "D:\project CS\RT-DETRv2-COCO-Attack"
.\.venv\Scripts\python.exe src\prepare_source_data.py
.\.venv\Scripts\python.exe src\run_transfer_attack.py smoke
.\.venv\Scripts\python.exe src\run_transfer_attack.py full
```
