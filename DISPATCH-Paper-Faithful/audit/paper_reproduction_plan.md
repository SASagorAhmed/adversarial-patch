# Optional true paper reproduction — PLAN ONLY (not executed)

Do not download INRIA-Person, APRICOT, MMDetection weights, or run this without explicit user approval.

## Paper setup to reproduce (arXiv:2509.04597v2 + official repo 80c59b9b)

- Dataset hiding: INRIA-Person (614 train / 288 test). Official README Google Drive preprocessed `datasets/INRIAPerson`.
- Dataset creating: APRICOT 873 test images.
- Detectors (MMDetection 3.3.0, COCO-pretrained): YOLOv3, Faster R-CNN, DETR (hiding); Faster R-CNN, SSD, RetinaNet (creating). Checkpoints listed in `mmdet_checkpoints/download_links.txt`.
- Attacks hiding: AdvPatch, NatPatch (scale 0.2 = patch height / bbox **diagonal**), AdvTexture (0.25). Retrain on INRIA train; apply to every person on test (`patch_apply.py`).
- Defense: official `python LDM/scripts/d3_inference.py --indir ... --outdir ...` (defaults inpaint_size=512, num_grids=32, steps=5). CompVis `inpainting_big/last.ckpt`.
- Metrics hiding: mAP@0.5 and AR. Creating: mAP@0.5 treating patch as GT (lower better) and APRICOT ASR (conf>0.3, IoU>0.1 with patch).
- Randomness: paper averages **3 runs**.
- Hardware in paper: NVIDIA A100. Official timing ~0.32 s/img on that class of GPU.

## What this plan must not do

- Do not replace Hyper-YOLO experiments with paper numbers.
- Do not modify DISPATCH-Defense.
- Do not claim testing_mitigations recovery % equals paper 89.3% mAP@0.5.

## Execution gate

Requires a separate user command after reviewing disk, GPU, and license/dataset terms.
