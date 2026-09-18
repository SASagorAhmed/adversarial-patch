# Code path comparison

## Current Automatic (testing_mitigations/mitigation_02/automatic)

`run_automatic.py`
→ load attacked JPEG (eval clean separately)
→ LANCZOS 512 RGB uint8
→ `generate_checkerboard_masks` N=32
→ `LDMInpaintAdapter.inpaint` pass0 (reseed)
→ `LDMInpaintAdapter.inpaint` pass1 (reseed)
→ `combine_regeneration`
→ `compute_l2_difference` + `smooth_difference` OpenCV k=5 σ=1
→ `predict_adversarial_mask` sklearn KMeans k=2 n_init=10 seeded
→ NEAREST mask to original (eval)
→ `rectify_image` at 512, LANCZOS to original JPEG
→ Hyper-YOLO cpu conf 0.25 iou 0.70 imgsz 640
→ `match_target` class + IoU≥0.50

True patch: rasterized from saved coords **after** predicted mask write (`*_localization_eval.json` note).

## Official Automatic (`LDM/scripts/d3_inference.py`)

load jpg → BICUBIC 512 → ToTensor
→ checkerboard integer tiles
→ batch=2 CompVis encode + one DDIM sample steps=5
→ compose predicted halves
→ torch L2 + torchvision GaussianBlur k=min(15,512/32-1) σ=1
→ KMeans k=2 n_init=auto unseeded, scalar D
→ MORPH_OPEN 3×3
→ rectify at 512
→ BICUBIC to original
→ (paper eval: MMDetection, not Hyper-YOLO)

## Isolated reference (`DISPATCH-Paper-Faithful/implementation`)

Follows official `d3_inference.py` (batched sample, kernel 15, MORPH_OPEN, BICUBIC I/O, integer checkerboard).
Keeps current **evaluation protocol** (NEAREST mask IoU at original res, Hyper-YOLO canonical rule) so mask/detector numbers are comparable to testing mit02.
KMeans `random_state=20260817` added only for diagnostic repeatability (documented MINOR vs official unseeded KMeans).
Does not use true patch / GT / class until after mask freeze.
