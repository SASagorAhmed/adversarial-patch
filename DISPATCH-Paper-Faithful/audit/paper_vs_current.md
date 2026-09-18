# Paper vs current DISPATCH-Defense implementation

Paper: DisPatch: Disarming Adversarial Patches in Object Detection with Diffusion Models
arXiv:2509.04597v2 (HTML https://arxiv.org/html/2509.04597v2)
Official code: https://github.com/MaJinWakeUp/DisPatch commit `80c59b9b4a6c4060107a8b4485d3885128c2ae81`

Current code inspected READ-ONLY under `D:\project CS\DISPATCH-Defense\` (scripts + testing_mitigations/mitigation_02). No edits.

## Algorithm 1 / equations (paper)

Regeneration:

- `I0_tilde = D(I; m0)`, `I1_tilde = D(I; m1)`  (Eq. 1)
- `I_tilde = I0_tilde ⊙ m0 ⊕ I1_tilde ⊙ m1`  (Eq. 2)
- Complementary masks: `m0 ⊕ m1 = J` (all-ones). Value 1 = regenerate, 0 = preserve. N×N checkerboard. Default **N=32**.

Rectification / Algorithm 1:

1. Normalize I and I_tilde to **[0,1]**
2. Pixel-wise RGB L2: `D[i,j] = ||I[i,j] - I_tilde[i,j]||_2`
3. Gaussian smoothing of D (citation: Gonzalez DIP; **kernel/sigma not specified in the paper**)
4. Flatten D; KMeans **k=2**
5. Smaller centroid = benign (A=0); **larger centroid = adversarial (A=1)**
6. `I_hat = A ⊙ I_tilde ⊕ (1-A) ⊙ I`  (Eq. 5)

Diffusion (Sec. IV-C):

- Backbone: Latent Diffusion inpainting (Rombach et al. / CompVis LDM)
- Resize all images to **512×512** for inpainting
- Default sampling steps **s=5**
- Results averaged over **3 runs** (randomness of DMs)
- LDM trained on MS-COCO, same family as paper detectors

Paper does **not** specify:

- Gaussian kernel size or sigma → `UNSPECIFIED_IN_PAPER`
- OpenCV vs torchvision blur
- Sequential vs batched two-pass sampling
- Morphology / shape completion
- PIL resample filter (LANCZOS vs BICUBIC)
- KMeans `n_init` / `random_state`

## Current behavior vs paper equations

| Paper | Current DISPATCH-Defense | Match |
|-------|--------------------------|-------|
| Eq. 1 two inpaint passes | Two `LDMInpaintAdapter.inpaint` calls | YES (sequential, not batched) |
| Eq. 2 complementary compose | `combine_regeneration`: `I0*m0 + I1*m1` | YES (equivalent after binary masks) |
| m0+m1=1 | `validate_masks` asserts allclose 1 | YES at 512 |
| 1=inpaint | Adapter: `masked=(1-mask)*image`; composite `mask*pred` | YES vs CompVis `inpaint.py` |
| Normalize [0,1] then L2 | `compute_l2_difference` divides by 255 if max>1 | YES |
| RGB L2 | NumPy RGB arrays, no BGR at L2 | YES |
| Gaussian smooth | OpenCV `GaussianBlur((5,5), sigmaX=1.0)` | PARTIAL — paper unspecified; official uses kernel 15, sigma 1.0 |
| KMeans k=2 on flattened D | sklearn KMeans k=2 on smoothed scalar D | YES |
| Higher centroid = adversarial | `adv_label = argmax(centers)` | YES |
| Eq. 5 rectify | `A*I_tilde + (1-A)*I` | YES |
| 512 stretch | `PIL.resize((512,512), LANCZOS)` no letterbox | YES stretch; resample differs from official BICUBIC |
| N=32, s=5 | `CHECKERBOARD_GRID_N=32`, `DIFFUSION_STEPS=5` | YES |
| No oracle in Automatic | True patch used only after mask freeze for eval | YES |
| 3-run average | Single seeded run `seed=20260817` | EXPERIMENT difference |

## What the paper is NOT claiming for our experiment

Paper metrics: mAP@0.5 / AR on INRIA-Person hiding attacks vs YOLOv3 / Faster R-CNN / DETR (MMDetection), patch scale 0.2 of **bbox diagonal**; APRICOT creating attacks. Average of 3 runs.

Our metrics: per-image target recovery on Hyper-YOLO-N, COCO val person/car, attack_02 30%-of-shorter-GT-side diffusion patch. **Not comparable as the same number.**
