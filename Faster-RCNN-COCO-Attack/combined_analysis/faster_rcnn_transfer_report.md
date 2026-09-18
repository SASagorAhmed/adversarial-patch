# Faster R-CNN Transfer Attack — Combined Analysis

## Experiment identity

- **Model:** Faster R-CNN ResNet-50 FPN V2
- **Weights:** `FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1`
- **Dataset:** COCO 2017
- **Attack source:** existing Hyper-YOLO adversarial images
- **Attack type:** Hyper-YOLO-to-Faster-R-CNN Transfer Attack

### Terminology (important)

This is a **transfer attack**.

The adversarial patches were generated for **Hyper-YOLO** and then evaluated on **Faster R-CNN** using the already completed clean/patched image pairs.

The patches were **not** regenerated, re-optimized, or trained against Faster R-CNN.

This is **not** a white-box Faster R-CNN attack.

### Source of numbers

All values below are taken from the **already completed** final experimental summaries:

- `results\attack_01\`
- `results\attack_02\`
- `results\attack_03\`
- `results\attack_04\`
- `results\attack_05\`

No inference was re-run for this combined analysis.
No images were regenerated.
No additional verification pass was performed.

Confidence/IoU drop metrics are from eligible targets where **both** clean and patched matches were valid (both-valid). Combined mean confidence drop and mean IoU drop are **weighted by both-valid n**. Combined median confidence drop is the **median of the five attack median values**.

---

## Attack-wise results

| Attack | Total images | Eligible | Successful | ASR | Person elig. | Person succ. | Person ASR | Car elig. | Car succ. | Car ASR | Mean conf. drop | Median conf. drop | Mean IoU drop | Both-valid n |
|--------|-------------:|---------:|-----------:|----:|-------------:|-------------:|-----------:|----------:|----------:|--------:|----------------:|------------------:|--------------:|-------------:|
| attack_01 | 200 | 187 | 2 | 1.07% | 92 | 1 | 1.09% | 95 | 1 | 1.05% | 0.006666 | 0.000119 | 0.003081 | 185 |
| attack_02 | 250 | 225 | 2 | 0.89% | 113 | 1 | 0.88% | 112 | 1 | 0.89% | 0.009843 | 0.000140 | 0.001531 | 223 |
| attack_03 | 300 | 277 | 1 | 0.36% | 134 | 0 | 0.00% | 143 | 1 | 0.70% | 0.007127 | 0.000130 | -0.001163 | 276 |
| attack_04 | 350 | 314 | 4 | 1.27% | 151 | 1 | 0.66% | 163 | 3 | 1.84% | 0.020942 | 0.000204 | 0.001895 | 310 |
| attack_05 | 400 | 362 | 7 | 1.93% | 178 | 0 | 0.00% | 184 | 7 | 3.80% | 0.013950 | 0.000221 | 0.001469 | 355 |

---

## Combined totals (attack_01–attack_05)

| Metric | Value |
|--------|------:|
| Total images | 1500 |
| Eligible targets | 1365 |
| Successful attacks | 16 |
| Overall ASR | **1.17%** (16 / 1365 × 100) |

### Combined person

| Metric | Value |
|--------|------:|
| Eligible | 668 (= 92+113+134+151+178) |
| Successful | 3 (= 1+1+0+1+0) |
| Person ASR | **0.45%** (3 / 668 × 100) |

### Combined car

| Metric | Value |
|--------|------:|
| Eligible | 697 (= 95+112+143+163+184) |
| Successful | 13 (= 1+1+1+3+7) |
| Car ASR | **1.87%** (13 / 697 × 100) |

### Combined confidence / IoU drop (from existing both-valid summaries)

| Metric | Value |
|--------|------:|
| Mean confidence drop (weighted by both-valid n) | 0.012483 |
| Median confidence drop (median of attack medians) | 0.000140 |
| Mean IoU drop (weighted by both-valid n) | 0.001260 |
| Sum of both-valid n | 1349 |

---

## Brief interpretation

Across all five transfer evaluations, overall ASR on Faster R-CNN remains low (**1.17%**), with car targets showing somewhat higher transfer success (**1.87%**) than person targets (**0.45%**). This is consistent with a **transfer** setting: patches optimized against Hyper-YOLO do not strongly transfer to Faster R-CNN ResNet-50 FPN V2 under the same ASR protocol (clean eligibility; success = clean hit and patched miss; target-match IoU ≥ 0.50).
