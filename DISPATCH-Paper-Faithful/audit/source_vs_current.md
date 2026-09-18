# Official source vs current DISPATCH-Defense

Official repository cloned to `reference_source/DisPatch`
- remote: https://github.com/MaJinWakeUp/DisPatch
- commit: `80c59b9b4a6c4060107a8b4485d3885128c2ae81` (2026-08-14, "Fix dataset download link in README")
- Canonical inference: `LDM/scripts/d3_inference.py`
- Adaptive-attack wrapper (not used for Automatic): `attackers/d3_wrapper.py` (defaults inpaint_size=256, num_grids=16, steps=50 — **not** the paper defaults)

Checkpoint: official README says download CompVis `inpainting_big/last.ckpt` into `LDM/models/ldm/inpainting_big/`. Clone does **not** include the weights. Current project already has that official CompVis file under DISPATCH-Defense `third_party/latent-diffusion/models/ldm/inpainting_big/last.ckpt`. Official and current YAML configs match the CompVis inpainting_big schema (`in_channels: 7`, concat mask). **LDM type: MATCH. Byte-identity of ckpt vs authors' file: UNKNOWN (same documented URL).**

## Code path

Official `d3_inference.py`:

1. `Image.open.convert("RGB")` → `resize((512,512))` (PIL default BICUBIC) → `ToTensor()` [0,1] RGB
2. Batch size 2: image repeated; masks `cat([mask0, mask1])`
3. `masked_image = (1-masks)*image` then `*2-1` to [-1,1]
4. Encode masked image, concat downsampled mask, **one** `sampler.sample(S=5, batch_size=2)`
5. `generated = mask[0]*predicted[0] + mask[1]*predicted[1]`  (predicted = full LDM decode in [0,1])
6. L2: `torch.norm(orig-gen, p=2, dim=0)` on [0,1] CHW
7. `GaussianBlur(kernel_size=min(15, size//num_grids-1), sigma=1.0)` → **15** at 512/32
8. KMeans k=2, `n_init='auto'`, **no random_state**; `with_loc=False` so **scalar D only** (spatial features exist but unused)
9. Higher centroid[0] → adversarial
10. `shape_completion`: `cv2.MORPH_OPEN` 3×3 (**dilate is commented out**)
11. `rectified = adv_mask*generated + (1-adv_mask)*input` at 512
12. Resize rectified and saved mask to original with `Image.resize` (BICUBIC on RGB; also on mask PNG — eval-unsafe)

Current Automatic (`ldm_inpaint_adapter.py`, `run_automatic.py` in testing mit02):

1. LANCZOS resize to 512, uint8 RGB NumPy
2. **Two sequential** `inpaint()` calls, each `seed_everything(20260817)` → **same RNG start for both passes**
3. Each pass returns CompVis composite `(1-mask)*I + mask*pred`; then `I_tilde = pass0*m0 + pass1*m1` (binary-equivalent to official predicted compose)
4. L2 on [0,1] RGB HWC
5. OpenCV Gaussian `(5,5)` sigma 1.0
6. KMeans k=2, `n_init=10`, `random_state=seed` on scalar D
7. argmax centroid → adversarial
8. **No morphology**
9. Rectify at 512 then LANCZOS back to original
10. Predicted mask NEAREST to original for eval

## Differences that can affect oversized masks

1. **MAJOR — sequential same-seed vs batched independent noise.** Current reseeds before every pass, so both checkerboard halves may share identical DDIM noise. Official samples batch_size=2 once (two noise tensors). Shared noise can imprint a global checkerboard mismatch vs I, inflating the high-difference cluster.
2. **MAJOR vs paper / present in official — MORPH_OPEN 3×3** after KMeans. Current omits it. Removes small white specks; **will not shrink a huge blob**.
3. **MINOR — Gaussian kernel 5 vs 15**, same sigma 1.0. Effective support of N(0,1) is ~3 pixels; kernels 5 and 15 with sigma=1 are nearly equivalent.
4. **MINOR — LANCZOS vs BICUBIC** resize.
5. **MINOR — KMeans n_init/seed.**
6. **MINOR — official saves mask with BICUBIC**; current eval uses NEAREST (better for IoU). Reference eval uses NEAREST to match current protocol.

## Confirmed matches

- N=32, 512, DDIM 5, CompVis inpainting_big, 1=inpaint
- Complementary checkerboard `(i+j)%2==0` white in m0; at 512/32 both constructions tile 16×16 cells with full coverage
- L2 on [0,1] RGB, k=2, higher centroid adversarial
- Rectify equation
- Localization at **512**, then resize output to original
- No GT/patch oracle in Automatic path (`with_loc=False`)

## Mask semantics numerical check (CompVis)

`masked_image = (1-mask)*image` then scale to [-1,1].
White (1) → hole (0 in [0,1], -1 after scale) → regenerated.
Black (0) → keep image.
Current adapter copies this. **Not reversed. Not a CRITICAL mask-semantics bug.**
