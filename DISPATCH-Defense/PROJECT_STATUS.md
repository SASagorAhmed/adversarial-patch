# DISPATCH-Defense — PROJECT STATUS

```text
[x] Root created
[x] CompVis cloned
[x] Checkpoint downloaded
[x] Environment created
[x] Dependencies installed
[x] GPU verified
[x] LDM loads
[x] Official inpainting smoke test passes
[x] Checkerboard mask generator works
[x] Two-pass regeneration works
[x] Difference map works
[x] KMeans mask works
[x] Rectification works
[x] DISPATCH one-image smoke test passes
[x] Safety audit passes
[x] Mitigation folder style implemented (`mitigations/mitigation_XX`)
[x] mitigation_01 completed (attack_02, 15+15)
[ ] Ready for mitigation_02 / further evaluations
```

## mitigation_01 (completed)

- Source: `attack_02` (independent selection; not old mitigation subsets)
- Layout: previous-mitigation-style under `DISPATCH-Defense\mitigations\mitigation_01\`
- DRR: **13.33%** (2/15) | CPR: **86.67%** (13/15) | CRR: **13.33%** (2/15)
- Recovered: `000000127263` (car), `000000393093` (car)
- Regressed controls: `000000295809` (car), `000000508602` (car)
- `clean_pixels_used_for_restoration = false`
- `known_patch_location_used_by_defense = false`
