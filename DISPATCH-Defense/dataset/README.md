# Dataset / manifests

Manifests and metadata for independent DISPATCH evaluation are stored here.

- `manifests/` — selected image lists for future runs
- `metadata/` — auxiliary metadata copies (never overwrite frozen attack CSVs)

Source attacked images are READ from `Adversarial-Patch-Experiment` and COPIED into
experiment `source_images/` before processing.

Do NOT reuse old mitigation_01 / mitigation_02 subsets.
