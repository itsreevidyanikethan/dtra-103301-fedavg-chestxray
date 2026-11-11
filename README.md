# DTRA-103301 — FedAvg and pooled chest-X-ray linear probes

This repository contains a new reconstruction of the supplied source package.
The original experiment archive was unavailable. A complete new run on the
public COVID-19 Radiography Database was executed on 11 November 2026.
**It does not reproduce the manuscript's numerical results.**

| Locked-test metric | New FedAvg | New pooled |
|---|---:|---:|
| Accuracy | 0.668346 | 0.779528 |
| Balanced accuracy | 0.434453 | 0.708994 |
| Macro F1 | 0.427257 | 0.741351 |
| Macro one-vs-rest AUC | 0.892838 | 0.926327 |

Selection used validation macro F1: FedAvg round 29 and pooled epoch 27.
See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) and the machine-readable
[comparison](runs/reconstruction_2026-11-11/manuscript_comparison.json).

## Run from the repository root

```bash
pip install -r requirements.txt
python src/download.py
python src/audit_and_split.py --help
python src/audit_and_split.py --root data/COVID-19_Radiography_Dataset
python src/extract_features.py --help
python src/extract_features.py --manifest manifest.csv
python src/experiment.py --manifest manifest.csv --features features.npz --threads 2
python make_checksums.py outputs --output CHECKSUMS.txt
python make_checksums.py --verify CHECKSUMS.txt
```

Feature extraction uses the frozen torchvision ResNet-18 ImageNet weights.
Inspect the entry-point help for optional batch and thread settings. Do not
change choices in response to test results. Use a new output directory for a
separate run; the experiment refuses to overwrite completed results.

## Evidence and interpretation

The new audit found 21,165 images, 21,106 unique decoded images, 44 duplicate
groups and 103 duplicate-related rows. The 14,814/3,176/3,175 split keeps exact
duplicates together. Patient-level independence cannot be established from
these identifiers. Four clients are synthetic; this is not a clinical or
multi-institutional federated validation.

`outputs/` receives the frozen manifest, client assignments, selected probe
weights, validation histories, labels, group identities, per-image predictions,
1,000 bootstrap draws and provenance. Small summaries are tracked under
`runs/reconstruction_2026-11-11/`; raw images are not redistributed.

The supplied code's sorted class order and resetting of SGD momentum at each
local/pooled epoch are retained and explicitly recorded. Protocol YAML is
reference documentation; the entry points define the executed settings.

## Check the code

```bash
python tests/smoke_test.py
python tests/test_reconstruction.py
python tests/test_checksums.py
python make_checksums.py --verify SOURCE_CHECKSUMS.txt
```

Source checksum verification checks distributed bytes, not scientific validity.
Code licence: MIT. Obtain the dataset from its official Kaggle source and follow
its own licence and attribution terms.
