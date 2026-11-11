# Reproducibility assessment — 11 November 2026

The original checkpoints, frozen splits, client assignments, predictions and
software environment have not been recovered. New evidence must not be cited
as the original manuscript experiment.

## What was executed

The public Kaggle version-4 archive was downloaded, decoded-image duplicates
were audited, and a group-safe split was generated using seed 340. Frozen
ResNet-18 features were extracted for all 21,165 images. The supplied FedAvg
and pooled probe training logic was run for 30 rounds/epochs, followed by a
single evaluation of validation-selected models and 1,000 class-stratified
duplicate-group bootstrap replicates. No tuning followed test evaluation.

The audit totals and split sizes agree with the manuscript. This does not
establish that the individual rows are assigned to the same original split.
The newly generated client class counts differ substantially from the
manuscript table. Without original artifacts, the exact cause cannot be isolated.

## Comparison with the manuscript

| Quantity | Manuscript | New run |
|---|---:|---:|
| FedAvg selected round | 26 | 29 |
| Pooled selected epoch | 29 | 27 |
| FedAvg accuracy | 0.670866 | 0.668346 |
| Pooled accuracy | 0.804094 | 0.779528 |
| FedAvg macro F1 | 0.373895 | 0.427257 |
| Pooled macro F1 | 0.780787 | 0.741351 |
| FedAvg COVID recall | 0 | 82/542 |
| FedAvg Viral Pneumonia recall | 0 | 0 |

The manuscript's claim of zero recall for BOTH COVID and Viral Pneumonia is
not reproduced by this new run. Pooled training still performs better under
this newly executed configuration. Do not replace original manuscript numbers
silently or claim exact reproduction.

## Improvements to the supplied package

The pipeline now exports labels and row identities, verifies feature ordering,
rejects cross-split duplicate leakage and conflicting duplicate labels, saves
client ownership, and implements paired class-stratified group bootstrapping.
Source tests include malformed feature identities and duplicate leakage.

The SGD optimizer still resets each epoch, as in the supplied implementation.
The alphabetic class order is COVID, Lung_Opacity, Normal, Viral Pneumonia.
These details are stored in run metadata. Hardware/library differences and
unavailable original source/splits remain material limits.

## Availability

Small authentic outputs are in `runs/reconstruction_2026-11-11/`. The separate
new-run archive contains source, feature rows, manifests, model weights,
predictions and artifact checksums, without raw image data. No persistent DOI
has been issued for the new run. Source and artifact checksums serve different
purposes and must not be confused with independent scientific validation.
