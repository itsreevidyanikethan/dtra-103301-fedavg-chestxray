# Reproducibility status for DTRA-103301

The source package has been checked on synthetic data. Exact recovery of the
reported experiment has not been established. No real-dataset training or
locked-test evaluation was performed for this maintenance update.

## Original evidence required

| Evidence | Status in this repository |
|---|---|
| Executed source and exact original software environment | Original-run provenance not verified; current source and validation environment are available |
| Frozen split manifest and decoded-image duplicate groups | Missing |
| Frozen synthetic-client assignment | Missing |
| Feature matrix, row-order metadata and feature-generation record | Missing |
| Selected FedAvg and pooled checkpoints | Missing |
| Validation histories and original selected round/epoch evidence | Missing |
| Locked-test labels, row/group identities and probability arrays | Missing |
| Metrics, 1,000 grouped bootstrap replicates, tables and figure scripts | Original archive missing |
| Original artifact checksums and persistent archive DOI | Missing |

## Implementation limits identified

The present experiment script does not implement the 1,000-replicate grouped
bootstrap described in its module description and protocol. It writes test
probabilities but does not export the accompanying labels, row/group identities
or complete row-level client assignments. Its SGD optimizer is recreated at
each call to `epoch_update`, including every pooled epoch; equivalence with the
original optimizer-state handling must be checked against the executed source.
The split generator regenerates a split and cannot establish identity with the
paper's missing frozen manifest. Protocol YAML is documentation: it is not read
by these training scripts, so changing it alone does not change execution.

These scientific implementation choices were not changed in the maintenance
update. Synthetic tests cannot establish the published zero-recall finding.
The reported original environment also differs from the current test environment.

## Completion procedure

Recover the original experiment directory, including the executed scripts and
raw machine-readable outputs. Check row order, class-index mapping, group/split
separation, selected checkpoints, and the reported metrics against that archive.
Resolve any implementation differences before claiming exact reproduction.
Then generate and verify artifact checksums and archive the permitted files with
a persistent identifier. Update the manuscript's availability statement only
after the promised files are actually retrievable.
