# DTRA-103301 — Duplicate-Aware FedAvg vs Pooled Linear Probe: Reproduction Package

Code for *"Duplicate-Aware Evaluation of Simulated Non-IID FedAvg and Pooled
Linear Probes on Frozen ResNet-18 Features for Four-Class Chest X-Ray
Classification"* (Digital Technologies Research and Applications, Article ID
103301). The study is a bounded failure-mode analysis: under one recorded
non-IID configuration (Dirichlet α = 0.5, four synthetic clients, seed 340),
simulated FedAvg collapsed to two predicted classes while a matched pooled
linear probe did not.

## Registered protocol
`configs/protocol.yaml` freezes every choice: duplicate-group-safe 70/15/15
split, frozen ImageNet-1K-V1 ResNet-18 features (512-d, per-image L2
normalisation), linear 512→4 probe, SGD (lr 0.1, momentum 0.9, wd 1e-4,
batch 256), 30 FedAvg rounds × 1 local epoch with sample-size-weighted
aggregation vs 30 pooled epochs, validation-macro-F1 selection, one locked-test
evaluation, 1,000-replicate grouped stratified bootstrap.

## Dataset (not redistributed)
Download the COVID-19 Radiography Database (version 4) from Kaggle:
https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database
and extract it so each class folder (COVID, Lung_Opacity, Normal,
Viral Pneumonia) is under one root directory.

## Reproduce
```bash
pip install -r requirements.txt
python -I src/audit_and_split.py --root /path/to/COVID-19_Radiography_Dataset
python -I src/extract_features.py            # frozen ResNet-18, CPU, batch 32
python -I src/experiment.py                  # FedAvg vs pooled + locked test
python -I tests/smoke_test.py                # fast synthetic end-to-end check
```

## Outputs
`outputs/` receives the client class counts, probe checkpoints, per-round and
per-epoch validation histories, locked-test probabilities, confusion matrices
and `results.json` with all summary measures.

## Honest-reproduction notes
- The audit/split script regenerates a duplicate-group-safe split with seed
  340; because the greedy stratified assignment depends on the RNG stream, use
  the archived `manifest.csv` from the paper's supplement when exact row-level
  agreement with the published split is required.
- Different hardware or library versions produce slightly different trained
  probes; the frozen protocol, not bit-exact outputs, is what this package
  reproduces. The qualitative failure mode (zero recall for two classes under
  FedAvg) is the registered finding under this configuration.

## License
Code: MIT (see LICENSE). The images remain under the Kaggle dataset's own
terms and are not redistributed here.
