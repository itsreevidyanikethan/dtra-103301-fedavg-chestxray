"""Simulated non-IID FedAvg vs pooled linear probe on frozen features.

Implements Algorithm 1 of the manuscript: Dirichlet(alpha=0.5) group-safe
client partition (seed 340), 30 FedAvg rounds (1 local epoch, size-weighted
aggregation) vs 30 pooled epochs, validation-macro-F1 checkpoint selection,
one locked-test evaluation with grouped stratified bootstrap (1,000 reps).
"""
import argparse
import csv
import json
import shutil
import hashlib
import platform
from importlib.metadata import version
from bootstrap import grouped_bootstrap
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score,
                             log_loss, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)

CLASSES = ["COVID", "Normal", "Lung_Opacity", "Viral Pneumonia"]  # fixed report order
SEED = 340


def l2_normalise(h, eps=1e-12):
    return h / (np.linalg.norm(h, axis=1, keepdims=True) + eps)


def load(manifest, features_npz):
    with open(manifest) as stream:
        rows = list(csv.DictReader(stream))
    archive = np.load(features_npz, allow_pickle=False)
    feats = l2_normalise(archive['features'].astype(np.float32))
    if feats.shape != (len(rows), 512) or not np.isfinite(feats).all():
        raise ValueError('Feature rows/dimensions do not match the manifest')
    if 'paths' in archive and archive['paths'].tolist() != [r['path'] for r in rows]:
        raise ValueError('Feature row order differs from the manifest')
    if 'content_hashes' in archive and archive['content_hashes'].tolist() != [r['content_hash'] for r in rows]:
        raise ValueError('Feature image identities differ from the manifest')
    cls_index = {c: i for i, c in enumerate(sorted({r["class"] for r in rows}))}
    y = np.array([cls_index[r["class"]] for r in rows])
    split = np.array([r["split"] for r in rows])
    group = np.array([r["group_id"] for r in rows])
    if set(split) != {'train', 'val', 'test'} or len(cls_index) != 4:
        raise ValueError('Expected four classes and train/val/test partitions')
    identities = defaultdict(set)
    for g, partition, label in zip(group, split, y): identities[g].add((partition, int(label)))
    for values in identities.values():
        if len({v[0] for v in values}) != 1: raise ValueError('Duplicate group crosses splits')
        if len({v[1] for v in values}) != 1: raise ValueError('Duplicate group has conflicting labels')
    return feats, y, split, group, cls_index


def dirichlet_clients(y, groups, n_clients=4, alpha=0.5, seed=SEED):
    """Assign TRAINING duplicate-groups to synthetic clients, Dirichlet by class."""
    rng = np.random.default_rng(seed)
    uniq_groups = defaultdict(list)          # group -> row indices
    for i, g in enumerate(groups):
        uniq_groups[g].append(i)
    client_of_row = np.full(len(y), -1)
    for cls in np.unique(y):
        gids = sorted({groups[i] for i in np.where(y == cls)[0]})
        rng.shuffle(gids)
        props = rng.dirichlet([alpha] * n_clients)
        bounds = (np.cumsum(props) * len(gids)).astype(int)
        start = 0
        for c, b in enumerate(bounds):
            for g in gids[start:b]:
                for i in uniq_groups[g]:
                    client_of_row[i] = c
            start = b
        for g in gids[start:]:
            for i in uniq_groups[g]:
                client_of_row[i] = n_clients - 1
    assert (client_of_row >= 0).all()
    return client_of_row


def make_probe(seed=SEED):
    torch.manual_seed(seed)
    return torch.nn.Linear(512, 4)


def epoch_update(probe, x, y, batch=256, lr=0.1, momentum=0.9, wd=1e-4, seed=0):
    opt = torch.optim.SGD(probe.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
    crit = torch.nn.CrossEntropyLoss()
    order = np.random.default_rng(seed).permutation(len(y))
    for i in range(0, len(y), batch):
        idx = order[i:i + batch]
        opt.zero_grad()
        crit(probe(x[idx]), y[idx]).backward()
        opt.step()


def macro_f1_of(probe, x, y):
    with torch.no_grad():
        pred = probe(x).argmax(1).numpy()
    return f1_score(y.numpy(), pred, average="macro")


def run(manifest, features_npz, out="outputs", bootstrap_reps=1000):
    import os
    os.makedirs(out, exist_ok=True)
    if os.path.exists(f"{out}/results.json"):
        raise ValueError("Results already exist; use a new output directory for another run")
    feats, y, split, group, cls_index = load(manifest, features_npz)
    shutil.copyfile(manifest, f'{out}/manifest.csv')
    json.dump({'class_to_index': cls_index, 'class_order': list(cls_index),
               'run_kind': 'new_reconstruction', 'original_artifacts_available': False,
               'seed': SEED, 'torch_threads': torch.get_num_threads(),
               'optimizer_state': 'SGD momentum resets on each local/pooled epoch, preserving supplied code',
               'python': platform.python_version(),
               'packages': {p: version(p) for p in ['numpy', 'scikit-learn', 'torch']},
               'manifest_sha256': hashlib.sha256(open(manifest, 'rb').read()).hexdigest(),
               'features_sha256': hashlib.sha256(open(features_npz, 'rb').read()).hexdigest()},
              open(f'{out}/run_metadata.json', 'w'), indent=2)
    tr, va, te = (split == s for s in ("train", "val", "test"))
    xt, yt = torch.tensor(feats[tr]), torch.tensor(y[tr])
    xv, yv = torch.tensor(feats[va]), torch.tensor(y[va])
    xe, ye = torch.tensor(feats[te]), torch.tensor(y[te])

    clients = dirichlet_clients(y[tr], group[tr])
    counts = {c: np.bincount(y[tr][clients == c], minlength=4).tolist() for c in range(4)}
    json.dump(counts, open(f"{out}/client_class_counts.json", "w"), indent=2)
    train_rows = np.flatnonzero(tr)
    with open(f'{out}/client_assignments.csv', 'w', newline='') as f:
        writer = csv.writer(f); writer.writerow(['manifest_row', 'group_id', 'label', 'client'])
        writer.writerows((int(i), str(group[i]), int(y[i]), int(c)) for i, c in zip(train_rows, clients))
    owners = defaultdict(set)
    for g, c in zip(group[tr], clients): owners[g].add(int(c))
    assert all(len(v) == 1 for v in owners.values())

    # ---- FedAvg: 30 rounds, 1 local epoch, size-weighted aggregation ----
    global_probe = make_probe()
    best = {"f1": -1, "round": -1, "state": None}
    hist_fed = []
    n_k = np.array([(clients == c).sum() for c in range(4)], dtype=np.float64)
    for rnd in range(1, 31):
        states = []
        for c in range(4):
            local = make_probe()
            local.load_state_dict(global_probe.state_dict())
            idx = np.where(clients == c)[0]
            epoch_update(local, xt[idx], yt[idx], seed=SEED * 1000 + rnd * 10 + c)
            states.append(local.state_dict())
        w = n_k / n_k.sum()
        agg = {k: sum(w[c] * states[c][k] for c in range(4)) for k in states[0]}
        global_probe.load_state_dict(agg)
        f1 = macro_f1_of(global_probe, xv, yv)
        hist_fed.append({"round": rnd, "val_macro_f1": f1})
        if f1 > best["f1"]:
            best = {"f1": f1, "round": rnd,
                    "state": {k: v.clone() for k, v in agg.items()}}
    print(f"FedAvg selected round {best['round']} val macro-F1 {best['f1']:.6f}")

    # ---- Pooled: 30 epochs, same hyperparameters ----
    pooled = make_probe()
    best_p = {"f1": -1, "epoch": -1, "state": None}
    hist_pool = []
    for ep in range(1, 31):
        epoch_update(pooled, xt, yt, seed=SEED * 2000 + ep)
        f1 = macro_f1_of(pooled, xv, yv)
        hist_pool.append({"epoch": ep, "val_macro_f1": f1})
        if f1 > best_p["f1"]:
            best_p = {"f1": f1, "epoch": ep,
                      "state": {k: v.clone() for k, v in pooled.state_dict().items()}}
    print(f"Pooled selected epoch {best_p['epoch']} val macro-F1 {best_p['f1']:.6f}")

    # ---- one locked-test evaluation ----
    results = {}
    probabilities = {}
    for name, state in (("fedavg", best["state"]), ("pooled", best_p["state"])):
        probe = make_probe()
        probe.load_state_dict(state)
        with torch.no_grad():
            prob = torch.softmax(probe(xe), 1).numpy()
        pred = prob.argmax(1)
        yn = ye.numpy()
        results[name] = {
            "accuracy": float((pred == yn).mean()),
            "balanced_accuracy": float(balanced_accuracy_score(yn, pred)),
            "macro_precision": float(precision_score(yn, pred, average="macro", zero_division=0)),
            "macro_recall": float(recall_score(yn, pred, average="macro", zero_division=0)),
            "macro_f1": float(f1_score(yn, pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(yn, pred, average="weighted", zero_division=0)),
            "mcc": float(matthews_corrcoef(yn, pred)),
            "log_loss": float(log_loss(yn, prob, labels=[0, 1, 2, 3])),
            "macro_ovr_auc": float(roc_auc_score(yn, prob, multi_class="ovr", average="macro")),
            "confusion_matrix": confusion_matrix(yn, pred).tolist(),
        }
        probabilities[name] = prob
        np.save(f"{out}/{name}_test_probs.npy", prob)
        torch.save(state, f"{out}/{name}_probe.pt")
    np.save(f'{out}/test_labels.npy', y[te])
    np.save(f'{out}/test_groups.npy', group[te])
    test_rows = np.flatnonzero(te)
    with open(f'{out}/test_predictions.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['manifest_row', 'group_id', 'true_label', 'model', 'prediction'] + [f'p_{c}' for c in cls_index])
        for name, prob in probabilities.items():
            writer.writerows([int(i), str(group[i]), int(y[i]), name, int(pr.argmax()), *pr.tolist()]
                             for i, pr in zip(test_rows, prob))
    boot, draws = grouped_bootstrap(y[te], probabilities, group[te], n_rep=bootstrap_reps)
    json.dump(boot, open(f'{out}/bootstrap_summary.json', 'w'), indent=2)
    with open(f'{out}/bootstrap_replicates.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(draws[0])); writer.writeheader(); writer.writerows(draws)
    json.dump({"fedavg_history": hist_fed, "pooled_history": hist_pool,
               "selected": {"fedavg_round": best["round"], "pooled_epoch": best_p["epoch"]},
               "locked_test": results},
              open(f"{out}/results.json", "w"), indent=2)
    print(json.dumps(results, indent=2)[:800])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifest.csv")
    ap.add_argument("--features", default="features.npz")
    ap.add_argument('--out', default='outputs')
    ap.add_argument('--threads', type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    run(a.manifest, a.features, out=a.out)
