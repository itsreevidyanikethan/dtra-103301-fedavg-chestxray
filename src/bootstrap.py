"""Class-stratified duplicate-group bootstrap for a new reconstruction run."""
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score


def scores(y, probabilities):
    pred = probabilities.argmax(1)
    return {
        'accuracy': float(accuracy_score(y, pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
        'macro_f1': float(f1_score(y, pred, labels=[0, 1, 2, 3], average='macro', zero_division=0)),
        'macro_ovr_auc': float(roc_auc_score(y, probabilities, multi_class='ovr', average='macro')),
    }


def grouped_bootstrap(y, probabilities, groups, n_rep=1000, seed=340):
    """Resample whole groups within each class, jointly for all model arms.

    Input groups must have one true class each. Repeated groups retain all
    member rows, including multiplicity when sampled more than once.
    """
    y = np.asarray(y); groups = np.asarray(groups)
    if len(y) != len(groups) or set(np.unique(y)) != {0, 1, 2, 3}:
        raise ValueError('All four classes and aligned group identifiers are required')
    index = {g: np.flatnonzero(groups == g) for g in np.unique(groups)}
    strata = {c: [] for c in range(4)}
    for g, rows in index.items():
        classes = np.unique(y[rows])
        if len(classes) != 1: raise ValueError('Conflicting labels within a duplicate group')
        strata[int(classes[0])].append(g)
    rng = np.random.default_rng(seed)
    samples = []
    for b in range(n_rep):
        chosen = [g for c in range(4) for g in rng.choice(strata[c], len(strata[c]), replace=True)]
        rows = np.concatenate([index[g] for g in chosen])
        for name, prob in probabilities.items():
            samples.append({'replicate': b, 'model': name, **scores(y[rows], prob[rows])})
    intervals = {}
    for name in probabilities:
        values = [r for r in samples if r['model'] == name]
        intervals[name] = {k: np.percentile([r[k] for r in values], [2.5, 97.5]).tolist()
                           for k in ('accuracy', 'balanced_accuracy', 'macro_f1', 'macro_ovr_auc')}
    return {'method': 'percentile, class-stratified duplicate-group bootstrap',
            'seed': seed, 'replicates': n_rep, 'ci95': intervals}, samples
