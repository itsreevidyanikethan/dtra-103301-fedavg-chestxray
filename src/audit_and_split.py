"""Decoded-pixel duplicate audit and duplicate-group-safe stratified split.

Expects the Kaggle 'COVID-19 Radiography Database' extracted so that each class
folder contains its images, e.g. data/COVID/images/*.png (see README).
Produces manifest.csv with: path, class, content_hash, group_id, split.
"""
import argparse
import csv
import hashlib
import os
import json
from collections import defaultdict

import numpy as np
from PIL import Image

CLASSES = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]


def content_hash(path):
    """SHA-256 of DECODED pixels (not file bytes), as registered."""
    with Image.open(path) as im:
        arr = np.asarray(im.convert("L"))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def collect(root):
    rows = []
    for cls in CLASSES:
        folder = os.path.join(root, cls, "images")
        if not os.path.isdir(folder):
            folder = os.path.join(root, cls)
        for name in sorted(os.listdir(folder)):
            if name.lower().endswith((".png", ".jpg", ".jpeg")):
                rows.append({"path": os.path.join(folder, name), "class": cls})
    return rows


def main(root, out_csv="manifest.csv", seed=340):
    rows = collect(root)
    print(f"found {len(rows)} images (expected 21,165)")
    groups = defaultdict(list)
    for r in rows:
        r["content_hash"] = content_hash(r["path"])
        groups[r["content_hash"]].append(r)
    dup = {h: g for h, g in groups.items() if len(g) > 1}
    print(f"unique decoded images: {len(groups)}; duplicate groups: {len(dup)}; "
          f"duplicate rows: {sum(len(g) for g in dup.values())}")

    conflicts = {h: g for h, g in groups.items() if len({r['class'] for r in g}) > 1}
    if conflicts:
        raise ValueError(f'{len(conflicts)} duplicate groups have conflicting labels; resolve a documented policy before training')
    # one representative class per group (verified label-consistent copies)
    rng = np.random.default_rng(seed)
    group_items = list(groups.items())
    rng.shuffle(group_items)
    split_counts = defaultdict(lambda: defaultdict(int))
    targets = {"train": 0.70, "val": 0.15, "test": 0.15}
    totals = defaultdict(int)
    for h, g in group_items:
        totals[g[0]["class"]] += len(g)
    assigned = defaultdict(lambda: defaultdict(int))
    for h, g in group_items:
        cls, n = g[0]["class"], len(g)
        # greedy fill: send the whole group to the split furthest below target
        deficits = {s: targets[s] * totals[cls] - assigned[cls][s] for s in targets}
        split = max(deficits, key=deficits.get)
        assigned[cls][split] += n
        for r in g:
            r["split"] = split
            r["group_id"] = h
            split_counts[split][cls] += 1

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "class", "content_hash", "group_id", "split"])
        w.writeheader()
        for h, g in groups.items():
            for r in g:
                w.writerow(r)
    for s in ("train", "val", "test"):
        print(s, dict(split_counts[s]), "total", sum(split_counts[s].values()))
    # verify no group crosses splits
    by_group = defaultdict(set)
    for h, g in groups.items():
        for r in g:
            by_group[h].add(r["split"])
    assert all(len(v) == 1 for v in by_group.values()), "duplicate group crossed splits!"
    report = {'total_rows': len(rows), 'unique_decoded_images': len(groups),
              'duplicate_groups': len(dup), 'duplicate_rows': sum(map(len, dup.values())),
              'conflicting_label_groups': len(conflicts), 'group_crossings': 0,
              'seed': seed, 'split_class_counts': {s: dict(v) for s, v in split_counts.items()},
              'duplicate_members': {h: [r['path'] for r in g] for h, g in dup.items()},
              'provenance': 'newly generated split; original frozen split unavailable'}
    with open(os.path.splitext(out_csv)[0] + '_audit.json', 'w') as f:
        json.dump(report, f, indent=2)
    print("group-safety verified: 0 duplicate groups cross splits")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="extracted dataset root")
    a = ap.parse_args()
    main(a.root)
