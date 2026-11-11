"""Smoke test with synthetic features (no dataset needed)."""
import csv
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from experiment import run  # noqa: E402


def main():
    rng = np.random.default_rng(0)
    n = 400
    classes = ["COVID", "Normal", "Lung_Opacity", "Viral Pneumonia"]
    with tempfile.TemporaryDirectory() as d:
        man = os.path.join(d, "manifest.csv")
        with open(man, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["path", "class", "content_hash", "group_id", "split"])
            w.writeheader()
            for i in range(n):
                split = "train" if (i // 4) % 10 < 7 else ("val" if (i // 4) % 10 < 9 else "test")
                w.writerow({"path": f"img{i}.png", "class": classes[i % 4],
                            "content_hash": f"h{i}", "group_id": f"g{i}",
                            "split": split})
        feats = rng.normal(size=(n, 512)).astype(np.float32)
        # make classes separable so metrics are non-degenerate
        for i in range(n):
            feats[i, (i % 4) * 8:(i % 4) * 8 + 8] += 2.0
        np.savez(os.path.join(d, "features.npz"), features=feats)
        out = os.path.join(d, "outputs")
        run(man, os.path.join(d, "features.npz"), out=out, bootstrap_reps=20)
        assert os.path.exists(os.path.join(out, "results.json"))
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
