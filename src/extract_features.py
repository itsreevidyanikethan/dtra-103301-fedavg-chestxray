"""Frozen ImageNet ResNet-18 features: one 512-d vector per image (CPU)."""
import argparse
import csv
import json
import hashlib
import time
from importlib.metadata import version

import numpy as np
import torch
from PIL import Image


def main(manifest="manifest.csv", out="features.npz", batch=32):
    from torchvision.models import resnet18, ResNet18_Weights
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    tf = weights.transforms()  # resize 224, ImageNet normalisation

    rows = list(csv.DictReader(open(manifest)))
    started = time.monotonic()
    feats = np.empty((len(rows), 512), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(rows), batch):
            ims = [tf(Image.open(r["path"]).convert("RGB")) for r in rows[i:i + batch]]
            feats[i:i + batch] = model(torch.stack(ims)).numpy()
            if i % (batch * 50) == 0:
                print(f"{i}/{len(rows)}")
    np.savez(out, features=feats, paths=np.array([r['path'] for r in rows]),
             content_hashes=np.array([r['content_hash'] for r in rows]))
    metadata = {'source': 'new reconstruction run', 'weights': str(weights),
                'transform': str(tf), 'shape': list(feats.shape), 'batch_size': batch,
                'device': 'cpu', 'torch_threads': torch.get_num_threads(),
                'elapsed_seconds': time.monotonic() - started,
                'torch': version('torch'), 'torchvision': version('torchvision'),
                'manifest_sha256': hashlib.sha256(open(manifest, 'rb').read()).hexdigest(),
                'feature_archive_sha256': hashlib.sha256(open(out, 'rb').read()).hexdigest()}
    with open(out + '.metadata.json', 'w') as f: json.dump(metadata, f, indent=2)
    print(f"saved {feats.shape} to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifest.csv")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    main(a.manifest)
