"""Frozen ImageNet ResNet-18 features: one 512-d vector per image (CPU)."""
import argparse
import csv

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
    feats = np.empty((len(rows), 512), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(rows), batch):
            ims = [tf(Image.open(r["path"]).convert("RGB")) for r in rows[i:i + batch]]
            feats[i:i + batch] = model(torch.stack(ims)).numpy()
            if i % (batch * 50) == 0:
                print(f"{i}/{len(rows)}")
    np.savez(out, features=feats)
    print(f"saved {feats.shape} to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifest.csv")
    a = ap.parse_args()
    main(a.manifest)
