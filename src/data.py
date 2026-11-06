"""MVTec AD loading.

Split policy (to respect "no test labels for training / tuning / threshold"):
  train/good  -> 85 % "fit" (used to learn normal appearance)
              -> 15 % "val" (held-out NORMAL images, used only to set the threshold)
  test/*      -> evaluation only (labels + masks never touch the models)
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from glob import glob

import numpy as np
from PIL import Image

DATA_ROOT = os.environ.get(
    "MVTEC_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "mvtec_ad"))
OBJECTS = {"bottle", "cable", "capsule", "hazelnut", "metal_nut", "pill", "screw",
           "toothbrush", "transistor", "zipper"}
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class Sample:
    path: str
    label: int          # 0 normal, 1 anomalous
    defect: str         # "good" or defect type
    mask_path: str | None


def list_split(category: str, val_frac: float = 0.15, seed: int = 0):
    root = os.path.join(DATA_ROOT, category)
    train = sorted(glob(os.path.join(root, "train", "good", "*.png")))
    rng = random.Random(seed)
    idx = list(range(len(train)))
    rng.shuffle(idx)
    n_val = max(1, int(round(val_frac * len(train))))
    val = [Sample(train[i], 0, "good", None) for i in sorted(idx[:n_val])]
    fit = [Sample(train[i], 0, "good", None) for i in sorted(idx[n_val:])]

    test = []
    for d in sorted(os.listdir(os.path.join(root, "test"))):
        for p in sorted(glob(os.path.join(root, "test", d, "*.png"))):
            if d == "good":
                test.append(Sample(p, 0, d, None))
            else:
                stem = os.path.splitext(os.path.basename(p))[0]
                m = os.path.join(root, "ground_truth", d, f"{stem}_mask.png")
                test.append(Sample(p, 1, d, m))
    return fit, val, test


def rel(path: str) -> str:
    """Path relative to the dataset root (machine-independent, used in saved CSVs)."""
    return os.path.relpath(path, DATA_ROOT)


def load_image(path: str, size: int) -> np.ndarray:
    """RGB float32 in [0,1], HxWx3, resized to size x size (whole image kept, no crop,
    so defects near the border are not cut away)."""
    im = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


def load_mask(sample: Sample, size: int) -> np.ndarray:
    if sample.mask_path is None:
        return np.zeros((size, size), dtype=np.uint8)
    m = Image.open(sample.mask_path).convert("L").resize((size, size), Image.NEAREST)
    return (np.asarray(m) > 127).astype(np.uint8)


def load_batch(samples, size: int, imagenet_norm: bool):
    x = np.stack([load_image(s.path, size) for s in samples])  # N,H,W,3
    if imagenet_norm:
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(x.transpose(0, 3, 1, 2))      # N,3,H,W
