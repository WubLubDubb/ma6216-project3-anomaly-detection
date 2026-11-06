"""Label-free model selection, step 1: synthetic defects on held-out NORMAL images.

The bonus variants were first compared on the test set. To choose among them
without any test label, we build a second validation set: every held-out
normal validation image (the 15 % of train/good never used for fitting) gets
synthetic defects of three published kinds. Generator settings are the
published defaults and were fixed before the run, which was then done once:

  cutpaste : CutPaste (Li et al., CVPR 2021). A rectangle of 2-15 % of the image
             area, aspect ratio in [0.3, 3.3], cut from the same image, colour
             jittered (0.1 brightness/contrast/saturation/hue), pasted elsewhere.
  scar     : CutPaste-scar. A thin 2-16 x 10-25 px strip (at 256 px), rotated
             by -45..45 degrees, colour jittered, pasted elsewhere.
  draem    : DRAEM (Zavrtanik et al., ICCV 2021). A thresholded (> 0.5) Perlin-noise
             mask (scale 2^0..2^5 per axis, rotated -90..90 degrees) is filled with a
             foreign texture, blended with the image with weight beta in [0, 0.8].
             DRAEM takes the texture from the DTD dataset; we take it from the
             normal training images of the other four categories (no extra
             download), with a random 90-degree rotation and gamma/brightness jitter.

On object categories the defect is placed on the part (the label-free Otsu
foreground of foreground.py, without dilation): a real defect is on the
product, not on the background. Each validation image gets 2 defects of each
kind, generated at the original resolution and saved as PNG with its mask.

  python src/synthetic.py                 # all 5 categories -> data/mvtec_synth/<cat>/
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import ndimage as ndi
from torchvision.transforms import ColorJitter

sys.path.insert(0, os.path.dirname(__file__))
from data import DATA_ROOT, OBJECTS, Sample, list_split, rel  # noqa: E402
from foreground import foreground_mask  # noqa: E402

SYNTH_ROOT = os.environ.get(
    "SYNTH_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "mvtec_synth"))
CATS = ["bottle", "metal_nut", "screw", "carpet", "leather"]
TYPES = ("cutpaste", "scar", "draem")
REPEATS = 2
REF = 256  # sizes in the papers are given for 256 px images


# ---------------------------------------------------------------- Perlin noise (DRAEM's rand_perlin_2d_np)
def _perlin(shape, res, rng):
    def fade(t):
        return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    ang = 2 * math.pi * rng.random((res[0] + 1, res[1] + 1))
    grads = np.stack((np.cos(ang), np.sin(ang)), axis=-1)

    def tile(s1, s2):
        return np.repeat(np.repeat(grads[s1[0]:s1[1], s2[0]:s2[1]], d[0], axis=0), d[1], axis=1)

    def dot(g, shift):
        return (np.stack((grid[:shape[0], :shape[1], 0] + shift[0], grid[:shape[0], :shape[1], 1] + shift[1]),
                         axis=-1) * g[:shape[0], :shape[1]]).sum(axis=-1)
    n00 = dot(tile([0, -1], [0, -1]), [0, 0])
    n10 = dot(tile([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile([0, -1], [1, None]), [0, -1])
    n11 = dot(tile([1, None], [1, None]), [-1, -1])
    t = fade(grid[:shape[0], :shape[1]])
    lerp = lambda a, b, w: (b - a) * w + a  # noqa: E731
    return math.sqrt(2) * lerp(lerp(n00, n10, t[..., 0]), lerp(n01, n11, t[..., 0]), t[..., 1])


# ---------------------------------------------------------------- helpers
def _region(path, category):
    """Where a defect may be centred, at REF resolution: the part (objects) or anywhere (textures)."""
    if category not in OBJECTS:
        return np.ones((REF, REF), bool)
    core = ndi.binary_erosion(foreground_mask(path, REF, dilate_px=0), iterations=3)
    return core if core.sum() > 50 else foreground_mask(path, REF, dilate_px=0)


def _centre(region, rng, scale):
    ys, xs = np.nonzero(region)
    i = rng.integers(len(ys))
    return (ys[i] + rng.random()) * scale, (xs[i] + rng.random()) * scale


def _jitter(patch, seed):
    torch.manual_seed(seed)
    im = Image.fromarray((patch * 255 + 0.5).astype(np.uint8))
    return np.asarray(ColorJitter(0.1, 0.1, 0.1, 0.1)(im), np.float32) / 255


def cutpaste(img, region, rng, seed):
    H, W = img.shape[:2]
    area = rng.uniform(0.02, 0.15) * H * W
    aspect = math.exp(rng.uniform(math.log(0.3), math.log(1 / 0.3)))
    cw = int(min(W - 1, round(math.sqrt(area * aspect))))
    ch = int(min(H - 1, round(math.sqrt(area / aspect))))
    sy, sx = rng.integers(0, H - ch + 1), rng.integers(0, W - cw + 1)
    patch = _jitter(img[sy:sy + ch, sx:sx + cw], seed)
    cy, cx = _centre(region, rng, H / REF)
    dy, dx = int(np.clip(cy - ch / 2, 0, H - ch)), int(np.clip(cx - cw / 2, 0, W - cw))
    out, m = img.copy(), np.zeros((H, W), bool)
    out[dy:dy + ch, dx:dx + cw] = patch
    m[dy:dy + ch, dx:dx + cw] = True
    return out, m


def scar(img, region, rng, seed):
    H, W = img.shape[:2]
    s = W / REF
    w = max(1, int(round(rng.uniform(2, 16) * s)))
    h = max(1, int(round(rng.uniform(10, 25) * s)))
    angle = rng.uniform(-45, 45)
    sy, sx = rng.integers(0, H - h + 1), rng.integers(0, W - w + 1)
    patch = Image.fromarray((_jitter(img[sy:sy + h, sx:sx + w], seed) * 255 + 0.5).astype(np.uint8))
    patch = np.asarray(patch.rotate(angle, resample=Image.BILINEAR, expand=True), np.float32) / 255
    pm = np.asarray(Image.new("L", (w, h), 255).rotate(angle, resample=Image.BILINEAR, expand=True)) > 127
    ph, pw = pm.shape
    cy, cx = _centre(region, rng, H / REF)
    dy, dx = int(np.clip(cy - ph / 2, 0, H - ph)), int(np.clip(cx - pw / 2, 0, W - pw))
    out, m = img.copy(), np.zeros((H, W), bool)
    win = out[dy:dy + ph, dx:dx + pw]
    win[pm] = patch[pm]
    m[dy:dy + ph, dx:dx + pw] = pm
    return out, m


def draem(img, region, rng, foreign_paths, grey):
    H, W = img.shape[:2]
    for _ in range(100):
        sx, sy = 2 ** int(rng.integers(0, 6)), 2 ** int(rng.integers(0, 6))
        noise = ndi.rotate(_perlin((REF, REF), (sx, sy), rng), rng.uniform(-90, 90), reshape=False, order=1,
                           mode="reflect")
        m = (noise > 0.5) & region
        if m.sum() >= 16:
            break
    m = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((W, H), Image.NEAREST)) > 127
    src = Image.open(foreign_paths[rng.integers(len(foreign_paths))]).convert("RGB").resize((W, H), Image.BILINEAR)
    a = np.rot90(np.asarray(src, np.float32) / 255, int(rng.integers(4))).copy()
    a = np.clip(a ** rng.uniform(0.5, 2.0) * rng.uniform(0.8, 1.2), 0, 1)
    if grey:
        a = np.repeat(a.mean(2, keepdims=True), 3, 2)
    beta = rng.uniform(0, 0.8)
    mm = m[..., None]
    return img * ~mm + mm * ((1 - beta) * a + beta * img), m


# ---------------------------------------------------------------- dataset
def list_synthetic(category):
    """Samples of the synthetic validation set (label 1, defect = generator type)."""
    df = pd.read_csv(os.path.join(SYNTH_ROOT, category, "index.csv"))
    return [Sample(os.path.join(SYNTH_ROOT, category, f), 1, t, os.path.join(SYNTH_ROOT, category, m))
            for f, t, m in zip(df["file"], df["type"], df["mask"])], df


def generate(category):
    _, val, _ = list_split(category)
    foreign = [s.path for c in CATS if c != category for s in list_split(c)[0]]
    out_dir = os.path.join(SYNTH_ROOT, category)
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for i, s in enumerate(val):
        im = Image.open(s.path)
        grey = im.mode == "L"
        img = np.asarray(im.convert("RGB"), np.float32) / 255
        region = _region(s.path, category)
        for ti, t in enumerate(TYPES):
            for r in range(REPEATS):
                seed = int(np.random.SeedSequence([CATS.index(category), i, ti, r]).generate_state(1)[0])
                rng = np.random.default_rng(seed)
                if t == "cutpaste":
                    o, m = cutpaste(img, region, rng, seed)
                elif t == "scar":
                    o, m = scar(img, region, rng, seed)
                else:
                    o, m = draem(img, region, rng, foreign, grey)
                o = Image.fromarray((np.clip(o, 0, 1) * 255 + 0.5).astype(np.uint8))
                if grey:
                    o = o.convert("L")
                stem = f"{i:03d}_{t}_{r}"
                o.save(os.path.join(out_dir, stem + ".png"), compress_level=3)
                Image.fromarray(m.astype(np.uint8) * 255).save(os.path.join(out_dir, stem + "_mask.png"))
                m256 = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((REF, REF), Image.NEAREST)) > 127
                rows.append({"file": stem + ".png", "mask": stem + "_mask.png", "source": rel(s.path),
                             "val_index": i, "type": t, "repeat": r, "seed": seed,
                             "defect_frac": float(m256.mean())})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "index.csv"), index=False)
    print(f"{category}: {len(val)} normal val images -> {len(df)} synthetic "
          f"(median defect area {100 * df.defect_frac.median():.2f} % of the image)", flush=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cats", nargs="+", default=CATS)
    args = ap.parse_args()
    print("dataset:", DATA_ROOT, "-> synthetic:", SYNTH_ROOT)
    for c in args.cats:
        generate(c)


if __name__ == "__main__":
    main()
