"""Why the autoencoder fails: input / reconstruction / 1-SSIM residual for normal and defective images."""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from ae import AEDetector  # noqa: E402
from data import list_split, load_mask  # noqa: E402

R = os.path.join(os.path.dirname(__file__), "..", "results")
PICKS = {"bottle": ["good", "broken_large"], "metal_nut": ["good", "flip", "color"],
         "carpet": ["good", "hole", "color"]}


def main():
    rows = [(c, d) for c, ds in PICKS.items() for d in ds]
    fig, axes = plt.subplots(len(rows), 3, figsize=(6.6, 2.2 * len(rows)), dpi=150)
    cache = {}
    for r, (cat, defect) in enumerate(rows):
        if cat not in cache:
            det = AEDetector()
            det.model.load_state_dict(torch.load(os.path.join(R, "ae", cat, "ae.pt"), map_location="cuda"))
            cache[cat] = det
        _, _, test = list_split(cat)
        s = next(x for x in test if x.defect == defect)
        m, rec = cache[cat].predict([s], return_recon=True)
        from data import load_image
        img = load_image(s.path, 256)
        axes[r, 0].imshow(img); axes[r, 0].set_ylabel(f"{cat}\n{defect}", fontsize=8)
        gt = load_mask(s, 256)
        if gt.any():
            axes[r, 0].contour(gt, levels=[.5], colors="lime", linewidths=.8)
        axes[r, 1].imshow(np.clip(rec[0].transpose(1, 2, 0), 0, 1))
        axes[r, 2].imshow(m[0], cmap="jet", vmin=0, vmax=1)
        for a in axes[r]:
            a.set_xticks([]); a.set_yticks([])
    for a, t in zip(axes[0], ["input (+GT)", "AE reconstruction", "1 - SSIM residual"]):
        a.set_title(t, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(R, "figures", "ae_reconstruction.png")); plt.close(fig)


if __name__ == "__main__":
    main()
