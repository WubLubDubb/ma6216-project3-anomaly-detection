"""Failure analysis support:
1. Do normal TEST images score like the held-out normal TRAIN images (validation)?
   Two-sided Mann-Whitney U test per method and category.
2. Where do false alarms come from? Fraction of normal test images whose map maximum
   lies within 8 px of the image border (border artefacts), and how many false alarms do.

  python src/score_shift_test.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

sys.path.insert(0, os.path.dirname(__file__))
from run import image_scores  # noqa: E402
from viz import val_maps  # noqa: E402

R = os.path.join(os.path.dirname(__file__), "..", "results")
CATS = ["bottle", "metal_nut", "screw", "carpet", "leather"]
TAGS = {"ae": "AE", "rd": "RD", "patchcore_wrn50": "PatchCore", "patchcore_dinov2_fg": "Ours",
        "fusion_wrn320_dinov2_fg": "Fusion WRN-50@320 + DINOv2 + fg"}


def main():
    rows = []
    for tag, name in TAGS.items():
        for c in CATS:
            v = image_scores(val_maps(tag, c))
            s = pd.read_csv(os.path.join(R, tag, c, "scores.csv"))
            thr = json.load(open(os.path.join(R, tag, c, "metrics.json")))["threshold"]
            maps = np.load(os.path.join(R, tag, c, "maps.npz"))["test"].astype(np.float32)
            good = np.where(s.label.values == 0)[0]
            g = s.score.values[good]
            border = []
            for i in good:
                y, x = np.unravel_index(np.argmax(maps[i]), maps[i].shape)
                border.append(min(y, x, maps.shape[1] - 1 - y, maps.shape[2] - 1 - x) < 8)
            border = np.array(border)
            fa = g >= thr
            rows.append({"method": name, "category": c,
                         "val_normal_median": float(np.median(v)), "test_normal_median": float(np.median(g)),
                         "mannwhitney_p": float(mannwhitneyu(v, g, alternative="two-sided").pvalue),
                         "false_alarms": int(fa.sum()), "n_test_normal": int(len(g)),
                         "max_at_border_frac": float(border.mean()), "false_alarms_at_border": int((fa & border).sum())})
    df = pd.DataFrame(rows)
    os.makedirs(os.path.join(R, "figures"), exist_ok=True)
    df.to_csv(os.path.join(R, "figures", "score_shift_test.csv"), index=False)
    pd.set_option("display.width", 200)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
