"""Dataset analysis: split sizes, defect types, defect size statistics, example grid."""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.ndimage import label  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from data import OBJECTS, list_split  # noqa: E402

CATS = ["bottle", "metal_nut", "screw", "carpet", "leather"]
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "dataset")


def main():
    os.makedirs(OUT, exist_ok=True)
    rows, defect_rows = [], []
    for c in CATS:
        fit, val, test = list_split(c)
        w, h = Image.open(test[0].path).size
        rows.append({"category": c, "type": "object" if c in OBJECTS else "texture", "resolution": f"{w}x{h}",
                     "train_fit": len(fit), "train_val": len(val),
                     "test_good": sum(s.label == 0 for s in test), "test_defect": sum(s.label == 1 for s in test),
                     "defect_types": len({s.defect for s in test if s.label})})
        for s in test:
            if s.label == 0:
                continue
            m = np.asarray(Image.open(s.mask_path).convert("L")) > 127
            _, n = label(m, np.ones((3, 3), int))
            defect_rows.append({"category": c, "defect": s.defect, "area_pct": 100 * m.mean(), "n_regions": n})
    df = pd.DataFrame(rows)
    dd = pd.DataFrame(defect_rows)
    df.to_csv(os.path.join(OUT, "splits.csv"), index=False)
    per_type = dd.groupby(["category", "defect"]).agg(n=("area_pct", "size"), area_pct_median=("area_pct", "median"),
                                                      area_pct_min=("area_pct", "min"), area_pct_max=("area_pct", "max")).round(3)
    per_type.to_csv(os.path.join(OUT, "defect_types.csv"))
    per_cat = dd.groupby("category").area_pct.describe(percentiles=[.1, .5, .9]).round(3)
    per_cat.to_csv(os.path.join(OUT, "defect_area_by_category.csv"))
    print(df.to_string(index=False)); print(per_type.to_string()); print(per_cat.to_string())

    # defect area distribution (log scale)
    fig, ax = plt.subplots(figsize=(7, 3.2), dpi=150)
    data = [dd[dd.category == c].area_pct.values for c in CATS]
    ax.boxplot(data, tick_labels=CATS, showfliers=True)
    ax.set_yscale("log"); ax.set_ylabel("defect area (% of image)")
    ax.set_title("Defect size per category (test set GT masks)")
    ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "defect_area_boxplot.png")); plt.close(fig)

    # example grid: one normal + one example per defect type, with GT contour
    for c in CATS:
        _, _, test = list_split(c)
        types = ["good"] + sorted({s.defect for s in test if s.label})
        fig, axes = plt.subplots(1, len(types), figsize=(2.2 * len(types), 2.5), dpi=150)
        for ax, t in zip(axes, types):
            s = next(x for x in test if x.defect == t)
            ax.imshow(Image.open(s.path).convert("RGB"))
            if s.mask_path:
                ax.contour(np.asarray(Image.open(s.mask_path).convert("L")) > 127, levels=[0.5], colors="red", linewidths=1)
            ax.set_title(t, fontsize=8); ax.axis("off")
        fig.suptitle(c, fontsize=10); fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"examples_{c}.png")); plt.close(fig)


if __name__ == "__main__":
    main()
