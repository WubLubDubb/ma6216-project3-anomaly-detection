"""Bonus ablation: every enhancement step vs. the PatchCore baseline (all variants reported).

The final variant is the one picked WITHOUT test labels on synthetic defects (synth_select.py);
the test columns are reported for every variant, including the ones with better test scores.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

R = os.path.join(os.path.dirname(__file__), "..", "results")
OUT = os.path.join(R, "figures")
CATS = ["bottle", "metal_nut", "screw", "carpet", "leather"]
VARIANTS = [
    ("patchcore_wrn50", "PatchCore WRN-50 @256 (baseline)"),
    ("patchcore_wrn50_fg", "WRN-50 @256 + fg mask"),
    ("patchcore_wrn50_320", "WRN-50 @320"),
    ("patchcore_wrn50_320_fg", "WRN-50 @320 + fg mask"),
    ("patchcore_dinov2", "DINOv2"),
    ("patchcore_dinov2_fg", "DINOv2 + fg mask (final: label-free pick)"),
    ("fusion_wrn50_dinov2", "Fusion WRN-50 + DINOv2"),
    ("fusion_wrn50_dinov2_fg", "Fusion WRN-50 + DINOv2 + fg mask"),
    ("fusion_wrn320_dinov2", "Fusion WRN-50@320 + DINOv2"),
    ("fusion_wrn320_dinov2_fg", "Fusion WRN-50@320 + DINOv2 + fg mask"),
    ("fusion_rd_wrn320_dinov2_fg", "Extra: + RD (3-model ensemble, not a candidate)"),
]
FINAL = "patchcore_dinov2_fg"
METRICS = ["image_auroc", "image_f1", "image_f1_p95", "pixel_auroc", "au_pro"]


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for tag, name in VARIANTS:
        p = os.path.join(R, tag, "summary.csv")
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p).set_index("category").reindex(CATS)
        row = {"variant": name, "tag": tag, **{f"mean_{m}": d[m].mean() for m in METRICS},
               **{f"{c}_image_auroc": d.loc[c, "image_auroc"] for c in CATS},
               "ms_per_img": d["ms_per_img"].mean()}
        rows.append(row)
    df = pd.DataFrame(rows)
    sel = pd.read_csv(os.path.join(R, "selection", "selection.csv")).set_index("test_tag")
    df["synthetic_val_auroc"] = df["tag"].map(sel["mean_syn_image_auroc"])   # selection criterion (no labels)
    base = df.iloc[0]
    for m in METRICS:
        df[f"delta_{m}"] = df[f"mean_{m}"] - base[f"mean_{m}"]
    df.to_csv(os.path.join(OUT, "ablation.csv"), index=False)
    show = df[["variant", "synthetic_val_auroc"] + [f"mean_{m}" for m in METRICS] + ["screw_image_auroc"]]
    print(show.round(4).to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), dpi=150, sharey=True)
    y = np.arange(len(df))[::-1]
    panels = [("synthetic_val_auroc", "Selection: synthetic-defect val AUROC (no labels)", 0.90),
              ("mean_image_auroc", "Test image AUROC (mean of 5)", 0.96),
              ("screw_image_auroc", "Test image AUROC on screw", 0.84)]
    colors = ["#9aa5b1" if i == 0 else "#e07a1f" if t == FINAL else "#2a9d63" if t.startswith("fusion_rd")
              else "#2f6fdb" for i, t in enumerate(df["tag"])]
    for ax, (col, title, lo) in zip(axes, panels):
        v = df[col]
        ax.barh(y, v.fillna(lo), color=colors)
        for yi, vi in zip(y, v):
            ax.text((lo if np.isnan(vi) else vi) + 0.001, yi, "n/a" if np.isnan(vi) else f"{vi:.3f}",
                    va="center", fontsize=7)
        ax.set_xlim(lo, 1.005); ax.set_title(title, fontsize=9); ax.grid(axis="x", alpha=.3)
    axes[0].set_yticks(y, df["variant"], fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "ablation.png")); plt.close(fig)


if __name__ == "__main__":
    main()
