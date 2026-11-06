"""Figures + tables for the report/slides.

  python src/viz.py --tags ae patchcore_wrn50 patchcore_dinov2 --names AE PatchCore PatchCore-DINOv2
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from data import list_split, load_image, load_mask  # noqa: E402

R = os.path.join(os.path.dirname(__file__), "..", "results")
CATS = ["bottle", "metal_nut", "screw", "carpet", "leather"]
SIZE = 256


def load(tag, cat):
    d = os.path.join(R, tag, cat)
    return pd.read_csv(os.path.join(d, "scores.csv")), np.load(os.path.join(d, "maps.npz"))["test"].astype(np.float32), \
        pd.read_json(os.path.join(d, "metrics.json"), typ="series")


def summary_table(tags, names, out):
    rows = []
    for t, n in zip(tags, names):
        s = pd.read_csv(os.path.join(R, t, "summary.csv"))
        s["method"] = n
        rows.append(s)
    df = pd.concat(rows)
    keep = ["method", "category", "type", "image_auroc", "image_f1", "image_f1_max_oracle", "pixel_auroc", "au_pro",
            "seg_iou", "fpr_on_test_good", "recall_on_test_anom", "ms_per_img", "fit_time_s", "peak_gpu_MB"]
    df = df[[k for k in keep if k in df]]
    df.to_csv(os.path.join(out, "all_results.csv"), index=False)
    m = df.groupby("method", sort=False)[["image_auroc", "image_f1", "pixel_auroc", "au_pro", "ms_per_img", "fit_time_s"]].mean()
    m.to_csv(os.path.join(out, "mean_results.csv"))
    piv = df.pivot_table(index="category", columns="method", values=["image_auroc", "pixel_auroc", "au_pro", "image_f1"], sort=False)
    piv = piv.reindex(CATS)
    piv.round(3).to_csv(os.path.join(out, "pivot_results.csv"))
    print(piv.round(3).to_string()); print(m.round(3).to_string())

    # grouped bar chart per metric
    metrics_ = [("image_auroc", "Image AUROC"), ("pixel_auroc", "Pixel AUROC"), ("au_pro", "AU-PRO (FPR≤0.3)")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4), dpi=150)
    colors = ["#9aa5b1", "#2f6fdb", "#e07a1f", "#2a9d63", "#8e44ad"]
    w = 0.8 / len(names)
    for ax, (k, title) in zip(axes, metrics_):
        for j, n in enumerate(names):
            v = [df[(df.method == n) & (df.category == c)][k].values[0] for c in CATS]
            ax.bar(np.arange(len(CATS)) + (j - (len(names) - 1) / 2) * w, v, w, label=n, color=colors[j % 5])
        ax.set_xticks(range(len(CATS)), CATS, fontsize=8); ax.set_title(title, fontsize=10)
        ax.set_ylim(0.4 if k != "au_pro" else 0.0, 1.0); ax.grid(axis="y", alpha=.3)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=len(names), fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(os.path.join(out, "bar_metrics.png")); plt.close(fig)
    return df


def per_defect_table(tags, names, out):
    """Image AUROC of each defect type vs. the good test images + recall at the label-free threshold."""
    rows = []
    for t, n in zip(tags, names):
        for c in CATS:
            sc, _, met = load(t, c)
            good = sc[sc.label == 0].score.values
            for d in sorted(sc[sc.label == 1].defect.unique()):
                a = sc[sc.defect == d].score.values
                y = np.r_[np.zeros(len(good)), np.ones(len(a))]
                rows.append({"method": n, "category": c, "defect": d, "n": len(a),
                             "auroc_vs_good": roc_auc_score(y, np.r_[good, a]),
                             "recall_at_thr": float((a >= met["threshold"]).mean())})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "per_defect.csv"), index=False)
    piv = df.pivot_table(index=["category", "defect"], columns="method", values="auroc_vs_good", sort=False)
    piv.round(3).to_csv(os.path.join(out, "per_defect_pivot.csv"))
    print(piv.round(3).to_string())
    return df


def val_maps(tag, cat):
    """Validation-normal maps exactly as used for the threshold (fused tags are stored already smoothed)."""
    from run import smooth
    v = np.load(os.path.join(R, tag, cat, "maps.npz"))["val_raw"].astype(np.float32)
    met = json.load(open(os.path.join(R, tag, cat, "metrics.json")))
    return v if "fused_from" in met else smooth(v, 4.0)


def score_hists(tag, name, out):
    from run import image_scores
    fig, axes = plt.subplots(1, len(CATS), figsize=(3 * len(CATS), 2.6), dpi=150)
    for ax, c in zip(axes, CATS):
        sc, _, met = load(tag, c)
        vsc = image_scores(val_maps(tag, c))
        bins = np.linspace(min(sc.score.min(), vsc.min()), sc.score.max(), 30)
        ax.hist(vsc, bins, alpha=.6, label="val normal", color="#2a9d63", density=True)
        ax.hist(sc[sc.label == 0].score, bins, alpha=.6, label="test normal", color="#2f6fdb", density=True)
        ax.hist(sc[sc.label == 1].score, bins, alpha=.6, label="test defect", color="#d1495b", density=True)
        ax.axvline(met["threshold"], color="k", ls="--", lw=1)
        ax.set_title(c, fontsize=9); ax.set_yticks([])
    axes[0].legend(fontsize=6)
    fig.suptitle(f"{name}: image anomaly score distributions (dashed = threshold from val normals)", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(out, f"hist_{tag}.png")); plt.close(fig)


def _norm(m, lo, hi):
    return np.clip((m - lo) / max(hi - lo, 1e-8), 0, 1)


def comparison_grid(tags, names, cat, idxs, out, fname, title=""):
    """Rows = images; cols = input+GT, then heatmap overlay (+ predicted contour) per method."""
    _, _, test = list_split(cat)
    data = [load(t, cat) for t in tags]
    # display range per method, from held-out NORMAL validation maps only:
    # [90th pct of normal pixels, threshold + half the gap above it]
    lows = [float(np.percentile(val_maps(t, cat), 90)) for t in tags]
    fig, axes = plt.subplots(len(idxs), 1 + len(tags), figsize=(2.3 * (1 + len(tags)), 2.3 * len(idxs)), dpi=150,
                             squeeze=False)
    for r, i in enumerate(idxs):
        s = test[i]
        img = load_image(s.path, SIZE)
        gt = load_mask(s, SIZE)
        ax = axes[r, 0]
        ax.imshow(img)
        if gt.any():
            ax.contour(gt, levels=[.5], colors="lime", linewidths=1)
        ax.set_title(f"{s.defect}", fontsize=7); ax.axis("off")
        for j, (sc, maps, met) in enumerate(data):
            ax = axes[r, j + 1]
            thr = met["threshold"]
            m = maps[i]
            lo = lows[j]
            hm = _norm(m, lo, thr + 0.5 * (thr - lo))
            ax.imshow(img); ax.imshow(hm, cmap="jet", alpha=.45, vmin=0, vmax=1)
            if gt.any():
                ax.contour(gt, levels=[.5], colors="lime", linewidths=.8)
            if (m >= thr).any():
                ax.contour(m >= thr, levels=[.5], colors="white", linewidths=.8, linestyles="--")
            pred = "DEFECT" if sc.score[i] >= thr else "ok"
            ok = (sc.score[i] >= thr) == bool(s.label)
            ax.set_title(f"{names[j]}: {pred}", fontsize=7, color="green" if ok else "red"); ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(out, fname)); plt.close(fig)


def pick_cases(tag, cat, k=2):
    """Indices: confident TP, missed defects (FN), false alarms (FP) according to `tag`."""
    sc, _, met = load(tag, cat)
    thr = met["threshold"]
    an = sc[sc.label == 1].sort_values("score")
    gd = sc[sc.label == 0].sort_values("score", ascending=False)
    tp = list(an.index[-k:])
    fn = list(an[an.score < thr].index[:k])
    fp = list(gd[gd.score >= thr].index[:k])
    return tp, fn, fp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--names", nargs="+", required=True)
    args = ap.parse_args()
    out = os.path.join(R, "figures")
    os.makedirs(out, exist_ok=True)
    summary_table(args.tags, args.names, out)
    per_defect_table(args.tags, args.names, out)
    for t, n in zip(args.tags, args.names):
        score_hists(t, n, out)
    # qualitative: for every category, 1 easy + the failures of the strongest method (last tag)
    ref = args.tags[-1]
    for c in CATS:
        tp, fn, fp = pick_cases(ref, c, k=2)
        _, _, test = list_split(c)
        # one example per defect type (median-scored by the reference method)
        sc, _, _ = load(ref, c)
        per_type = [int(sc[sc.defect == d].score.sort_values().index[len(sc[sc.defect == d]) // 2])
                    for d in sorted(sc[sc.label == 1].defect.unique())]
        comparison_grid(args.tags, args.names, c, per_type, out, f"qual_{c}.png", f"{c}: one example per defect type")
        fails = fn + fp
        if fails:
            comparison_grid(args.tags, args.names, c, fails, out, f"fail_{c}.png",
                            f"{c}: missed defects / false alarms of {args.names[-1]}")
    print("figures -> results/figures")


if __name__ == "__main__":
    main()
