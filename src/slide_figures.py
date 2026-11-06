"""Landscape, large-font figures for the presentation (results/figures/slides/).

Example images are chosen by a neutral rule (the median-scored sample of a defect type, or
the worst miss / worst false alarm of Ours), never hand-picked. Ours = PatchCore with a DINOv2
backbone + foreground mask, the variant picked without test labels (synth_select.py).
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from data import list_split, load_image, load_mask  # noqa: E402
from run import image_scores  # noqa: E402
from viz import load, val_maps  # noqa: E402

R = os.path.join(os.path.dirname(__file__), "..", "results")
OUT = os.path.join(R, "figures", "slides")
CATS = ["bottle", "metal_nut", "screw", "carpet", "leather"]
FINAL = "patchcore_dinov2_fg"
METHODS = [("ae", "AE", "#B8C0C8"), ("rd", "RD", "#3B78C4"),
           ("patchcore_wrn50", "PatchCore", "#6F7F8F"), (FINAL, "Ours", "#E07A1F")]
INK, MUTED = "#16212B", "#3E4A55"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": "#C9CFD4"})


def median_sample(tag, cat, defect):
    sc, _, _ = load(tag, cat)
    sub = sc[sc.defect == defect].sort_values("score")
    return int(sub.index[len(sub) // 2])


def overlay(ax, tag, cat, idx, test, show_pred=True):
    sc, maps, met = load(tag, cat)
    thr, m = met["threshold"], maps[idx]
    lo = float(np.percentile(val_maps(tag, cat), 90))
    hm = np.clip((m - lo) / max(thr + 0.5 * (thr - lo) - lo, 1e-8), 0, 1)
    ax.imshow(load_image(test[idx].path, 256)); ax.imshow(hm, cmap="jet", alpha=.45, vmin=0, vmax=1)
    gt = load_mask(test[idx], 256)
    if gt.any():
        ax.contour(gt, levels=[.5], colors="lime", linewidths=1.6)
    if show_pred and (m >= thr).any():
        ax.contour(m >= thr, levels=[.5], colors="white", linewidths=1.4, linestyles="--")
    return bool(sc.score[idx] >= thr)


def fig_dataset():
    picks = [("bottle", "broken_large"), ("metal_nut", "scratch"), ("screw", "thread_top"),
             ("carpet", "hole"), ("leather", "fold")]
    fig, axes = plt.subplots(2, 5, figsize=(11, 4.7), dpi=200)
    for j, (c, d) in enumerate(picks):
        _, _, test = list_split(c)
        g = next(s for s in test if s.defect == "good")
        a = test[median_sample(FINAL, c, d)]
        axes[0, j].imshow(load_image(g.path, 256)); axes[0, j].set_title(c, fontsize=14, fontweight="bold")
        axes[1, j].imshow(load_image(a.path, 256))
        axes[1, j].contour(load_mask(a, 256), levels=[.5], colors="red", linewidths=1.6)
        axes[1, j].set_xlabel(d, fontsize=13)
    axes[0, 0].set_ylabel("normal", fontsize=14); axes[1, 0].set_ylabel("defect (GT)", fontsize=14)
    for a in axes.ravel():
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "dataset_examples.png")); plt.close(fig)


def fig_bars():
    metrics = [("image_auroc", "Image AUROC", 0.5), ("pixel_auroc", "Pixel AUROC", 0.5), ("au_pro", "AU-PRO", 0.0)]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.9), dpi=200)
    w = 0.2
    for ax, (k, title, lo) in zip(axes, metrics):
        for j, (t, n, col) in enumerate(METHODS):
            d = pd.read_csv(os.path.join(R, t, "summary.csv")).set_index("category").reindex(CATS)
            ax.bar(np.arange(5) + (j - 1.5) * w, d[k].values, w, color=col, label=n)
        ax.set_xticks(range(5), ["bottle", "metal\nnut", "screw", "carpet", "leather"], fontsize=11)
        ax.set_title(title, fontsize=14); ax.set_ylim(lo, 1.0); ax.tick_params(axis="y", labelsize=11)
        ax.grid(axis="y", alpha=.3); ax.spines[["top", "right"]].set_visible(False)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=4, fontsize=13, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(os.path.join(OUT, "bars.png")); plt.close(fig)


def fig_localisation():
    picks = [("bottle", "broken_large"), ("metal_nut", "scratch"), ("screw", "scratch_neck"),
             ("carpet", "thread"), ("leather", "glue")]
    rows = [("input + GT", None)] + [(n, t) for t, n, _ in METHODS if n in ("AE", "PatchCore", "Ours")]
    fig, axes = plt.subplots(len(rows), 5, figsize=(10, 8.2), dpi=200)
    for j, (c, d) in enumerate(picks):
        _, _, test = list_split(c)
        i = median_sample(FINAL, c, d)
        axes[0, j].imshow(load_image(test[i].path, 256))
        axes[0, j].contour(load_mask(test[i], 256), levels=[.5], colors="lime", linewidths=1.6)
        axes[0, j].set_title(f"{c}\n{d}", fontsize=12)
        for r, (n, t) in enumerate(rows[1:], start=1):
            overlay(axes[r, j], t, c, i, test)
    for r, (n, _) in enumerate(rows):
        axes[r, 0].set_ylabel(n, fontsize=14, fontweight="bold" if n == "Ours" else "normal")
    for a in axes.ravel():
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout(h_pad=0.4, w_pad=0.3); fig.savefig(os.path.join(OUT, "localisation.png")); plt.close(fig)


def fig_failures():
    tag = FINAL
    cases = []
    sc, _, met = load(tag, "screw")
    cases.append(("screw", int(sc[sc.label == 1].score.idxmin()), "missed: screw"))
    # the false alarm that exceeds its category's threshold by the largest factor
    fa = []
    for c in CATS:
        sc, _, met = load(tag, c)
        g = sc[(sc.label == 0) & (sc.score >= met["threshold"])]
        fa += [(r.score / met["threshold"], c, int(i)) for i, r in g.iterrows()]
    _, c_fa, i_fa = max(fa)
    cases.append((c_fa, i_fa, f"false alarm: normal {c_fa}"))
    # bent: the sample whose GT lies most outside the (label-free) foreground mask
    from foreground import foreground_mask
    _, _, t_mn = list_split("metal_nut")
    bent = [i for i, s in enumerate(t_mn) if s.defect == "bent"]
    outside = [1 - (load_mask(t_mn[i], 256).astype(bool) & foreground_mask(t_mn[i].path)).sum()
               / load_mask(t_mn[i], 256).sum() for i in bent]
    cases.append(("metal_nut", bent[int(np.argmax(outside))], "metal_nut bent: GT off the part"))
    fig, axes = plt.subplots(2, 3, figsize=(8.4, 5.9), dpi=200)
    for j, (c, i, title) in enumerate(cases):
        _, _, test = list_split(c)
        if title == "missed: screw":
            title = f"missed: screw {test[i].defect}"
        axes[0, j].imshow(load_image(test[i].path, 256))
        gt = load_mask(test[i], 256)
        if gt.any():
            axes[0, j].contour(gt, levels=[.5], colors="lime", linewidths=1.6)
        axes[0, j].set_title(title, fontsize=12)
        overlay(axes[1, j], tag, c, i, test)
    axes[0, 0].set_ylabel("input + GT", fontsize=13); axes[1, 0].set_ylabel("Ours", fontsize=13, fontweight="bold")
    for a in axes.ravel():
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "failures.png")); plt.close(fig)


def fig_shift():
    tag = FINAL
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), dpi=200)
    for ax, c in zip(axes, ["screw", "carpet", "leather"]):
        sc, _, met = load(tag, c)
        v = image_scores(val_maps(tag, c))
        hi = 1.5 * met["threshold"]
        bins = np.linspace(min(v.min(), sc.score.min()), hi, 32)
        ax.hist(v, bins, alpha=.75, color="#2A9D63", density=True, label="normal (train folder, held out)")
        ax.hist(sc[sc.label == 0].score, bins, alpha=.7, color="#3B78C4", density=True, label="normal (test folder)")
        ax.hist(np.clip(sc[sc.label == 1].score, None, hi), bins, alpha=.55, color="#D1495B", density=True,
                label="defect (test)")
        ax.axvline(met["threshold"], color=INK, ls="--", lw=1.6)
        ax.set_title(c, fontsize=14); ax.set_yticks([]); ax.tick_params(labelsize=11)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.set_xlabel("anomaly score (nearest-normal distance)", fontsize=11)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=3, fontsize=12, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.86)); fig.savefig(os.path.join(OUT, "score_shift.png")); plt.close(fig)


def fig_ae():
    from ae import AEDetector
    cases = [("bottle", "broken_large"), ("metal_nut", "flip"), ("carpet", "good"), ("carpet", "color")]
    fig, axes = plt.subplots(3, 4, figsize=(8.6, 6.6), dpi=200)
    dets = {}
    for j, (c, d) in enumerate(cases):
        if c not in dets:
            dets[c] = AEDetector()
            dets[c].model.load_state_dict(torch.load(os.path.join(R, "ae", c, "ae.pt"), map_location="cuda"))
        _, _, test = list_split(c)
        s = next(x for x in test if x.defect == d)
        m, rec = dets[c].predict([s], return_recon=True)
        axes[0, j].imshow(load_image(s.path, 256)); axes[0, j].set_title(f"{c} · {d}", fontsize=12)
        gt = load_mask(s, 256)
        if gt.any():
            axes[0, j].contour(gt, levels=[.5], colors="lime", linewidths=1.4)
        axes[1, j].imshow(np.clip(rec[0].transpose(1, 2, 0), 0, 1))
        axes[2, j].imshow(m[0], cmap="jet", vmin=0, vmax=1)
    for r, n in enumerate(["input", "AE output", "1 − SSIM"]):
        axes[r, 0].set_ylabel(n, fontsize=13)
    for a in axes.ravel():
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "ae_recon.png")); plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    for f in (fig_dataset, fig_bars, fig_localisation, fig_failures, fig_shift, fig_ae):
        f(); print("done", f.__name__, flush=True)


if __name__ == "__main__":
    main()
