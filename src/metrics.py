"""Evaluation metrics: image AUROC, F1, pixel AUROC, AU-PRO (FPR <= 0.3).

AU-PRO follows the reference implementation released with MVTec AD
(Bergmann et al., IJCV 2021): per-connected-component overlap averaged over
all GT regions, plotted against the pixel FPR on normal pixels, integrated up
to FPR = 0.3 and normalised by 0.3.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import label
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score


def image_auroc(labels, scores):
    return float(roc_auc_score(labels, scores))


def f1_at(labels, scores, thr):
    pred = (np.asarray(scores) >= thr).astype(int)
    return float(f1_score(labels, pred, zero_division=0))


def f1_max_oracle(labels, scores):
    """Best F1 over all thresholds. USES TEST LABELS -> reference only, never the decision rule."""
    p, r, _ = precision_recall_curve(labels, scores)
    f = 2 * p * r / np.clip(p + r, 1e-12, None)
    return float(np.nanmax(f))


def pixel_auroc(masks, maps, max_pixels: int = 4_000_000, seed: int = 0):
    y = masks.ravel().astype(np.uint8)
    s = maps.ravel().astype(np.float32)
    if y.size > max_pixels:  # uniform subsample keeps it fast; AUROC is unbiased under it
        idx = np.random.default_rng(seed).choice(y.size, max_pixels, replace=False)
        y, s = y[idx], s[idx]
    return float(roc_auc_score(y, s))


def _pro_curve(maps, masks):
    structure = np.ones((3, 3), dtype=int)
    n_ok, n_regions = 0, 0
    fp_changes = np.zeros(maps.shape, dtype=np.uint32)
    pro_changes = np.zeros(maps.shape, dtype=np.float64)
    for i, gt in enumerate(masks):
        lab, n = label(gt, structure)
        n_regions += n
        ok = lab == 0
        n_ok += ok.sum()
        fp_changes[i, ok] = 1
        for k in range(n):
            reg = lab == (k + 1)
            pro_changes[i, reg] = 1.0 / reg.sum()
    s = maps.ravel().astype(np.float64)
    order = np.argsort(s, kind="stable")[::-1]
    s = s[order]
    fpr = np.cumsum(fp_changes.ravel()[order]) / n_ok
    pro = np.cumsum(pro_changes.ravel()[order]) / n_regions
    keep = np.append(np.diff(s) != 0, True)
    fpr, pro = np.clip(fpr[keep], 0, 1), np.clip(pro[keep], 0, 1)
    return np.concatenate([[0.0], fpr]), np.concatenate([[0.0], pro])


def au_pro(masks, maps, fpr_limit: float = 0.3):
    fpr, pro = _pro_curve(maps, masks)
    # interpolate the curve exactly at the integration limit
    idx = np.searchsorted(fpr, fpr_limit, side="right")
    x = np.concatenate([fpr[:idx], [fpr_limit]])
    y = np.concatenate([pro[:idx], [np.interp(fpr_limit, fpr, pro)]])
    return float(np.trapezoid(y, x) / fpr_limit)


def seg_iou_f1(masks, maps, thr):
    """Pixel IoU / Dice of the binarised map at the (label-free) threshold, anomalous images only."""
    pred = maps >= thr
    gt = masks.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    dice = 2 * inter / max(pred.sum() + gt.sum(), 1)
    return float(inter / max(union, 1)), float(dice)
