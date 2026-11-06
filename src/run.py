"""Train / evaluate one method configuration on a list of MVTec AD categories.

Decision rule (label-free): threshold = mean + 3*std of image scores of the
held-out NORMAL validation images (15 % of train/good). The same value is used
as pixel threshold for the binary localisation overlay, so "an image is
anomalous" <=> "at least one pixel of its map exceeds the threshold".

Example:
  python src/run.py --method patchcore --tag patchcore_wrn50 --cats bottle carpet
  python src/run.py --method ae --tag ae --cats bottle carpet
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(__file__))
from data import OBJECTS, list_split, load_mask, rel  # noqa: E402
import metrics as M  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
TEXTURES = {"carpet", "grid", "leather", "tile", "wood"}


def smooth(maps, sigma):
    if sigma <= 0:
        return maps
    return np.stack([gaussian_filter(m, sigma) for m in maps]).astype(np.float32)


def image_scores(maps, mode="max", topk_frac=0.01):
    flat = maps.reshape(len(maps), -1)
    if mode == "max":
        return flat.max(1)
    k = max(1, int(flat.shape[1] * topk_frac))
    return np.sort(flat, 1)[:, -k:].mean(1)


def build(args, cat):
    if args.method == "ae":
        from ae import AEDetector
        return AEDetector(size=args.size, epochs=args.epochs, z_dim=args.z_dim,
                          augment=(cat in TEXTURES) and not args.no_aug)
    if args.method == "rd":
        from rd import RDDetector
        return RDDetector(size=args.size, epochs=args.epochs)
    from patchcore import PatchCore
    return PatchCore(backbone=args.backbone, layers=tuple(args.layers), coreset=args.coreset,
                     out_size=args.size, dino_arch=args.dino_arch, dino_size=args.dino_size,
                     input_size=args.pc_input)


def evaluate(cat, test, val_maps, test_maps, sigma, score_mode, size):
    vm, tm = smooth(val_maps, sigma), smooth(test_maps, sigma)
    vs, ts = image_scores(vm, score_mode), image_scores(tm, score_mode)
    # Both rules use ONLY held-out normal validation images:
    #   primary  : mean + 3*std  (conservative, ~0.1 % false alarms under a Gaussian assumption)
    #   secondary: 95th percentile (an explicit 5 % false-alarm budget, common on a production line)
    thr = float(vs.mean() + 3 * vs.std())
    thr95 = float(np.percentile(vs, 95))
    labels = np.array([s.label for s in test])
    masks = np.stack([load_mask(s, size) for s in test])
    anom = labels == 1
    iou, dice = M.seg_iou_f1(masks[anom], tm[anom], thr)
    res = {
        "category": cat, "type": "object" if cat in OBJECTS else "texture",
        "n_test": int(len(test)), "n_anom": int(anom.sum()),
        "threshold": thr,
        "image_auroc": M.image_auroc(labels, ts),
        "image_f1": M.f1_at(labels, ts, thr),
        "image_f1_max_oracle": M.f1_max_oracle(labels, ts),
        "pixel_auroc": M.pixel_auroc(masks, tm),
        "au_pro": M.au_pro(masks, tm),
        "seg_iou": iou, "seg_dice": dice,
        "fpr_on_test_good": float((ts[~anom] >= thr).mean()),
        "recall_on_test_anom": float((ts[anom] >= thr).mean()),
        "threshold_p95": thr95,
        "image_f1_p95": M.f1_at(labels, ts, thr95),
        "fpr_on_test_good_p95": float((ts[~anom] >= thr95).mean()),
        "recall_on_test_anom_p95": float((ts[anom] >= thr95).mean()),
    }
    return res, ts, tm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["ae", "rd", "patchcore"], required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cats", nargs="+", default=["bottle", "metal_nut", "screw", "carpet", "leather"])
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--sigma", type=float, default=4.0)
    ap.add_argument("--score", choices=["max", "topk"], default="max")
    # AE
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--z_dim", type=int, default=256)
    ap.add_argument("--no_aug", action="store_true")
    # PatchCore
    ap.add_argument("--backbone", default="wrn50")
    ap.add_argument("--layers", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--coreset", type=float, default=0.1)
    ap.add_argument("--pc_input", type=int, default=256, help="WRN-50 input resolution (maps are evaluated at --size)")
    ap.add_argument("--dino_arch", default="vit_base_patch14_dinov2.lvd142m")
    ap.add_argument("--dino_size", type=int, default=448)
    args = ap.parse_args()

    out_dir = os.path.join(RESULTS, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for cat in args.cats:
        print(f"== {args.tag} / {cat}", flush=True)
        fit, val, test = list_split(cat)
        torch.cuda.reset_peak_memory_stats()
        det = build(args, cat).fit(fit)
        val_maps = det.predict(val)
        torch.cuda.synchronize()
        t0 = time.time()
        test_maps = det.predict(test)
        torch.cuda.synchronize()
        ms_per_img = (time.time() - t0) / len(test) * 1000
        res, ts, tm = evaluate(cat, test, val_maps, test_maps, args.sigma, args.score, args.size)
        res.update({"method": args.tag, "n_fit": len(fit), "n_val": len(val), "ms_per_img": ms_per_img,
                    "peak_gpu_MB": torch.cuda.max_memory_allocated() / 2 ** 20, **det.info})
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}), flush=True)
        rows.append(res)
        cdir = os.path.join(out_dir, cat)
        os.makedirs(cdir, exist_ok=True)
        np.savez_compressed(os.path.join(cdir, "maps.npz"), test=tm.astype(np.float16),
                            val_raw=val_maps.astype(np.float16), test_raw=test_maps.astype(np.float16))
        pd.DataFrame({"path": [rel(s.path) for s in test], "label": [s.label for s in test],
                      "defect": [s.defect for s in test], "score": ts,
                      "pred": (ts >= res["threshold"]).astype(int)}).to_csv(os.path.join(cdir, "scores.csv"), index=False)
        with open(os.path.join(cdir, "metrics.json"), "w") as f:
            json.dump({**res, "args": vars(args)}, f, indent=2)
        if args.method == "ae":
            torch.save(det.model.state_dict(), os.path.join(cdir, "ae.pt"))
        del det
        torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    cols = ["category", "image_auroc", "image_f1", "pixel_auroc", "au_pro", "ms_per_img"]
    print(df[cols].round(4).to_string(index=False))
    print("mean", df[cols[1:]].mean().round(4).to_dict())


if __name__ == "__main__":
    main()
