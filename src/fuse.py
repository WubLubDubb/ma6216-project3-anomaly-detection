"""Bonus — label-free score-level fusion of several methods' anomaly maps.

Each method's (smoothed) anomaly map is standardised with statistics of its own
image scores on the held-out NORMAL validation images:
    z_k(x) = (map_k(x) - mean(val scores_k)) / std(val scores_k)
i.e. "how many normal-standard-deviations above normal". The fused map is the
mean of the z-maps. No test label or mask is used anywhere.

  python src/fuse.py --tags patchcore_wrn50 patchcore_dinov2 --out fusion_wrn50_dinov2
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from data import list_split, rel  # noqa: E402
from run import RESULTS, evaluate, image_scores, smooth  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sigma", type=float, default=4.0)
    ap.add_argument("--size", type=int, default=256)
    args = ap.parse_args()
    cats = sorted(set.intersection(*[set(d for d in os.listdir(os.path.join(RESULTS, t))
                                         if os.path.isdir(os.path.join(RESULTS, t, d))) for t in args.tags]))
    rows = []
    for cat in cats:
        _, _, test = list_split(cat)
        zv, zt, ms = 0, 0, 0.0
        for t in args.tags:
            z = np.load(os.path.join(RESULTS, t, cat, "maps.npz"))
            v = smooth(z["val_raw"].astype(np.float32), args.sigma)
            te = smooth(z["test_raw"].astype(np.float32), args.sigma)
            vs = image_scores(v)
            mu, sd = vs.mean(), vs.std() + 1e-8
            zv = zv + (v - mu) / sd
            zt = zt + (te - mu) / sd
            ms += json.load(open(os.path.join(RESULTS, t, cat, "metrics.json")))["ms_per_img"]
        zv, zt = zv / len(args.tags), zt / len(args.tags)
        res, ts, tm = evaluate(cat, test, zv, zt, 0.0, "max", args.size)  # already smoothed
        res.update({"method": args.out, "fused_from": args.tags, "ms_per_img": ms})
        d = os.path.join(RESULTS, args.out, cat)
        os.makedirs(d, exist_ok=True)
        np.savez_compressed(os.path.join(d, "maps.npz"), test=tm.astype(np.float16),
                            val_raw=zv.astype(np.float16), test_raw=zt.astype(np.float16))
        pd.DataFrame({"path": [rel(s.path) for s in test], "label": [s.label for s in test],
                      "defect": [s.defect for s in test], "score": ts,
                      "pred": (ts >= res["threshold"]).astype(int)}).to_csv(os.path.join(d, "scores.csv"), index=False)
        json.dump({**res, "args": {**vars(args), "sigma_applied_before_fusion": args.sigma}},
                  open(os.path.join(d, "metrics.json"), "w"), indent=2)
        rows.append(res)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, args.out, "summary.csv"), index=False)
    print(df[["category", "image_auroc", "image_f1", "image_f1_p95", "pixel_auroc", "au_pro"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
