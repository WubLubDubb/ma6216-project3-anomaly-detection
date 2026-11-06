"""Recompute all metrics from saved raw anomaly maps (no model re-run), e.g. after
adding a metric or changing post-processing.

  python src/reeval.py --tags ae patchcore_wrn50 patchcore_dinov2
  python src/reeval.py --tags patchcore_wrn50 --sigma 2 --score topk --out_suffix _s2_topk
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
from run import RESULTS, evaluate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--sigma", type=float, default=4.0)
    ap.add_argument("--score", choices=["max", "topk"], default="max")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--out_suffix", default="", help="'' = overwrite metrics in place; otherwise write a new tag")
    ap.add_argument("--fg", action="store_true", help="suppress background on object categories (foreground.py)")
    args = ap.parse_args()
    fg_cache = {}
    for tag in args.tags:
        src = os.path.join(RESULTS, tag)
        dst = src + args.out_suffix
        rows = []
        for cat in sorted(d for d in os.listdir(src) if os.path.isdir(os.path.join(src, d))):
            old = json.load(open(os.path.join(src, cat, "metrics.json")))
            z = np.load(os.path.join(src, cat, "maps.npz"))
            _, val, test = list_split(cat)
            vr, tr = z["val_raw"].astype(np.float32), z["test_raw"].astype(np.float32)
            if args.fg:
                # mask the raw maps; the usual Gaussian smoothing is applied afterwards in evaluate()
                import foreground as FG
                if cat not in fg_cache:
                    fg_cache[cat] = (FG.masks_for(val, cat, args.size), FG.masks_for(test, cat, args.size))
                vr = FG.apply(vr, fg_cache[cat][0])
                tr = FG.apply(tr, fg_cache[cat][1])
            res, ts, tm = evaluate(cat, test, vr, tr, args.sigma, args.score, args.size)
            keep = {k: old[k] for k in old if k not in res and k != "args"}
            res = {**keep, **res, "method": os.path.basename(dst),
                   "postproc": {"sigma": args.sigma, "score": args.score, "fg_mask": args.fg}}
            os.makedirs(os.path.join(dst, cat), exist_ok=True)
            json.dump({**res, "args": old.get("args")}, open(os.path.join(dst, cat, "metrics.json"), "w"), indent=2)
            pd.DataFrame({"path": [rel(s.path) for s in test], "label": [s.label for s in test],
                          "defect": [s.defect for s in test], "score": ts,
                          "pred": (ts >= res["threshold"]).astype(int)}).to_csv(os.path.join(dst, cat, "scores.csv"), index=False)
            if args.out_suffix:
                # store the (masked, unsmoothed) maps as "raw" so that viz/fuse treat this tag like any other
                np.savez_compressed(os.path.join(dst, cat, "maps.npz"), test=tm.astype(np.float16),
                                    val_raw=vr.astype(np.float16), test_raw=tr.astype(np.float16))
            rows.append(res)
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(dst, "summary.csv"), index=False)
        cols = ["category", "image_auroc", "image_f1", "image_f1_p95", "pixel_auroc", "au_pro"]
        print(f"== {os.path.basename(dst)}"); print(df[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
