"""Inference-speed benchmark on an otherwise idle GPU (model time only: images are
pre-loaded, so disk I/O and JPEG/PNG decoding are excluded).

  python src/benchmark.py --cat bottle
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from data import list_split  # noqa: E402

R = os.path.join(os.path.dirname(__file__), "..", "results")


def bench(fn, n_warm=3, n_rep=20):
    for _ in range(n_warm):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n_rep):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n_rep * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cat", default="bottle")
    ap.add_argument("--bs", type=int, default=1)
    args = ap.parse_args()
    fit, val, test = list_split(args.cat)
    out = {}

    # PatchCore variants: patch-feature extraction + nearest-neighbour search against the memory bank
    from patchcore import PatchCore, _aggregate
    from data import load_batch
    for name, kw in [("PatchCore WRN-50 @256", dict(backbone="wrn50")),
                     ("PatchCore WRN-50 @320", dict(backbone="wrn50", input_size=320)),
                     ("PatchCore DINOv2 @448", dict(backbone="dinov2", layers=(7, 11)))]:
        pc = PatchCore(**kw).fit(fit)
        x = torch.from_numpy(load_batch(test[:args.bs], pc.fx.input_size, True)).cuda()

        def f():
            with torch.no_grad():
                e = _aggregate(pc.fx(x))
                q = e.permute(0, 2, 3, 1).reshape(-1, e.shape[1])
                torch.cdist(q, pc.bank).min(1)
        out[name] = {"ms_per_batch": bench(f), "bank_MB": pc.info["bank_MB"]}
        print(name, out[name], flush=True)
        del pc
        torch.cuda.empty_cache()

    # AE and RD: forward pass only (weights do not matter for timing)
    from ae import ConvAE
    from rd import OCBE, Student, Teacher
    x = torch.from_numpy(load_batch(test[:args.bs], 256, False)).cuda()
    ae = ConvAE().cuda().eval()

    def ae_fwd():
        with torch.no_grad():
            ae(x)
    out["AE"] = {"ms_per_batch": bench(ae_fwd)}
    t, o, s = Teacher().cuda().eval(), OCBE().cuda().eval(), Student().cuda().eval()

    def rd():
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            s(o(t(x)))
    out["RD"] = {"ms_per_batch": bench(rd)}
    for k in ("AE", "RD"):
        print(k, out[k])
    a, b = out["PatchCore WRN-50 @320"]["ms_per_batch"], out["PatchCore DINOv2 @448"]["ms_per_batch"]
    out["Final (fusion WRN-50@320 + DINOv2)"] = {"ms_per_batch": a + b}
    for v in out.values():
        v["ms_per_image"] = v["ms_per_batch"] / args.bs
        v["images_per_s"] = 1000 / v["ms_per_image"]
    os.makedirs(os.path.join(R, "figures"), exist_ok=True)
    with open(os.path.join(R, "figures", f"benchmark_bs{args.bs}.json"), "w") as fh:
        json.dump({"gpu": torch.cuda.get_device_name(), "batch_size": args.bs, "category": args.cat, **out}, fh, indent=2)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
