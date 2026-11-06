"""Label-free model selection, step 2: rank the PatchCore variants on synthetic defects.

Selection rule (fixed before the run): the variant with the highest image AUROC
on "held-out normal validation images vs. the same images with synthetic
defects" (synthetic.py), averaged over the 5 categories; ties broken by the
synthetic AU-PRO. No test image, label or mask is read to make the choice.
Test results are loaded afterwards only to report how well the label-free
ranking agrees with the test ranking.

Everything after the memory bank is exactly the pipeline of run.py / reeval.py /
fuse.py: raw map -> optional foreground mask -> Gaussian sigma 4 -> (fusion:
z-score each map with its own normal-validation image scores, then average)
-> image score = max of the map.

  python src/synth_select.py predict --tag patchcore_wrn50        # one per backbone, GPU
  python src/synth_select.py predict --tag patchcore_wrn50_320
  python src/synth_select.py predict --tag patchcore_dinov2
  python src/synth_select.py score                                # CPU, writes results/selection/
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from data import list_split, load_mask  # noqa: E402
import foreground as FG  # noqa: E402
import metrics as M  # noqa: E402
from run import RESULTS, image_scores, smooth  # noqa: E402
from synthetic import CATS, TYPES, list_synthetic  # noqa: E402

OUT = os.path.join(RESULTS, "selection")
BASES = {"wrn50": "patchcore_wrn50", "wrn320": "patchcore_wrn50_320", "dino": "patchcore_dinov2"}
# (name, components, foreground mask, tag of the same variant on the test set)
CANDIDATES = [
    ("PatchCore WRN-50 @256 (baseline)", ["wrn50"], False, "patchcore_wrn50"),
    ("WRN-50 @256 + fg mask", ["wrn50"], True, "patchcore_wrn50_fg"),
    ("WRN-50 @320", ["wrn320"], False, "patchcore_wrn50_320"),
    ("WRN-50 @320 + fg mask", ["wrn320"], True, "patchcore_wrn50_320_fg"),
    ("DINOv2", ["dino"], False, "patchcore_dinov2"),
    ("DINOv2 + fg mask", ["dino"], True, "patchcore_dinov2_fg"),
    ("Fusion WRN-50 + DINOv2", ["wrn50", "dino"], False, "fusion_wrn50_dinov2"),
    ("Fusion WRN-50 + DINOv2 + fg mask", ["wrn50", "dino"], True, "fusion_wrn50_dinov2_fg"),
    ("Fusion WRN-50@320 + DINOv2", ["wrn320", "dino"], False, "fusion_wrn320_dinov2"),
    ("Fusion WRN-50@320 + DINOv2 + fg mask", ["wrn320", "dino"], True, "fusion_wrn320_dinov2_fg"),
]
PREVIOUS_FINAL = "fusion_wrn320_dinov2_fg"   # the final before this check, chosen on test results
SIGMA, N_BOOT = 4.0, 1000


# ------------------------------------------------------------------ step A: maps (GPU)
def predict(tag, cats):
    import torch
    from patchcore import PatchCore
    for cat in cats:
        a = json.load(open(os.path.join(RESULTS, tag, cat, "metrics.json")))["args"]
        fit, val, _ = list_split(cat)
        syn, _ = list_synthetic(cat)
        pc = PatchCore(backbone=a["backbone"], layers=tuple(a["layers"]), coreset=a["coreset"], out_size=a["size"],
                       dino_arch=a["dino_arch"], dino_size=a["dino_size"], input_size=a.get("pc_input", 256))
        bank = os.path.join(RESULTS, tag, cat, "bank.pt")
        if os.path.exists(bank):
            pc.bank = torch.load(bank).to(pc.device)
        else:   # same code, same seed as the original run; memory bank kept for re-use
            pc.fit(fit)
            torch.save(pc.bank.cpu(), bank)
        v, s = pc.predict(val), pc.predict(syn)
        # does the re-fitted model reproduce the validation maps of the reported run?
        old = np.load(os.path.join(RESULTS, tag, cat, "maps.npz"))["val_raw"].astype(np.float32)
        vs_new, vs_old = image_scores(smooth(v, SIGMA)), image_scores(smooth(old, SIGMA))
        check = {"max_abs_diff_rel": float(np.abs(v - old).max() / np.abs(old).max()),
                 "val_score_corr": float(np.corrcoef(vs_new, vs_old)[0, 1]),
                 "val_score_max_rel_diff": float(np.max(np.abs(vs_new - vs_old) / vs_old))}
        d = os.path.join(OUT, tag, cat)
        os.makedirs(d, exist_ok=True)
        np.savez_compressed(os.path.join(d, "maps.npz"), val_raw=v.astype(np.float16), syn_raw=s.astype(np.float16))
        json.dump(check, open(os.path.join(d, "check.json"), "w"), indent=2)
        print(tag, cat, len(val), "val +", len(syn), "synthetic;", check, flush=True)
        del pc
        torch.cuda.empty_cache()


# ------------------------------------------------------------------ step B: scores (CPU)
def _maps(cat, val, syn):
    """Smoothed val / synthetic maps of each base, with and without the foreground mask."""
    fg_v, fg_s = FG.masks_for(val, cat), FG.masks_for(syn, cat)
    out = {}
    for b, tag in BASES.items():
        z = np.load(os.path.join(OUT, tag, cat, "maps.npz"))
        v, s = z["val_raw"].astype(np.float32), z["syn_raw"].astype(np.float32)
        out[b, False] = smooth(v, SIGMA), smooth(s, SIGMA)
        out[b, True] = smooth(FG.apply(v, fg_v), SIGMA), smooth(FG.apply(s, fg_s), SIGMA)
    return out


def _variant(maps, comps, fg):
    zv, zs = 0, 0
    for b in comps:
        v, s = maps[b, fg]
        vs = image_scores(v)
        mu, sd = vs.mean(), vs.std() + 1e-8
        zv, zs = zv + (v - mu) / sd, zs + (s - mu) / sd
    return zv / len(comps), zs / len(comps)


def score():
    rows, boot_scores = [], {}
    rng = np.random.default_rng(0)
    boots = {}
    for cat in CATS:
        _, val, _ = list_split(cat)
        syn, idx = list_synthetic(cat)
        maps = _maps(cat, val, syn)
        masks = np.concatenate([np.zeros((len(val), 256, 256), np.uint8), np.stack([load_mask(s, 256) for s in syn])])
        boots[cat] = rng.integers(0, len(val), (N_BOOT, len(val)))
        for name, comps, fg, test_tag in CANDIDATES:
            mv, ms = _variant(maps, comps, fg)
            sv, ss = image_scores(mv), image_scores(ms)
            y = np.r_[np.zeros(len(sv)), np.ones(len(ss))]
            sc = np.r_[sv, ss]
            r = {"variant": name, "test_tag": test_tag, "category": cat, "n_val": len(sv), "n_syn": len(ss),
                 "syn_image_auroc": roc_auc_score(y, sc)}
            for t in TYPES:
                k = idx["type"].values == t
                r[f"syn_auroc_{t}"] = roc_auc_score(np.r_[np.zeros(len(sv)), np.ones(k.sum())], np.r_[sv, ss[k]])
            allm = np.concatenate([mv, ms])
            r["syn_pixel_auroc"] = M.pixel_auroc(masks, allm)
            r["syn_au_pro"] = M.au_pro(masks, allm)
            rows.append(r)
            # paired bootstrap over validation images (an image brings its normal and its synthetic versions)
            vi = idx["val_index"].values
            bs = []
            for b in range(N_BOOT):
                cnt = np.bincount(boots[cat][b], minlength=len(sv))
                bs.append(roc_auc_score(y, sc, sample_weight=np.r_[cnt, cnt[vi]]))
            boot_scores[name, cat] = np.array(bs)
            print(f"{cat:10s} {name:40s} syn AUROC {r['syn_image_auroc']:.4f}  AU-PRO {r['syn_au_pro']:.4f}", flush=True)
    per = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    per.to_csv(os.path.join(OUT, "selection_per_category.csv"), index=False)

    names = [c[0] for c in CANDIDATES]
    g = per.groupby("variant", sort=False)
    df = g[["syn_image_auroc", "syn_auroc_cutpaste", "syn_auroc_scar", "syn_auroc_draem",
            "syn_pixel_auroc", "syn_au_pro"]].mean().reindex(names).add_prefix("mean_")
    df["test_tag"] = [c[3] for c in CANDIDATES]
    for c in CATS:
        df[f"{c}_syn_auroc"] = per[per.category == c].set_index("variant").reindex(names)["syn_image_auroc"].values
    boot_mean = np.stack([np.mean([boot_scores[n, c] for c in CATS], 0) for n in names])  # V x B
    df["boot_win_frac"] = np.bincount(boot_mean.argmax(0), minlength=len(names)) / N_BOOT
    df["boot_ci_low"], df["boot_ci_high"] = np.percentile(boot_mean, 2.5, 1), np.percentile(boot_mean, 97.5, 1)
    order = df.sort_values(["mean_syn_image_auroc", "mean_syn_au_pro"], ascending=False)
    pick = order.index[0]

    # only now: how do the label-free choices compare with the test results?
    for tag_col, m in [("test_image_auroc", "image_auroc"), ("test_au_pro", "au_pro")]:
        df[tag_col] = [pd.read_csv(os.path.join(RESULTS, t, "summary.csv"))[m].mean() for t in df.test_tag]
    df["test_screw_image_auroc"] = [pd.read_csv(os.path.join(RESULTS, t, "summary.csv")).set_index("category")
                                    .loc["screw", "image_auroc"] for t in df.test_tag]
    df["rank_syn"] = df["mean_syn_image_auroc"].rank(ascending=False, method="min").astype(int)
    df["rank_test"] = df["test_image_auroc"].rank(ascending=False, method="min").astype(int)
    df.index.name = "variant"
    df.to_csv(os.path.join(OUT, "selection.csv"))
    fin = df.index[df.test_tag == PREVIOUS_FINAL][0]
    i_pick, i_fin = names.index(pick), names.index(fin)
    diff = boot_mean[i_pick] - boot_mean[i_fin]
    rho = spearmanr(df.mean_syn_image_auroc, df.test_image_auroc)
    summary = {"rule": "argmax mean synthetic-validation image AUROC over 5 categories (tie-break: synthetic AU-PRO)",
               "label_free_pick": pick, "pick_test_tag": df.loc[pick, "test_tag"],
               "previous_final_chosen_on_test": fin, "same_as_previous_final": bool(pick == fin),
               "pick_boot_win_frac": float(df.loc[pick, "boot_win_frac"]),
               "pick_minus_previous_final_syn_auroc_ci95": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
               "spearman_syn_vs_test_image_auroc": {"rho": float(rho.statistic), "p": float(rho.pvalue)},
               "n_bootstrap": N_BOOT}
    json.dump(summary, open(os.path.join(OUT, "selection.json"), "w"), indent=2)
    pd.set_option("display.width", 250)
    cols = ["mean_syn_image_auroc", "mean_syn_auroc_cutpaste", "mean_syn_auroc_scar", "mean_syn_auroc_draem",
            "mean_syn_au_pro", "boot_win_frac", "rank_syn", "test_image_auroc", "rank_test"]
    print(df[cols].round(4).to_string())
    print(json.dumps(summary, indent=2))
    figures(df, pick, fin)


def figures(df, pick, fin):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=150)
    for n, r in df.iterrows():
        c = "#e07a1f" if n == pick else "#3b78c4" if n == fin else "#8a96a3"
        ax.scatter(r.mean_syn_image_auroc, r.test_image_auroc, s=60 if n in (pick, fin) else 30, color=c, zorder=3)
        ax.annotate(n, (r.mean_syn_image_auroc, r.test_image_auroc), fontsize=6.5, xytext=(4, 3),
                    textcoords="offset points", color=c)
    ax.set_xlabel("Synthetic validation image AUROC (no test labels)")
    ax.set_ylabel("Test image AUROC")
    ax.grid(alpha=.3)
    ax.set_title("Label-free selection vs. test (mean of 5 categories)", fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.join(RESULTS, "figures"), exist_ok=True)
    fig.savefig(os.path.join(RESULTS, "figures", "selection_synthetic.png"))
    plt.close(fig)


def examples():
    """One synthetic defect of each kind per category (figure for the report)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    fig, ax = plt.subplots(len(TYPES), len(CATS), figsize=(2.4 * len(CATS), 2.5 * len(TYPES)), dpi=150)
    for j, c in enumerate(CATS):
        syn, idx = list_synthetic(c)
        for i, t in enumerate(TYPES):
            k = int(np.flatnonzero((idx["type"] == t).values & (idx["repeat"] == 0).values)[2])
            im = np.asarray(Image.open(syn[k].path).convert("RGB").resize((256, 256), Image.BILINEAR))
            ax[i, j].imshow(im)
            ax[i, j].contour(load_mask(syn[k], 256), levels=[.5], colors="#e07a1f", linewidths=.8)
            ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
            if i == 0:
                ax[i, j].set_title(c, fontsize=9)
            if j == 0:
                ax[i, j].set_ylabel(t, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "figures", "synthetic_examples.png"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["predict", "score", "examples"])
    ap.add_argument("--tag", choices=list(BASES.values()))
    ap.add_argument("--cats", nargs="+", default=CATS)
    args = ap.parse_args()
    if args.step == "predict":
        predict(args.tag, args.cats)
    elif args.step == "score":
        score()
    else:
        examples()


if __name__ == "__main__":
    main()
