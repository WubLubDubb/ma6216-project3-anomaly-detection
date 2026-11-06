# MA6216 CA3 — Project 3: Industrial Visual Anomaly Detection (MVTec AD)

Unsupervised anomaly detection and localisation on MVTec AD: the models learn only from
**defect-free** images, then score and localise defects on unseen test images.

> Results tables, figures and failure analysis are added below as experiments finish.

## Methods (two different principles + bonus)

| | Method | Principle | Training |
|---|---|---|---|
| A (baseline) | Convolutional autoencoder (SSIM-AE) | **Reconstruction**: learn to reproduce normal images; anomaly map = 1 − SSIM(x, AE(x)) | 300 epochs from scratch on normal images |
| B | PatchCore (Roth et al., CVPR 2022), WideResNet-50 layers 2+3 | **Feature embedding / memory bank**: anomaly = distance of each patch feature to its nearest normal patch | No training: frozen ImageNet features + greedy coreset (10 %) |
| Bonus | PatchCore with a **DINOv2** ViT-B/14 backbone | Same algorithm, stronger self-supervised foundation-model features | Same |

## Categories

| Category | Type | Defect types | Median defect area |
|---|---|---|---|
| bottle | object | broken_large, broken_small, contamination | 5.7 % |
| metal_nut | object | bent, color, flip, scratch | 3.7 % |
| screw | object | manipulated_front, scratch_head, scratch_neck, thread_side, thread_top | 0.29 % |
| carpet | texture | color, cut, hole, metal_contamination, thread | 1.5 % |
| leather | texture | color, cut, fold, glue, poke | 0.46 % |

## Protocol (what is and is not allowed to see labels)

* `train/good` is split 85 % **fit** / 15 % **val** (seed 0). Only *fit* images train the models.
* **Decision threshold = mean + 3·std of the image scores of the held-out normal *val* images.**
  No test label or mask is used to train, tune or threshold. The same threshold binarises the
  anomaly map for the localisation overlay (image is anomalous ⇔ some pixel exceeds it).
* `image_f1_max_oracle` (best F1 over all thresholds) **uses test labels** and is reported only
  as an upper-bound reference, never as our decision rule.
* Preprocessing: whole image resized to 256×256 (no centre crop, so border defects are kept);
  ImageNet normalisation for PatchCore; [0,1] input for the AE; flips/90° rotations as AE
  augmentation **for textures only** (a flipped metal nut is itself a defect).
  Anomaly maps are Gaussian-smoothed (σ = 4); image score = max of the map.
* Metrics: image AUROC, image F1 (at our threshold), pixel AUROC, AU-PRO (FPR ≤ 0.3,
  reference-implementation definition), all at 256×256.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Dataset: download the categories from the
[MVTec AD page](https://www.mvtec.com/company/research/datasets/mvtec-ad) and extract them to
`data/mvtec_ad/<category>/` inside this repo (or point `MVTEC_ROOT` at another folder). The dataset is
CC BY-NC-SA 4.0 and is **not** included in this repo.

## Run

```bash
.venv/bin/python src/dataset_analysis.py                     # dataset stats + example figures
.venv/bin/python src/run.py --method patchcore --tag patchcore_wrn50
.venv/bin/python src/run.py --method ae --tag ae
.venv/bin/python src/run.py --method patchcore --tag patchcore_dinov2 --backbone dinov2 --layers 7 11
.venv/bin/python src/viz.py --tags ae patchcore_wrn50 patchcore_dinov2 --names AE PatchCore PatchCore-DINOv2
```

Each run writes `results/<tag>/<category>/{metrics.json,scores.csv}` and `results/<tag>/summary.csv`
(anomaly maps `maps.npz` and AE weights are git-ignored; regenerate them with the commands above).

## Code layout

```
src/data.py              MVTec AD file lists, fit/val split, image/mask loading
src/ae.py                Method A: convolutional autoencoder + SSIM anomaly map
src/patchcore.py         Method B: PatchCore (WRN-50 or DINOv2 backbone, greedy coreset)
src/metrics.py           image AUROC, F1, pixel AUROC, AU-PRO, segmentation IoU
src/run.py               fit -> threshold on val normals -> evaluate -> save
src/dataset_analysis.py  split sizes, defect types, defect-area statistics, example grids
src/viz.py               result tables, bar charts, score histograms, heatmap overlays, failure cases
```

## References

* P. Bergmann, M. Fauser, D. Sattlegger, C. Steger. *MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection.* CVPR 2019.
* P. Bergmann, K. Batzner, M. Fauser, D. Sattlegger, C. Steger. *The MVTec Anomaly Detection Dataset.* IJCV 2021.
* P. Bergmann, S. Löwe, M. Fauser, D. Sattlegger, C. Steger. *Improving Unsupervised Defect Segmentation by Applying Structural Similarity to Autoencoders.* VISAPP 2019.
* K. Roth, L. Pemula, J. Zepeda, B. Schölkopf, T. Brox, P. Gehler. *Towards Total Recall in Industrial Anomaly Detection.* CVPR 2022.
* T. Defard, A. Setkov, A. Loesch, R. Audigier. *PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization.* ICPR Workshops 2021.
* M. Oquab et al. *DINOv2: Learning Robust Visual Features without Supervision.* TMLR 2024.
