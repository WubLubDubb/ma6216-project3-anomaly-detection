# MA6216 CA3 — Project 3: Industrial Visual Anomaly Detection (MVTec AD)

Unsupervised anomaly detection and localisation on MVTec AD. All models learn only from
**defect-free** images, then score and localise defects on unseen test images. No test label or
mask is used to train a model, fit a parameter, set the decision threshold or choose the final
model: our enhanced model was picked among ten variants on synthetic defects added to held-out
normal images (see Protocol and Bonus).

## Headline results (5 categories, mean)

| Method | Principle | Image AUROC | Image F1 | Pixel AUROC | AU-PRO | ms / image |
|---|---|---|---|---|---|---|
| AE (SSIM autoencoder) | reconstruction | 0.683 | 0.186 | 0.866 | 0.641 | 3 |
| RD4AD (Reverse Distillation) | knowledge distillation | **0.995** | 0.953 | 0.989 | **0.966** | 18 |
| PatchCore (WRN-50, 256) | feature memory bank | 0.986 | 0.936 | 0.990 | 0.950 | 12–17 |
| **Ours** — PatchCore + DINOv2 + mask (bonus) | memory bank of self-supervised ViT features | 0.991 | **0.969** | **0.992** | 0.965 | 47–54 |

Image F1 uses our label-free threshold (mean + 3σ of held-out normal scores). Timing: NVIDIA L4,
batch size 1, images pre-loaded (no disk I/O or decoding): feature extraction + nearest-neighbour
search or forward pass, plus the CPU foreground mask for Ours (8–9 ms); smoothing excluded. Measured
on bottle (smallest memory bank) and screw (largest), `src/benchmark.py` →
`results/figures/benchmark_<category>_bs1.json`. The `ms_per_img` column in the per-run CSVs is
end-to-end on a shared GPU (includes disk I/O) and is not comparable.

**Ours = PatchCore with a DINOv2 ViT-B/14 backbone (self-supervised; blocks 7 + 11, 448 px input)
and label-free masking of the background on object categories.** It was selected among ten PatchCore
variants without any test label. A fusion of WRN-50 @320 and DINOv2 has a higher *test* image AUROC
(0.996) but was not selected; choosing it would have meant choosing on the test set. It is reported in
the ablation.

### Per category

Image AUROC / Pixel AUROC / AU-PRO:

| Category | Type | AE | RD | PatchCore | **Ours** |
|---|---|---|---|---|---|
| bottle | object | 0.967 / 0.941 / 0.861 | 1.000 / 0.987 / 0.963 | 1.000 / 0.987 / 0.947 | 1.000 / 0.992 / 0.962 |
| metal_nut | object | 0.596 / 0.911 / 0.635 | 1.000 / 0.976 / 0.952 | 0.996 / 0.989 / 0.943 | 1.000 / 0.987 / 0.936 |
| screw | object | 0.575 / 0.978 / 0.907 | 0.989 / 0.995 / 0.975 | 0.949 / 0.990 / 0.942 | 0.954 / 0.994 / 0.970 |
| carpet | texture | 0.662 / 0.698 / 0.216 | 0.987 / 0.990 / 0.953 | 0.987 / 0.990 / 0.947 | 1.000 / 0.996 / 0.982 |
| leather | texture | 0.613 / 0.804 / 0.589 | 1.000 / 0.994 / 0.984 | 1.000 / 0.993 / 0.969 | 1.000 / 0.992 / 0.974 |

### How many defective parts are caught? (decision rule: mean + 3σ)

| Method | Defects detected | False alarms on good parts | Accuracy |
|---|---|---|---|
| AE | 87 / 456 (19.1 %) | 28 / 143 (19.6 %) | 33.7 % |
| RD | 415 / 456 (91.0 %) | 5 / 143 (3.5 %) | 92.3 % |
| PatchCore | 412 / 456 (90.4 %) | 19 / 143 (13.3 %) | 89.5 % |
| **Ours** | **428 / 456 (93.9 %)** | **4 / 143 (2.8 %)** | **94.7 %** |

Per category (Ours): bottle 63/63 detected, 1/20 false alarms · metal_nut 93/93, 0/22 · screw 91/119, 3/41 ·
carpet 89/89, 0/28 · leather 92/92, 0/32. With the secondary rule (95th percentile of validation normals,
i.e. a 5 % false-alarm budget) Ours detects 442/456 (96.9 %) at 11/143 false alarms.

Ours scores 1.000 image AUROC (defect type vs. good) on 17 of the 22 defect types; the other five are
the five screw types (0.913 – 0.995; `manipulated_front` 0.913 and `thread_side` 0.917 are the hardest).
See `results/figures/per_defect_pivot.csv`.

## Methods

| | Method | How it models "normal" | How it scores a test image |
|---|---|---|---|
| A | **SSIM autoencoder** (Bergmann et al., VISAPP 2019) — `src/ae.py` | Conv encoder → 256-d bottleneck → decoder, trained 300 epochs on normal images (MSE + SSIM loss) | 1 − SSIM(x, AE(x)) per pixel: what the AE cannot reproduce |
| B | **PatchCore** (Roth et al., CVPR 2022) — `src/patchcore.py` | Frozen ImageNet WRN-50 features (layers 2+3, 3×3 local aggregation) of all normal patches; greedy k-center coreset keeps 10 % as memory bank | Distance of every test patch to its nearest normal patch |
| C | **Reverse Distillation** (Deng & Li, CVPR 2022) — `src/rd.py` | Frozen WRN-50 teacher; trainable one-class bottleneck + reverse student decoder learn to reproduce teacher layer1–3 features of normal images (200 epochs) | Σ over scales of (1 − cosine similarity) between teacher and student features |
| Bonus | **Ours** (`src/patchcore.py` with the DINOv2 backbone, `src/foreground.py`) | PatchCore memory bank of DINOv2 ViT-B/14 patch tokens (blocks 7 + 11, 448 px input; self-supervised pre-training, no labels) | Distance to the nearest normal patch, as in B; background suppressed on object categories by a label-free Otsu mask |

Why these methods suit our categories: the AE was expected to work on centred, aligned objects
(bottle) but to struggle with stochastic textures, which it cannot reproduce. PatchCore compares
*local* patch features, which suits both textures (local statistics) and small local defects (screw,
leather), and it needs no training per product. RD compares teacher and student features at three
scales, which covers defects of very different sizes (0.2 % to 48 % of the image), and it is fast at
inference. For the bonus, DINOv2's self-supervised features describe texture and local shape better
than ImageNet-classification features (best on carpet and leather), and the foreground mask removes
the background responses that otherwise dominate on screw.

### Literature considered

| Family | Examples | Idea | Why (not) chosen |
|---|---|---|---|
| Reconstruction | AE / SSIM-AE, VAE, DRAEM | Reproduce normal images; residual = anomaly | Simplest baseline (**A**); known to generalise too well / blur textures |
| Feature embedding | PaDiM, PatchCore, SPADE | Pretrained patch features + statistics or memory bank | State of the art, no training, explainable (**B**, bonus) |
| Knowledge distillation | US, STFPM, RD4AD, EfficientAD | Student imitates a pretrained teacher on normal data only | Different principle, strong accuracy, fast inference (**C**) |
| Normalizing flows | DifferNet, CFlow, FastFlow | Exact likelihood of normal features | Strong but heavier to train/tune in the available time |
| Synthetic anomalies | CutPaste, DRAEM, SimpleNet | Train a classifier on artificially created defects | Not used to train; CutPaste / DRAEM-style defects only to *select* our variant without labels |
| Foundation / zero-shot | WinCLIP, AnomalyCLIP, DINOv2 features | Large pretrained vision(-language) models | DINOv2 is the backbone of Ours |

## Protocol: what may see labels

* `train/good` is split 85 % **fit** / 15 % **val** (seed 0). Only *fit* images build the models.
* **Decision threshold = mean + 3·std of the image scores of the held-out normal *val* images**
  (pre-registered primary rule). Secondary rule, also label-free: 95th percentile of the same scores.
  The same value binarises the anomaly map for the overlays (image anomalous ⇔ some pixel ≥ threshold).
* `image_f1_max_oracle` in the CSVs is the best F1 over all thresholds; it **uses test labels** and is an
  upper-bound reference only, never a decision rule.
* Preprocessing: whole image resized to 256×256 (no centre crop, so border defects are kept; PatchCore
  @320 / DINOv2 @448 read larger inputs, all maps are evaluated at 256×256); ImageNet normalisation for
  the pretrained networks; [0,1] input for the AE; flips / 90° rotations as AE augmentation **for
  textures only** (a flipped metal nut *is* a defect). Maps are Gaussian-smoothed (σ = 4); image score
  = max of the map.
* Metrics: image AUROC, image F1, pixel AUROC, AU-PRO (normalised area under the per-region-overlap
  curve up to FPR 0.3, reference-implementation definition), at 256×256. Pixel AUROC is computed on a
  uniform random sample of 4 M pixels per category (difference to the exact value ≤ 3·10⁻⁴). Our
  AU-PRO was checked against an independent threshold-sweep implementation (agreement to 10⁻⁴).
* No per-category tuning: every setting is the same for all categories. The only type-level rules come
  from the dataset metadata: the foreground mask is applied to object categories, and AE flip/rotation
  augmentation to textures.
* The secondary 95th-percentile rule was added after the first runs. It is label-free and does not
  change the primary rule (mean + 3σ, σ = 4 smoothing, image score = max), which was fixed from the
  first commit.
* All hyperparameters (smoothing σ = 4, 10 % coreset, 320 px input, DINOv2 blocks 7 + 11, 12 px mask
  dilation, AE/RD epochs) were fixed in advance and never searched.
* **Model selection without test labels** (`src/synthetic.py`, `src/synth_select.py`). Every held-out
  normal *val* image receives six synthetic defects: 2 × CutPaste, 2 × CutPaste-scar and 2 × DRAEM-style
  (Perlin-noise mask filled with a foreign texture taken from the normal training images of the other
  categories). The generator settings are the published defaults; on objects the defect is placed on
  the part (label-free Otsu foreground). This gives 1,146 synthetic images next to the 191 val normals.
  The rule was fixed before the run, and the run was done once: **pick the variant with the highest mean
  image AUROC of val normals vs. their synthetic versions**. DINOv2 + mask wins (0.974) in all 1,000
  bootstrap resamples of the val images. Test results were read only afterwards. The re-fitted models
  reproduce the reported runs (val-score correlation > 0.9999999).
* **Remaining caveats.** (1) The *ideas* for the enhancements (mask, resolution, DINOv2) came from
  inspecting test failure cases of the baselines, e.g. background false alarms on screw. (2) The
  synthetic proxy ranks the ten variants quite differently from the test set (Spearman ρ = 0.15): cut-and-
  paste of carpet onto carpet is almost invisible to WRN-50 (synthetic carpet AUROC 0.73–0.75 vs 1.000 on
  real carpet defects), and synthetic screw defects are easier than the real ones (all variants ≈ 0.99).
  The selected model is therefore not the one with the best test AUROC; its test numbers played no part
  in choosing it. The AE, RD and PatchCore baselines involve no selection.

## Dataset analysis

| Category | Type | Train (fit / val) | Test good / defect | Defect types | Median defect area |
|---|---|---|---|---|---|
| bottle | object | 178 / 31 | 20 / 63 | broken_large, broken_small, contamination | 5.7 % |
| metal_nut | object | 187 / 33 | 22 / 93 | bent, color, flip, scratch | 3.7 % (flip ≈ 48 %) |
| screw | object | 272 / 48 | 41 / 119 | manipulated_front, scratch_head, scratch_neck, thread_side, thread_top | **0.29 %** |
| carpet | texture | 238 / 42 | 28 / 89 | color, cut, hole, metal_contamination, thread | 1.5 % |
| leather | texture | 208 / 37 | 32 / 92 | color, cut, fold, glue, poke | 0.46 % (poke 0.23 %) |

How the characteristics matter (see `results/dataset/`):

* **Object geometry / pose**: screws lie at arbitrary orientation, so "normal" covers many poses; small
  defects must be separated from pose variation. Bottle and metal nut are centred and aligned → easy.
* **Background variation**: screw sits on a light, unevenly lit background; patches there look novel to
  a memory bank and caused false alarms → foreground masking (bonus).
* **Defect size**: screw and leather defects cover < 0.5 % of the image; at 256 px a screw defect spans
  only ~2×2 WRN-50 layer-2 patches → higher resolution helps (bonus).
* **Surface texture**: carpet/leather are stochastic textures; an AE cannot reproduce them and its
  residual is high everywhere, while patch-feature methods handle them almost perfectly.
* **Logical vs structural defects**: metal_nut `flip` and `color` are globally plausible shapes; a
  reconstruction model reproduces a "normal-looking" nut and misses them.

## Bonus: enhancement of PatchCore (ablation, all variants)

The first column is the selection criterion (no labels); the other columns are test results, reported
for every variant, including those that score higher on the test set than the selected one.

| Variant | Synthetic val AUROC (selection) | Test image AUROC | Image F1 | Pixel AUROC | AU-PRO | screw I-AUROC |
|---|---|---|---|---|---|---|
| PatchCore WRN-50 @256 (baseline) | 0.923 | 0.986 | 0.936 | 0.990 | 0.950 | 0.949 |
| WRN-50 @256 + fg mask | 0.923 | 0.987 | 0.930 | 0.990 | 0.949 | 0.952 |
| WRN-50 @320 | 0.932 | 0.995 | 0.947 | 0.991 | 0.962 | 0.978 |
| WRN-50 @320 + fg mask | 0.932 | 0.994 | 0.944 | 0.991 | 0.958 | 0.975 |
| DINOv2 | 0.972 | 0.972 | 0.953 | 0.987 | 0.944 | 0.859 |
| **DINOv2 + fg mask (Ours: label-free pick)** | **0.974** | 0.991 | **0.969** | 0.992 | 0.965 | 0.954 |
| Fusion WRN-50 + DINOv2 | 0.960 | 0.987 | 0.958 | 0.991 | 0.957 | 0.936 |
| Fusion WRN-50 + DINOv2 + fg mask | 0.961 | 0.995 | 0.966 | 0.992 | 0.962 | 0.976 |
| Fusion WRN-50@320 + DINOv2 | 0.960 | 0.992 | 0.962 | 0.992 | 0.963 | 0.959 |
| Fusion WRN-50@320 + DINOv2 + fg mask | 0.961 | **0.996** | 0.962 | **0.993** | **0.965** | **0.981** |
| *Extra: + RD (3-model ensemble; not a candidate)* | — | *0.998* | *0.966* | *0.992* | *0.967* | *0.988* |

Fusion = each map standardised by its own validation-normal score statistics, then averaged
(`src/fuse.py`). The RD ensemble was built before the label-free selection and is shown only for
completeness (RD maps of the synthetic set were not computed).

Findings:

1. **The label-free selection picks DINOv2 + mask.** On the test set it has the best F1 (0.969), the best
   accuracy (94.7 %) and the fewest false alarms (4/143) of all variants, but not the best AUROC.
2. **DINOv2 alone is better on textures and metal_nut but much worse on screw** (0.859). Its failure is
   mostly background: with the foreground mask, DINOv2 on screw rises to 0.954.
3. **Resolution is the most effective change for WRN-50** (test +0.008 image AUROC, screw 0.949 → 0.978):
   small defects need a finer patch grid (40×40 instead of 32×32). The synthetic set cannot show this,
   because its screw defects are already detected by every variant (≈ 0.99).
4. **Fusion with WRN-50 @320 has the best test AUROC** (0.996, screw 0.981) but 18 false alarms on carpet
   and leather, against 0 for Ours: its WRN-50 branch reacts to the acquisition shift of the texture
   normals (see Failure analysis). It is also slower (72–91 ms/image).
5. Foreground masking barely changes WRN-50 alone and lowers metal_nut AU-PRO (0.943 → 0.919),
   because part of the `bent` ground truth marks where material is *missing*, i.e. background.
6. Cost: Ours runs one backbone plus a CPU mask: 47–54 ms/image (19–21 images/s on one L4) vs 12–17 ms
   for the WRN-50 baseline.

Figures: `results/figures/ablation.png`, `selection_synthetic.png` (synthetic vs. test AUROC of every
variant), `synthetic_examples.png`; tables: `results/selection/selection.csv`, `selection_per_category.csv`.

## Failure analysis

Figures: `results/figures/fail_<category>.png` (missed defects / false alarms of Ours, with the other
methods' maps side by side), `qual_<category>.png` (one example per defect type),
`ae_reconstruction.png`, `hist_<method>.png` (score distributions with threshold).

* **AE fails on textures and logical defects** (`ae_reconstruction.png`): it reconstructs carpet as a
  blurred periodic pattern, so 1 − SSIM is high everywhere on *normal* images too; on metal_nut it
  reproduces a "normal" nut for `flip`, and normal nuts already have large residuals on their specular
  edges. Normal and defective scores overlap → image AUROC 0.6–0.7, and mean + 3σ classifies almost
  everything as normal.
* **Missed defects (Ours): small screw defects.** 28 of 119 defective screws are missed at the primary
  threshold (`manipulated_front` 10, `thread_side` 9, `scratch_head` 8, `scratch_neck` 1). These
  deformations cover < 0.3 % of the image and give a weaker response than the natural pose and lighting
  variation of the thread. Screw is where Ours is weakest (image AUROC 0.954, vs RD 0.989 and the
  WRN-50 @320 variants 0.975–0.981): DINOv2 at 448 px sees a 32×32 patch grid, coarser than WRN-50 @320
  (40×40).
* **False alarms (Ours): 4 of 143** (bottle 1, screw 3). The strongest one is a compact local response
  on the shank of a normal screw (`results/figures/slides/failures.png`).
* **Acquisition shift of the texture normals.** The scores of `test/good` carpet and leather images are
  significantly higher than those of our held-out `train/good` validation images for every feature
  method (Mann-Whitney p ≤ 2.4·10⁻⁴ for PatchCore, RD and the fusion; Ours: leather p = 3·10⁻⁵,
  carpet p = 0.012; `src/score_shift_test.py` → `results/figures/score_shift_test.csv`). For PatchCore
  (18) and the WRN-50 @320 + DINOv2 fusion (18) this pushes normal textures over the threshold although
  their image AUROC is 1.000. For Ours the shifted normals stay below the threshold (0 texture false
  alarms): measured in standard deviations of the validation scores, the shift of the median is only
  0.07 σ (carpet) and 0.66 σ (leather) for Ours, against 1.1–2.5 σ for PatchCore and the fusion, and no
  normal test texture exceeds 2.5 σ. This is still a calibration risk, not a ranking problem: on a real
  line the threshold should be re-calibrated on normal parts from the actual production conditions.
* **Border artefacts of RD and AE on textures.** For RD, the maximum of the anomaly map lies within
  8 px of the image border on 50 % (carpet) and 72 % (leather) of the normal test images, and 3 of its
  4 carpet false alarms are such border responses. AE on leather: 69 % of normal images, and 21 of its
  28 false alarms. PatchCore and Ours: 0 %. So part of RD's and AE's texture false alarms comes from
  border effects of the convolutional decoders, not only from the acquisition shift.
* **metal_nut `bent`**: localisation is penalised when the ground truth includes the region where a
  bent tab *should* be (now background) — the foreground mask removes exactly that region.

## Conclusions and industrial applicability

* Feature-based methods (PatchCore, RD) clearly beat reconstruction: pretrained features describe
  local appearance robustly, while an AE either reproduces defects or fails to reproduce normal texture.
* **PatchCore**: no training (fit = feature extraction + coreset, 4–8 min per product on a shared L4),
  easy to explain ("this patch has no similar normal patch"), memory bank 107–163 MB per product.
  Ours (DINOv2 + mask, chosen without test labels) has the best F1 (0.969), accuracy (94.7 %) and pixel
  AUROC (0.992) and the fewest false alarms (4/143, 2.8 %); it catches 428 of 456 defects.
* **RD**: needs 6–17 min of GPU training per product (measured on a shared L4), but is fast at inference
  (18 ms) and has the highest image AUROC (0.995) and AU-PRO (0.966), and the best screw result (0.989).
* **AE**: fastest (3 ms) but not reliable enough for inspection on these categories.
* Main open issues for deployment: very small defects on complex geometry (screw threads: higher
  resolution helps), threshold calibration under acquisition shift (see failure analysis), and model
  selection without real defects: our synthetic-defect check is label-free but agrees only weakly with
  the test ranking (ρ = 0.15).

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Dataset: download the categories from the
[MVTec AD page](https://www.mvtec.com/company/research/datasets/mvtec-ad) and extract them to
`data/mvtec_ad/<category>/` inside this repo (or point `MVTEC_ROOT` at another folder). The dataset is
CC BY-NC-SA 4.0 and is **not** included in this repo.

## Reproduce

```bash
P=.venv/bin/python
$P src/dataset_analysis.py                                                     # dataset statistics + examples
$P src/run.py --method ae --tag ae                                             # A
$P src/run.py --method patchcore --tag patchcore_wrn50                         # B
$P src/run.py --method rd --tag rd --epochs 200                                # C
$P src/run.py --method patchcore --tag patchcore_wrn50_320 --pc_input 320      # bonus: resolution
$P src/run.py --method patchcore --tag patchcore_dinov2 --backbone dinov2 --layers 7 11   # bonus: DINOv2
$P src/reeval.py --tags patchcore_wrn50 patchcore_wrn50_320 patchcore_dinov2 rd --fg --out_suffix _fg
$P src/fuse.py --tags patchcore_wrn50 patchcore_dinov2 --out fusion_wrn50_dinov2
$P src/fuse.py --tags patchcore_wrn50_fg patchcore_dinov2_fg --out fusion_wrn50_dinov2_fg
$P src/fuse.py --tags patchcore_wrn50_320 patchcore_dinov2 --out fusion_wrn320_dinov2
$P src/fuse.py --tags patchcore_wrn50_320_fg patchcore_dinov2_fg --out fusion_wrn320_dinov2_fg
$P src/fuse.py --tags rd_fg patchcore_wrn50_320_fg patchcore_dinov2_fg --out fusion_rd_wrn320_dinov2_fg
# label-free model selection: synthetic defects on held-out normals (images -> data/mvtec_synth/)
$P src/synthetic.py
for t in patchcore_wrn50 patchcore_wrn50_320 patchcore_dinov2; do $P src/synth_select.py predict --tag $t; done
$P src/synth_select.py score && $P src/synth_select.py examples      # -> picks patchcore_dinov2_fg (Ours)
$P src/viz.py --tags ae rd patchcore_wrn50 patchcore_dinov2_fg --names AE RD PatchCore Ours
$P src/ablation.py
$P src/ae_recon_fig.py
$P src/benchmark.py --bs 1 --cat bottle && $P src/benchmark.py --bs 1 --cat screw
$P src/slide_figures.py                                                        # presentation figures
$P src/score_shift_test.py                                                     # normal-score shift + border artefacts
```

Each run writes `results/<tag>/<category>/{metrics.json,scores.csv}` and `results/<tag>/summary.csv`.
Anomaly maps (`maps.npz`), AE weights, PatchCore memory banks (`bank.pt`) and the synthetic images are
git-ignored (large); regenerate them with the commands above. Training logs are in `logs/`.

## Code layout

```
src/data.py              MVTec AD file lists, fit/val split, image/mask loading
src/ae.py                A: convolutional autoencoder + SSIM anomaly map
src/patchcore.py         B: PatchCore (WRN-50 or DINOv2 backbone, greedy coreset)
src/rd.py                C: Reverse Distillation (teacher, OCBE bottleneck, reverse student)
src/foreground.py        label-free foreground masks for object categories (Otsu)
src/fuse.py              label-free score-level fusion of several methods
src/synthetic.py         synthetic defects (CutPaste, scar, DRAEM-style) on held-out normal images
src/synth_select.py      label-free selection of the final variant on the synthetic defects
src/metrics.py           image AUROC, F1, pixel AUROC, AU-PRO, segmentation IoU
src/run.py               fit -> threshold on val normals -> evaluate -> save
src/reeval.py            recompute metrics from saved maps (post-processing variants)
src/dataset_analysis.py  split sizes, defect types, defect-area statistics, example grids
src/viz.py               result tables, bar charts, histograms, heatmap overlays, failure cases
src/ablation.py          bonus ablation table + chart
src/ae_recon_fig.py      AE input / reconstruction / residual figure
src/benchmark.py         inference speed on an idle GPU
src/slide_figures.py     landscape figures for the presentation (results/figures/slides/)
src/score_shift_test.py  Mann-Whitney test of val-normal vs test-normal scores; border-artefact counts
```

## References

* P. Bergmann, M. Fauser, D. Sattlegger, C. Steger. *MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection.* CVPR 2019.
* P. Bergmann, K. Batzner, M. Fauser, D. Sattlegger, C. Steger. *The MVTec Anomaly Detection Dataset.* IJCV 2021.
* P. Bergmann, S. Löwe, M. Fauser, D. Sattlegger, C. Steger. *Improving Unsupervised Defect Segmentation by Applying Structural Similarity to Autoencoders.* VISAPP 2019.
* K. Roth, L. Pemula, J. Zepeda, B. Schölkopf, T. Brox, P. Gehler. *Towards Total Recall in Industrial Anomaly Detection.* CVPR 2022.
* H. Deng, X. Li. *Anomaly Detection via Reverse Distillation from One-Class Embedding.* CVPR 2022.
* T. Defard, A. Setkov, A. Loesch, R. Audigier. *PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization.* ICPR Workshops 2021.
* M. Oquab et al. *DINOv2: Learning Robust Visual Features without Supervision.* TMLR 2024.
* V. Zavrtanik, M. Kristan, D. Skočaj. *DRAEM — A Discriminatively Trained Reconstruction Embedding for Surface Anomaly Detection.* ICCV 2021.
* C.-L. Li, K. Sohn, J. Yoon, T. Pfister. *CutPaste: Self-Supervised Learning for Anomaly Detection and Localization.* CVPR 2021.
* M. Rudolph, B. Wandt, B. Rosenhahn. *Same Same But DifferNet.* WACV 2021.
* K. Batzner, L. Heckler, R. König. *EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies.* WACV 2024.
* J. Jeong et al. *WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation.* CVPR 2023.
* N. Cohen, Y. Hoshen. *Sub-Image Anomaly Detection with Deep Pyramid Correspondences (SPADE).* arXiv 2020.
* G. Wang, S. Han, E. Ding, D. Huang. *Student-Teacher Feature Pyramid Matching for Anomaly Detection (STFPM).* BMVC 2021.
* D. Gudovskiy, S. Ishizaka, K. Kozuka. *CFLOW-AD: Real-Time Unsupervised Anomaly Detection with Localization via Conditional Normalizing Flows.* WACV 2022.
* J. Yu et al. *FastFlow: Unsupervised Anomaly Detection and Localization via 2D Normalizing Flows.* arXiv 2021.
* Z. Liu, Y. Zhou, Y. Xu, Z. Wang. *SimpleNet: A Simple Network for Image Anomaly Detection and Localization.* CVPR 2023.
