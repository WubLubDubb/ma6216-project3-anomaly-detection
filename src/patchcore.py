"""Method B — feature-embedding method: PatchCore (Roth et al., CVPR 2022).

Principle: a frozen ImageNet-pretrained CNN describes every local patch of an
image. All patch features of the normal training images form a "memory bank";
greedy k-center coreset subsampling keeps a small subset that still covers it.
At test time, the anomaly score of a patch is its distance to the nearest
normal patch in the memory bank: a defect patch has no close normal neighbour.

Backbones:
  wrn50       : WideResNet-50-2 (ImageNet), layers {2,3} by default (paper setting)
  dinov2_*    : DINOv2 ViT (self-supervised foundation model) — bonus enhancement
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from data import load_batch


class WRN50Features(torch.nn.Module):
    def __init__(self, layers=(2, 3), input_size=256):
        super().__init__()
        m = torchvision.models.wide_resnet50_2(weights="IMAGENET1K_V1")
        self.stem = torch.nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.blocks = torch.nn.ModuleList([m.layer1, m.layer2, m.layer3, m.layer4])
        self.layers = tuple(layers)
        self.input_size = input_size

    @torch.no_grad()
    def forward(self, x):
        feats, h = [], self.stem(x)
        for i, b in enumerate(self.blocks, start=1):
            h = b(h)
            if i in self.layers:
                feats.append(h)
            if i >= max(self.layers):
                break
        return feats


class DINOv2Features(torch.nn.Module):
    """Patch tokens from several transformer blocks of a DINOv2 ViT (via timm)."""

    def __init__(self, arch="vit_base_patch14_dinov2.lvd142m", blocks=(7, 11), input_size=448):
        super().__init__()
        import timm
        self.m = timm.create_model(arch, pretrained=True, img_size=input_size, num_classes=0)
        self.blocks = list(blocks)
        self.input_size = input_size

    @torch.no_grad()
    def forward(self, x):
        out = self.m.forward_intermediates(x, indices=self.blocks, output_fmt="NCHW",
                                           intermediates_only=True, norm=True)
        return list(out)


def _aggregate(feats, local=3):
    """PatchCore locally-aware patch features: avg-pool each map over a local
    neighbourhood, resize all to the finest map, concatenate along channels."""
    tgt = feats[0].shape[-2:]
    out = []
    for f in feats:
        if local > 1:
            f = F.avg_pool2d(f, local, 1, local // 2)
        if f.shape[-2:] != tgt:
            f = F.interpolate(f, size=tgt, mode="bilinear", align_corners=False)
        out.append(f)
    return torch.cat(out, 1)  # N,D,h,w


def greedy_coreset(feats, ratio, proj_dim=128, device="cuda", seed=0):
    """Approximate greedy k-center selection (as in the PatchCore paper), computed on a
    random projection of the features for speed."""
    n = feats.shape[0]
    k = max(1, int(n * ratio))
    g = torch.Generator(device="cpu").manual_seed(seed)
    P = torch.randn(feats.shape[1], proj_dim, generator=g).to(device) / proj_dim ** 0.5
    z = feats @ P
    sel = [int(torch.randint(0, n, (1,), generator=g))]
    mind = torch.cdist(z, z[sel[-1]][None]).squeeze(1)
    for _ in range(k - 1):
        i = int(torch.argmax(mind))
        sel.append(i)
        mind = torch.minimum(mind, torch.cdist(z, z[i][None]).squeeze(1))
    return torch.tensor(sel, device=device)


class PatchCore:
    def __init__(self, backbone="wrn50", layers=(2, 3), coreset=0.1, out_size=256, device="cuda",
                 dino_arch="vit_base_patch14_dinov2.lvd142m", dino_size=448, input_size=256):
        self.device = device
        if backbone == "wrn50":
            self.fx = WRN50Features(layers, input_size).to(device).eval()
        else:
            self.fx = DINOv2Features(dino_arch, layers, dino_size).to(device).eval()
        self.coreset = coreset
        self.out_size = out_size
        self.bank = None
        self.info = {}

    @torch.no_grad()
    def _embed(self, samples, bs=16):
        embs = []
        for i in range(0, len(samples), bs):
            x = torch.from_numpy(load_batch(samples[i:i + bs], self.fx.input_size, imagenet_norm=True)).to(self.device)
            embs.append(_aggregate(self.fx(x)))
        e = torch.cat(embs)
        self.grid = e.shape[-2:]
        return e  # N,D,h,w

    @torch.no_grad()
    def fit(self, fit_samples):
        t0 = time.time()
        e = self._embed(fit_samples)
        feats = e.permute(0, 2, 3, 1).reshape(-1, e.shape[1])
        n_all = feats.shape[0]
        idx = greedy_coreset(feats, self.coreset, device=self.device)
        self.bank = feats[idx].contiguous()
        del feats, e
        torch.cuda.empty_cache()
        self.info = {"fit_time_s": time.time() - t0, "bank_size": int(self.bank.shape[0]),
                     "n_patches_total": int(n_all), "feat_dim": int(self.bank.shape[1]),
                     "bank_MB": self.bank.numel() * 4 / 2 ** 20, "grid": list(self.grid)}
        return self

    @torch.no_grad()
    def predict(self, samples, bs=16):
        """Raw anomaly maps (N,out,out): nearest-neighbour distance per patch, bilinearly upsampled."""
        maps = []
        for i in range(0, len(samples), bs):
            x = torch.from_numpy(load_batch(samples[i:i + bs], self.fx.input_size, imagenet_norm=True)).to(self.device)
            e = _aggregate(self.fx(x))
            n, d, h, w = e.shape
            q = e.permute(0, 2, 3, 1).reshape(-1, d)
            dist = torch.cat([torch.cdist(q[j:j + 8192], self.bank).min(1).values for j in range(0, len(q), 8192)])
            m = dist.view(n, 1, h, w)
            m = F.interpolate(m, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False)
            maps.append(m[:, 0].cpu().numpy())
        return np.concatenate(maps)
