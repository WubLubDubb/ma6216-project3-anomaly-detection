"""Method C — knowledge distillation: Reverse Distillation (RD4AD, Deng & Li, CVPR 2022).

Principle: a frozen ImageNet teacher (WideResNet-50-2) extracts multi-scale
features. A trainable "one-class bottleneck" (OCBE) compresses them into a
compact embedding, and a *reverse* student decoder must reconstruct the
teacher's layer1/2/3 features from that embedding. Trained on normal images
only, the student learns to reproduce teacher features of normal patterns;
on a defect, teacher and student features disagree -> anomaly map = sum over
scales of (1 - cosine similarity), upsampled to image size.

Speed-up (same maths as the paper): the teacher is frozen and no augmentation
is used, so its features are computed once and cached on the GPU in fp16;
training uses bf16 autocast.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.resnet import Bottleneck

from data import load_batch


class Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        m = torchvision.models.wide_resnet50_2(weights="IMAGENET1K_V1")
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.l1, self.l2, self.l3 = m.layer1, m.layer2, m.layer3

    @torch.no_grad()
    def forward(self, x):
        a = self.l1(self.stem(x))
        b = self.l2(a)
        return [a, b, self.l3(b)]  # 256@64, 512@32, 1024@16 for a 256x256 input


def _conv_bn_relu(i, o, k, s):
    return nn.Sequential(nn.Conv2d(i, o, k, s, k // 2, bias=False), nn.BatchNorm2d(o), nn.ReLU(True))


class OCBE(nn.Module):
    """Multi-scale feature fusion + one-class embedding (wide bottleneck blocks, stride 2)."""

    def __init__(self):
        super().__init__()
        self.a = nn.Sequential(_conv_bn_relu(256, 512, 3, 2), _conv_bn_relu(512, 1024, 3, 2))
        self.b = _conv_bn_relu(512, 1024, 3, 2)
        down = nn.Sequential(nn.Conv2d(3072, 2048, 1, 2, bias=False), nn.BatchNorm2d(2048))
        self.emb = nn.Sequential(Bottleneck(3072, 512, 2, down, base_width=128),
                                 Bottleneck(2048, 512, base_width=128),
                                 Bottleneck(2048, 512, base_width=128))

    def forward(self, f):
        return self.emb(torch.cat([self.a(f[0]), self.b(f[1]), f[2]], 1))  # 2048@8


class DeBottleneck(nn.Module):
    """Bottleneck whose 3x3 conv is replaced by a 2x2 transposed conv when upsampling."""
    expansion = 4

    def __init__(self, inplanes, planes, up=False):
        super().__init__()
        w = planes * 2  # "wide" variant
        out = planes * 4
        self.c1 = nn.Sequential(nn.Conv2d(inplanes, w, 1, bias=False), nn.BatchNorm2d(w), nn.ReLU(True))
        conv2 = nn.ConvTranspose2d(w, w, 2, 2, bias=False) if up else nn.Conv2d(w, w, 3, 1, 1, bias=False)
        self.c2 = nn.Sequential(conv2, nn.BatchNorm2d(w), nn.ReLU(True))
        self.c3 = nn.Sequential(nn.Conv2d(w, out, 1, bias=False), nn.BatchNorm2d(out))
        self.short = None
        if up or inplanes != out:
            sc = nn.ConvTranspose2d(inplanes, out, 2, 2, bias=False) if up else nn.Conv2d(inplanes, out, 1, bias=False)
            self.short = nn.Sequential(sc, nn.BatchNorm2d(out))

    def forward(self, x):
        idt = x if self.short is None else self.short(x)
        return F.relu(self.c3(self.c2(self.c1(x))) + idt)


def _stage(inplanes, planes, n):
    layers = [DeBottleneck(inplanes, planes, up=True)]
    layers += [DeBottleneck(planes * 4, planes) for _ in range(n - 1)]
    return nn.Sequential(*layers)


class Student(nn.Module):
    """Reverse WideResNet-50 decoder: 2048@8 -> 1024@16 -> 512@32 -> 256@64."""

    def __init__(self):
        super().__init__()
        self.s3 = _stage(2048, 256, 3)
        self.s2 = _stage(1024, 128, 4)
        self.s1 = _stage(512, 64, 6)

    def forward(self, z):
        c = self.s3(z)
        b = self.s2(c)
        a = self.s1(b)
        return [a, b, c]


def rd_loss(ft, fs):
    # as in the official implementation: cosine distance of the flattened feature maps, per scale
    return sum((1 - F.cosine_similarity(t.flatten(1), s.flatten(1), dim=1)).mean() for t, s in zip(ft, fs))


class RDDetector:
    name = "rd"

    def __init__(self, size=256, epochs=200, bs=16, lr=0.005, device="cuda", seed=0):
        torch.manual_seed(seed)
        self.size, self.epochs, self.bs, self.lr, self.device = size, epochs, bs, lr, device
        self.teacher = Teacher().to(device).eval()
        self.ocbe = OCBE().to(device)
        self.student = Student().to(device)
        self.info = {}

    @torch.no_grad()
    def _teacher_feats(self, samples, bs=16):
        out = [[], [], []]
        for i in range(0, len(samples), bs):
            x = torch.from_numpy(load_batch(samples[i:i + bs], self.size, imagenet_norm=True)).to(self.device)
            for k, f in enumerate(self.teacher(x)):
                out[k].append(f.half())
        return [torch.cat(o) for o in out]

    def fit(self, fit_samples):
        t0 = time.time()
        T = self._teacher_feats(fit_samples)
        params = list(self.ocbe.parameters()) + list(self.student.parameters())
        opt = torch.optim.Adam(params, lr=self.lr, betas=(0.5, 0.999))
        n = T[0].shape[0]
        self.ocbe.train(); self.student.train()
        for ep in range(self.epochs):
            perm = torch.randperm(n, device=self.device)
            tot = 0.0
            for i in range(0, n, self.bs):
                idx = perm[i:i + self.bs]
                if len(idx) < 2:  # BatchNorm needs >1 sample
                    continue
                ft = [t[idx].float() for t in T]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    fs = self.student(self.ocbe(ft))
                loss = rd_loss(ft, [f.float() for f in fs])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                tot += loss.item() * len(idx)
            if ep % 25 == 0 or ep == self.epochs - 1:
                print(f"  [rd] epoch {ep:4d} loss {tot / n:.4f}", flush=True)
        del T
        torch.cuda.empty_cache()
        n_par = sum(p.numel() for p in params)
        self.info = {"fit_time_s": time.time() - t0, "n_params_trainable": n_par}
        return self

    @torch.no_grad()
    def predict(self, samples, bs=16):
        self.ocbe.eval(); self.student.eval()
        maps = []
        for i in range(0, len(samples), bs):
            x = torch.from_numpy(load_batch(samples[i:i + bs], self.size, imagenet_norm=True)).to(self.device)
            ft = self.teacher(x)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                fs = self.student(self.ocbe(ft))
            m = 0
            for t, s in zip(ft, fs):
                d = 1 - F.cosine_similarity(t, s.float(), dim=1)[:, None]
                m = m + F.interpolate(d, size=(self.size, self.size), mode="bilinear", align_corners=False)
            maps.append(m[:, 0].cpu().numpy())
        return np.concatenate(maps)
