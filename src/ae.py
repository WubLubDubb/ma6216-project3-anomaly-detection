"""Method A — reconstruction-based baseline: convolutional autoencoder (SSIM-AE style,
Bergmann et al., VISAPP 2019).

Principle: trained only on normal images to reproduce them through a narrow
bottleneck. It learns to reconstruct "normal" appearance; regions it cannot
reproduce (defects) give a large structural (1-SSIM) residual -> anomaly map.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import load_batch


def _gauss_window(size=11, sigma=1.5, device="cpu"):
    g = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(g ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]


def ssim_map(x, y, win):
    """Per-pixel SSIM, averaged over channels. x,y: N,C,H,W in [0,1]."""
    C = x.shape[1]
    w = win.expand(C, 1, -1, -1)
    pad = win.shape[-1] // 2
    mu_x = F.conv2d(x, w, padding=pad, groups=C)
    mu_y = F.conv2d(y, w, padding=pad, groups=C)
    sxx = F.conv2d(x * x, w, padding=pad, groups=C) - mu_x ** 2
    syy = F.conv2d(y * y, w, padding=pad, groups=C) - mu_y ** 2
    sxy = F.conv2d(x * y, w, padding=pad, groups=C) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))
    return s.mean(1)  # N,H,W


def _down(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 4, 2, 1), nn.BatchNorm2d(o), nn.LeakyReLU(0.2, True),
                         nn.Conv2d(o, o, 3, 1, 1), nn.BatchNorm2d(o), nn.LeakyReLU(0.2, True))


def _up(i, o):
    return nn.Sequential(nn.ConvTranspose2d(i, o, 4, 2, 1), nn.BatchNorm2d(o), nn.LeakyReLU(0.2, True),
                         nn.Conv2d(o, o, 3, 1, 1), nn.BatchNorm2d(o), nn.LeakyReLU(0.2, True))


class ConvAE(nn.Module):
    """256x256x3 -> 8x8x256 -> dense latent (z_dim) -> 256x256x3."""

    def __init__(self, z_dim=256):
        super().__init__()
        ch = [3, 32, 64, 128, 256, 256]
        self.enc = nn.Sequential(*[_down(ch[i], ch[i + 1]) for i in range(5)])
        self.fc_e = nn.Linear(256 * 8 * 8, z_dim)
        self.fc_d = nn.Linear(z_dim, 256 * 8 * 8)
        rch = ch[::-1]
        self.dec = nn.Sequential(*[_up(rch[i], rch[i + 1]) for i in range(4)],
                                 nn.ConvTranspose2d(rch[4], 3, 4, 2, 1), nn.Sigmoid())

    def forward(self, x):
        z = self.fc_e(self.enc(x).flatten(1))
        return self.dec(self.fc_d(z).view(-1, 256, 8, 8))


class AEDetector:
    name = "ae"

    def __init__(self, size=256, z_dim=256, epochs=300, bs=16, lr=2e-4, augment=False,
                 device="cuda", seed=0):
        self.size, self.epochs, self.bs, self.lr, self.augment = size, epochs, bs, lr, augment
        self.device = device
        torch.manual_seed(seed)
        self.model = ConvAE(z_dim).to(device)
        self.win = _gauss_window(device=device)
        self.info = {}

    def _aug(self, x):
        # Only geometric flips / 90-degree rotations, applied for textures (isotropic);
        # never for objects, where e.g. a flipped metal nut IS a defect.
        if torch.rand(1) < 0.5:
            x = x.flip(3)
        if torch.rand(1) < 0.5:
            x = x.flip(2)
        return torch.rot90(x, int(torch.randint(0, 4, (1,))), (2, 3))

    def fit(self, fit_samples):
        t0 = time.time()
        X = torch.from_numpy(load_batch(fit_samples, self.size, imagenet_norm=False)).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.epochs)
        self.model.train()
        n = len(X)
        for ep in range(self.epochs):
            perm = torch.randperm(n, device=self.device)
            tot = 0.0
            for i in range(0, n, self.bs):
                xb = X[perm[i:i + self.bs]]
                if self.augment:
                    xb = self._aug(xb)
                rec = self.model(xb)
                loss = F.mse_loss(rec, xb) + (1 - ssim_map(rec, xb, self.win)).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item() * len(xb)
            sched.step()
            if ep % 50 == 0 or ep == self.epochs - 1:
                print(f"  [ae] epoch {ep:4d} loss {tot / n:.4f}", flush=True)
        self.info = {"fit_time_s": time.time() - t0, "n_params": sum(p.numel() for p in self.model.parameters())}
        return self

    @torch.no_grad()
    def predict(self, samples, bs=32, return_recon=False):
        """Returns raw anomaly maps (N,size,size) = 1 - SSIM(x, AE(x)); optionally reconstructions."""
        self.model.eval()
        maps, recs = [], []
        for i in range(0, len(samples), bs):
            x = torch.from_numpy(load_batch(samples[i:i + bs], self.size, imagenet_norm=False)).to(self.device)
            r = self.model(x)
            maps.append((1 - ssim_map(r, x, self.win)).cpu().numpy())
            if return_recon:
                recs.append(r.cpu().numpy())
        maps = np.concatenate(maps)
        return (maps, np.concatenate(recs)) if return_recon else maps
