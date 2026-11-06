"""Label-free foreground estimation for object categories.

Objects in MVTec AD are photographed on a fairly uniform background. We split
the grey image into two classes with Otsu's threshold, call "background" the
class that covers most of the image border, fill holes, keep the largest
connected component and dilate it (so defects on the object outline, e.g. a
broken screw tip, stay inside). Only the input image itself is used — no
labels, no masks. Texture categories have no background, so they get an
all-ones mask.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu

from data import OBJECTS, load_image


def foreground_mask(path: str, size: int = 256, dilate_px: int = 12) -> np.ndarray:
    return mask_from_image(load_image(path, size), dilate_px)


def mask_from_image(img: np.ndarray, dilate_px: int = 12) -> np.ndarray:
    """img: HxWx3 in [0,1] (already resized)."""
    g = ndi.gaussian_filter(img.mean(2), 2)
    fg = g > threshold_otsu(g)
    border = np.concatenate([fg[0], fg[-1], fg[:, 0], fg[:, -1]])
    if border.mean() > 0.5:          # bright class touches the border -> it is the background
        fg = ~fg
    fg = ndi.binary_fill_holes(fg)
    lab, n = ndi.label(fg)
    if n > 1:
        sizes = ndi.sum(fg, lab, range(1, n + 1))
        fg = lab == (1 + int(np.argmax(sizes)))
    if dilate_px > 0:                # (iterations=0 would mean "dilate until nothing changes" in scipy)
        fg = ndi.binary_dilation(fg, iterations=dilate_px)
    if fg.mean() < 0.02:             # segmentation failed -> do not mask
        fg[:] = True
    return fg


def masks_for(samples, category: str, size: int = 256) -> np.ndarray:
    if category not in OBJECTS:
        return np.ones((len(samples), size, size), dtype=bool)
    return np.stack([foreground_mask(s.path, size) for s in samples])


def apply(maps: np.ndarray, fg: np.ndarray) -> np.ndarray:
    """Background pixels get the per-image minimum of the map (i.e. 'as normal as it gets')."""
    out = maps.copy()
    mins = maps.reshape(len(maps), -1).min(1)
    for i in range(len(maps)):
        out[i][~fg[i]] = mins[i]
    return out
