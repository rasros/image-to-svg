from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageChops, ImageFilter, ImageStat

@dataclass(frozen=True)
class ScoreConfig:
    target_long_side: int = 256  # scoring resolution (controls shift sensitivity & speed)
    w_struct: float = 0.55
    w_color: float = 0.35
    w_edge: float = 0.10
    ssim_c1: float = 0.01**2
    ssim_c2: float = 0.03**2
    ssim_blur_radius: float = 1.0


_CFG = ScoreConfig()


def _resize_long_side(im: Image.Image, long_side: int) -> Image.Image:
    w, h = im.size
    if max(w, h) <= long_side:
        return im
    if w >= h:
        new_w = long_side
        new_h = int(round(h * (long_side / float(w))))
    else:
        new_h = long_side
        new_w = int(round(w * (long_side / float(h))))
    new_w = max(1, new_w)
    new_h = max(1, new_h)
    return im.resize((new_w, new_h), resample=Image.BILINEAR)


def _to_float_gray(im_rgb: Image.Image) -> np.ndarray:
    # float32 in [0,1]
    g = im_rgb.convert("L")
    arr = np.asarray(g, dtype=np.float32) / 255.0
    return arr


def _ssim_like(a: np.ndarray, b: np.ndarray, c1: float, c2: float) -> float:
    """
    Global SSIM-like (not windowed SSIM) on already-smoothed images.
    Returns in [-1, 1], typically [0,1] for reasonable inputs.
    """
    a = a.astype(np.float32, copy=False)
    b = b.astype(np.float32, copy=False)

    mu_a = float(a.mean())
    mu_b = float(b.mean())

    var_a = float(((a - mu_a) ** 2).mean())
    var_b = float(((b - mu_b) ** 2).mean())
    cov = float(((a - mu_a) * (b - mu_b)).mean())

    num = (2.0 * mu_a * mu_b + c1) * (2.0 * cov + c2)
    den = (mu_a * mu_a + mu_b * mu_b + c1) * (var_a + var_b + c2)
    if den <= 0:
        return 0.0
    return float(num / den)


def _lab_l1(a_rgb: Image.Image, b_rgb: Image.Image) -> float:
    a_lab = a_rgb.convert("LAB")
    b_lab = b_rgb.convert("LAB")
    diff = ImageChops.difference(a_lab, b_lab)
    stat = ImageStat.Stat(diff)
    mean_abs = float(sum(stat.mean) / 3.0)  # [0..255]
    return mean_abs / 255.0


def _edge_l1(a_rgb: Image.Image, b_rgb: Image.Image) -> float:
    a_e = a_rgb.convert("L").filter(ImageFilter.FIND_EDGES)
    b_e = b_rgb.convert("L").filter(ImageFilter.FIND_EDGES)
    diff = ImageChops.difference(a_e, b_e)
    stat = ImageStat.Stat(diff)
    mean_abs = float(stat.mean[0])  # [0..255]
    return mean_abs / 255.0


def get_scoring_reference(original_rgb: Image.Image) -> Image.Image:
    """
    Pre-process the original image to the target scoring resolution.
    Call this ONCE per worker, not per candidate.
    """
    return _resize_long_side(original_rgb, _CFG.target_long_side)


def pixel_diff_score(reference_small: Image.Image, candidate_png: bytes) -> float:
    cand = Image.open(io.BytesIO(candidate_png)).convert("RGB")

    # Resize candidate to match the pre-computed reference
    if cand.size != reference_small.size:
        # Note: We rely on reference_small being the authority on size logic
        cand = cand.resize(reference_small.size, resample=Image.BILINEAR)

    a = reference_small
    b = cand

    # Structural term (SSIM-like on smoothed grayscale)
    if _CFG.ssim_blur_radius > 0:
        a_s = a.filter(ImageFilter.GaussianBlur(radius=_CFG.ssim_blur_radius))
        b_s = b.filter(ImageFilter.GaussianBlur(radius=_CFG.ssim_blur_radius))
    else:
        a_s, b_s = a, b

    a_g = _to_float_gray(a_s)
    b_g = _to_float_gray(b_s)
    ssim = _ssim_like(a_g, b_g, c1=_CFG.ssim_c1, c2=_CFG.ssim_c2)
    struct = float(max(0.0, min(1.0, 1.0 - ssim)))

    # Perceptual-ish color term in Lab
    color = float(max(0.0, min(1.0, _lab_l1(a, b))))

    # Edge term
    edge = float(max(0.0, min(1.0, _edge_l1(a, b))))

    score = (_CFG.w_struct * struct) + (_CFG.w_color * color) + (_CFG.w_edge * edge)

    if not np.isfinite(score):
        return 1.0
    return float(max(0.0, min(1.0, score)))