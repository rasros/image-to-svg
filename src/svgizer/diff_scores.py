from __future__ import annotations

import io
from PIL import Image


def pixel_diff_score(original_rgb: Image.Image, candidate_png: bytes) -> float:
    """
    Mean absolute RGB diff normalized to [0, 1]. Lower is better.
    Simple baseline scorer.
    """
    cand = Image.open(io.BytesIO(candidate_png)).convert("RGB")
    if cand.size != original_rgb.size:
        cand = cand.resize(original_rgb.size, resample=Image.BILINEAR)

    a = original_rgb.tobytes()
    b = cand.tobytes()

    total = 0
    for x, y in zip(a, b):
        total += abs(x - y)

    mean = total / (len(a) * 255.0)
    return float(mean)

