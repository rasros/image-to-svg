import io
import logging
import random
import re

from PIL import Image

from svgizer.search.models import INVALID_SCORE

log = logging.getLogger(__name__)

# Regex to match Typst numeric attributes with units like 12pt, 1.5em, 50%
_NUM_RE = re.compile(r"(\b|-)(\d+(?:\.\d+)?)(pt|em|%|mm|cm|in)\b")


def _random_numeric_tweak(typst_code: str) -> str:
    """Find a random numeric value with a unit and scale it."""
    matches = list(_NUM_RE.finditer(typst_code))
    if not matches:
        return typst_code

    m = random.choice(matches)
    prefix = m.group(1)
    val = float(m.group(2))
    unit = m.group(3)

    factor = random.uniform(0.8, 1.2)
    new_val = max(0.1, val * factor)  # Prevent zero or negative sizes
    formatted = f"{prefix}{new_val:.2f}{unit}".replace(".00", "")

    return typst_code[: m.start()] + formatted + typst_code[m.end() :]


def _random_block_swap(typst_code: str) -> str:
    """Randomly swap two lines or blocks to shake up the rendering order."""
    lines = [line for line in typst_code.split("\n") if line.strip()]
    if len(lines) < 3:
        return typst_code

    idx1, idx2 = random.sample(range(1, len(lines)), 2)  # keep index 0 (page setup)
    lines[idx1], lines[idx2] = lines[idx2], lines[idx1]
    return "\n".join(lines)


def _apply_one_mutation(typst_code: str) -> str:
    ops = [_random_numeric_tweak, _random_numeric_tweak, _random_block_swap]
    return random.choice(ops)(typst_code)


def _rasterize_typst(typst_code: str) -> bytes | None:
    try:
        import typst

        # Compile to PNG at standard DPI
        png_bytes = typst.compile(typst_code, format="png", ppi=144)
        if isinstance(png_bytes, list) and len(png_bytes) > 0:
            return png_bytes[0]  # Take first page if multi-page array
        return png_bytes
    except Exception:
        return None


def _fast_lab_l1(png_a: bytes, png_b: bytes, size: int = 64) -> float:
    from svgizer.image_utils import resize_long_side
    from svgizer.score.utils import lab_l1

    try:
        img_a = resize_long_side(Image.open(io.BytesIO(png_a)).convert("RGB"), size)
        img_b = resize_long_side(Image.open(io.BytesIO(png_b)).convert("RGB"), size)
        return lab_l1(img_a, img_b)
    except Exception:
        return 1.0


def mutate_with_micro_search(
    parent_code: str,
    orig_img_fast: Image.Image,
    num_trials: int = 15,
) -> tuple[str, str]:
    orig_buf = io.BytesIO()
    orig_img_fast.save(orig_buf, format="PNG")
    orig_png = orig_buf.getvalue()

    best_code = parent_code
    best_score = INVALID_SCORE

    for _ in range(num_trials):
        candidate = _apply_one_mutation(parent_code)
        png = _rasterize_typst(candidate)
        if png is None:
            continue
        score = _fast_lab_l1(orig_png, png)
        if score < best_score:
            best_score = score
            best_code = candidate

    return best_code, "local typst mutation"


def crossover_with_micro_search(
    code_a: str,
    code_b: str,
    orig_img_fast: Image.Image,
    num_trials: int = 15,
) -> tuple[str, str]:
    orig_buf = io.BytesIO()
    orig_img_fast.save(orig_buf, format="PNG")
    orig_png = orig_buf.getvalue()

    # Simple crossover: split by double newline, inject a random block from B into A
    blocks_a = code_a.split("\n\n")
    blocks_b = [b for b in code_b.split("\n\n") if b.strip()]

    if not blocks_b or len(blocks_a) < 2:
        return mutate_with_micro_search(code_a, orig_img_fast, num_trials)

    best_code = code_a
    best_score = INVALID_SCORE

    for _ in range(num_trials):
        idx_a = random.randint(1, len(blocks_a))
        block_b = random.choice(blocks_b)

        candidate_blocks = [*blocks_a[:idx_a], block_b, *blocks_a[idx_a:]]
        candidate = "\n\n".join(candidate_blocks)

        png = _rasterize_typst(candidate)
        if png is None:
            continue
        score = _fast_lab_l1(orig_png, png)
        if score < best_score:
            best_score = score
            best_code = candidate

    return best_code, "typst block crossover"
