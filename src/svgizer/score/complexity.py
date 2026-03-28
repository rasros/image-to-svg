import io

from PIL import Image


def visual_complexity(png_bytes: bytes) -> float:
    """Visual complexity measured as JPEG compressed size.

    JPEG encodes spatial redundancy the same way the human visual system weighs
    detail: a flat-coloured region compresses to almost nothing; a region with
    fine detail or many colour transitions requires many more bytes.  This gives
    a perceptual complexity score that is immune to SVG structural tricks (e.g.
    a single large <rect> matching the dominant colour scores near zero even if
    the SVG file itself is small).
    """
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return float(len(buf.getvalue()))
