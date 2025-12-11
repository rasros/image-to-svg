#!/usr/bin/env python3
import argparse
import base64
import difflib
import os
import sys
import xml.etree.ElementTree as ET

from openai import OpenAI

try:
    import cairosvg
except ImportError:
    print("cairosvg is required. Install with: pip install cairosvg", file=sys.stderr)
    sys.exit(1)

MODEL_NAME = "gpt-5.1"
DEFAULT_MAX_ITER = 8
DEFAULT_BASE_TEMP = 0.2
TEMP_STEP = 0.3
MAX_TEMP = 1.6
STALENESS_THRESHOLD = 0.995
STALENESS_HITS_BEFORE_TEMP_INCREASE = 1


def guess_mime_type(path: str) -> str:
    ext = os.path.splitext(path.lower())[1]
    if ext in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if ext == ".webp":
        return "image/webp"
    if ext == ".gif":
        return "image/gif"
    return "image/png"


def encode_image_to_data_url(path: str) -> str:
    mime = guess_mime_type(path)
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def rasterize_svg_to_data_url(svg_text: str) -> str:
    png_bytes = cairosvg.svg2png(bytestring=svg_text.encode("utf-8"))
    b64 = base64.b64encode(png_bytes).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def extract_svg_fragment(raw: str) -> str:
    lower = raw.lower()
    start_idx = lower.find("<svg")
    end_idx = lower.rfind("</svg>")
    if start_idx != -1 and end_idx != -1:
        end_idx += len("</svg>")
        return raw[start_idx:end_idx].strip()
    return raw.strip()


def is_valid_svg(svg_text: str):
    try:
        root = ET.fromstring(svg_text)
    except ET.ParseError as e:
        return False, f"XML parse error: {e}"
    if root.tag.lower().endswith("svg"):
        return True, None
    return False, f"Root tag is not <svg>: got <{root.tag}>"


def is_stale(prev_svg: str | None, new_svg: str) -> bool:
    if prev_svg is None:
        return False
    ratio = difflib.SequenceMatcher(None, prev_svg, new_svg).ratio()
    return ratio >= STALENESS_THRESHOLD


def call_openai_for_svg(
    client: OpenAI,
    original_data_url: str,
    iter_index: int,
    temperature: float,
    svg_prev: str | None = None,
    svg_prev_invalid_msg: str | None = None,
    rasterized_svg_data_url: str | None = None,
) -> str:
    lines = [
        "You convert a raster input image into clean, valid SVG markup.",
        "Output ONLY a complete <svg>...</svg> document with no explanations or backticks.",
        "The SVG must be faithful as close to possible to the input image.",
        "Use shapes, paths, groups, and filters; avoid embedded rasters or <image> tags.",
        "Ensure the SVG is valid XML and has a single <svg> root.",
        f"Iteration #{iter_index}.",
    ]

    if svg_prev is None:
        lines.append("This is the first attempt. Produce your best initial SVG approximation.")
    else:
        lines.append("Refine the previous SVG. Prefer small, meaningful improvements over total rewrites.")

    if svg_prev_invalid_msg:
        lines.append(
            f"The previous SVG was INVALID. Validation error:\n{svg_prev_invalid_msg}\n"
            "Return a corrected, valid SVG."
        )

    if rasterized_svg_data_url:
        lines.append(
            "You are given the original raster image and a rasterized rendering of your latest SVG. "
            "Use their differences to refine the SVG."
        )

    if svg_prev:
        lines.append("Here is the previous SVG markup to reuse and improve:\n" + svg_prev)

    instruction_text = "\n".join(lines)

    content = [
        {"type": "input_text", "text": instruction_text},
        {"type": "input_image", "image_url": original_data_url},
    ]

    if rasterized_svg_data_url:
        content.append({"type": "input_image", "image_url": rasterized_svg_data_url})

    response = client.responses.create(
        model=MODEL_NAME,
        input=[{"role": "user", "content": content}],
        temperature=temperature,
        text={"format": {"type": "text"}},
    )

    return response.output_text


def run(
    image_path: str,
    output_svg_path: str,
    max_iter: int,
    base_temperature: float,
):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    if not os.path.isfile(image_path):
        print(f"Input image '{image_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    original_data_url = encode_image_to_data_url(image_path)

    base_name, ext = os.path.splitext(output_svg_path)
    if not ext:
        ext = ".svg"

    current_svg = None
    previous_svg = None
    current_raster_data_url = None
    previous_invalid_msg = None
    temperature = base_temperature
    stale_hits = 0

    for i in range(1, max_iter + 1):
        print(f"\n=== Iteration {i}/{max_iter} | temperature={temperature:.2f} ===")

        raw_text = call_openai_for_svg(
            client=client,
            original_data_url=original_data_url,
            iter_index=i,
            temperature=temperature,
            svg_prev=previous_svg,
            svg_prev_invalid_msg=previous_invalid_msg,
            rasterized_svg_data_url=current_raster_data_url,
        )

        svg_candidate = extract_svg_fragment(raw_text)
        print(f"Received SVG text length: {len(svg_candidate)}")

        iter_path = f"{base_name}_{i:02d}{ext}"
        try:
            with open(iter_path, "w", encoding="utf-8") as f:
                f.write(svg_candidate)
            print(f"Wrote iteration SVG to: {iter_path}")
        except Exception as e:
            print(f"Failed to write iteration SVG '{iter_path}': {e}", file=sys.stderr)

        valid, err_msg = is_valid_svg(svg_candidate)
        if valid:
            print("SVG validated successfully.")
            previous_invalid_msg = None
            current_svg = svg_candidate
            try:
                current_raster_data_url = rasterize_svg_to_data_url(current_svg)
                print("Rasterized SVG for next iteration.")
            except Exception as e:
                current_raster_data_url = None
                print(f"Rasterization of SVG failed: {e}", file=sys.stderr)
        else:
            print(f"SVG is invalid: {err_msg}", file=sys.stderr)
            previous_invalid_msg = err_msg

        if is_stale(previous_svg, svg_candidate):
            stale_hits += 1
            print(f"Staleness detected (hit {stale_hits}).")
            if stale_hits >= STALENESS_HITS_BEFORE_TEMP_INCREASE and temperature < MAX_TEMP:
                new_temp = min(MAX_TEMP, temperature + TEMP_STEP)
                if new_temp > temperature:
                    print(f"Increasing temperature from {temperature:.2f} to {new_temp:.2f}.")
                    temperature = new_temp
                stale_hits = 0
        else:
            stale_hits = 0

        previous_svg = svg_candidate

    final_svg = current_svg if current_svg is not None else previous_svg

    if final_svg is None:
        print("No SVG produced across all iterations.", file=sys.stderr)
        sys.exit(1)

    try:
        with open(output_svg_path, "w", encoding="utf-8") as f:
            f.write(final_svg)
        print(f"\nFinal SVG written to: {output_svg_path}")
    except Exception as e:
        print(f"Failed to write final SVG '{output_svg_path}': {e}", file=sys.stderr)
        sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Iteratively approximate an image with SVG using OpenAI."
    )
    parser.add_argument("image", help="Path to input raster image (PNG/JPEG/WEBP/GIF).")
    parser.add_argument(
        "--output",
        "-o",
        default="output.svg",
        help="Final SVG path (default: output.svg). Iteration files: output_01.svg, output_02.svg, etc.",
    )
    parser.add_argument(
        "--max-iter",
        "-n",
        type=int,
        default=DEFAULT_MAX_ITER,
        help=f"Maximum number of refinement iterations (default: {DEFAULT_MAX_ITER}).",
    )
    parser.add_argument(
        "--base-temp",
        type=float,
        default=DEFAULT_BASE_TEMP,
        help=f"Base sampling temperature (default: {DEFAULT_BASE_TEMP}).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        image_path=args.image,
        output_svg_path=args.output,
        max_iter=args.max_iter,
        base_temperature=args.base_temp,
    )

