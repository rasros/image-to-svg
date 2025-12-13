import argparse

DEFAULT_MAX_ITER = 8
DEFAULT_BASE_TEMP = 0.2

DEFAULT_NUM_BEAMS = 4
DEFAULT_CANDIDATES_PER_NODE = 4
DEFAULT_WORKERS = 4


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fully pipelined beam/best-first SVG approximation of an image using OpenAI."
    )
    parser.add_argument("image", help="Path to input raster image (PNG/JPEG/WEBP/GIF).")
    parser.add_argument(
        "--output",
        "-o",
        default="output.svg",
        help="Final SVG path (default: output.svg).",
    )
    parser.add_argument(
        "--max-iter",
        "-n",
        type=int,
        default=DEFAULT_MAX_ITER,
        help=f"Expansion budget factor (default: {DEFAULT_MAX_ITER}). Total accepts ~= max_iter * num_beams.",
    )
    parser.add_argument(
        "--base-temp",
        type=float,
        default=DEFAULT_BASE_TEMP,
        help=f"Base sampling temperature (default: {DEFAULT_BASE_TEMP}).",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=DEFAULT_NUM_BEAMS,
        help=f"Beam width (default: {DEFAULT_NUM_BEAMS}).",
    )
    parser.add_argument(
        "--candidates-per-node",
        type=int,
        default=DEFAULT_CANDIDATES_PER_NODE,
        help=f"Number of candidates sampled per accepted node (default: {DEFAULT_CANDIDATES_PER_NODE}).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of multiprocessing workers (default: {DEFAULT_WORKERS}).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR). Default: INFO.",
    )
    parser.add_argument(
        "--write-top-k-each",
        type=int,
        default=10,
        help="Write TOP snapshot every N accepted nodes (0 disables). Default: 10.",
    )
    return parser.parse_args()

