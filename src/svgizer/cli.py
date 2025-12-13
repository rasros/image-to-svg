import argparse

DEFAULT_MAX_ACCEPTS = 32
DEFAULT_WORKERS = 4
DEFAULT_MODEL_TEMP = 0.6
DEFAULT_MAX_TOTAL_TASKS = 10_000
DEFAULT_MAX_WALL_SECONDS = 0  # 0 disables
DEFAULT_RESUME = True
DEFAULT_TOP_K = 3
DEFAULT_WRITE_LINEAGE = True
DEFAULT_ELITE_PROB_START = 0.70
DEFAULT_ELITE_PROB_END = 0.10
DEFAULT_OPENAI_IMAGE_LONG_SIDE = 512

def parse_args():
    parser = argparse.ArgumentParser(
        description="Pipelined SVG approximation of an image using OpenAI (pool-based refinement)."
    )
    parser.add_argument("image", help="Path to input raster image (PNG/JPEG/WEBP/GIF).")
    parser.add_argument("--output", "-o", default="output.svg", help="Final SVG path.")
    parser.add_argument("--seed-svg", default=None, help="Path to an SVG file to seed the pool.")

    parser.add_argument("--max-accepts", type=int, default=DEFAULT_MAX_ACCEPTS)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--model-temp", type=float, default=DEFAULT_MODEL_TEMP)
    parser.add_argument("--openai-image-long-side", type=int, default=DEFAULT_OPENAI_IMAGE_LONG_SIDE)
    parser.add_argument("--max-total-tasks", type=int, default=DEFAULT_MAX_TOTAL_TASKS)
    parser.add_argument("--max-wall-seconds", type=float, default=DEFAULT_MAX_WALL_SECONDS)

    parser.add_argument("--resume", dest="resume", action=argparse.BooleanOptionalAction, default=DEFAULT_RESUME)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--elite-prob-start", type=float, default=DEFAULT_ELITE_PROB_START)
    parser.add_argument("--elite-prob-end", type=float, default=DEFAULT_ELITE_PROB_END)
    parser.add_argument("--write-lineage", dest="write_lineage", action=argparse.BooleanOptionalAction, default=DEFAULT_WRITE_LINEAGE)
    parser.add_argument("--log-level", default="INFO")

    args = parser.parse_args()

    if args.max_wall_seconds is not None and args.max_wall_seconds <= 0:
        args.max_wall_seconds = None

    if args.max_accepts <= 0 or args.workers <= 0 or args.max_total_tasks <= 0 or args.top_k <= 0:
        raise SystemExit("Configuration values must be > 0")
    if args.model_temp < 0 or args.openai_image_long_side < 0:
        raise SystemExit("Configuration values must be >= 0")
    if not (0.0 <= args.elite_prob_start <= 1.0) or not (0.0 <= args.elite_prob_end <= 1.0):
        raise SystemExit("Probabilities must be in [0, 1]")

    return args