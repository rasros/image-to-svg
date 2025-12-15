import os
import sys

from .cli import parse_args
from .search import run_search


def main():
    args = parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY environment variable is not set.", file=sys.stderr)
        raise SystemExit(1)

    run_search(
        image_path=args.image,
        output_svg_path=args.output,
        seed_svg_path=args.seed_svg,
        max_accepts=args.max_accepts,
        workers=args.workers,
        base_model_temperature=args.model_temp,
        openai_image_long_side=args.openai_image_long_side,
        max_wall_seconds=args.max_wall_seconds,
        resume=args.resume,
        write_lineage=args.write_lineage,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
