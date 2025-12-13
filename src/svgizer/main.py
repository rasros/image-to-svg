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
        max_iter=args.max_iter,
        base_temperature=args.base_temp,
        num_beams=args.num_beams,
        candidates_per_node=args.candidates_per_node,
        workers=args.workers,
        log_level=args.log_level,
        write_top_k_each=args.write_top_k_each,
    )


if __name__ == "__main__":
    main()

