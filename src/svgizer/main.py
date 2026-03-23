import os
import sys

from svgizer.search.base import StrategyType
from svgizer.svg.runner import run_svg_search
from svgizer.svg.storage import FileStorageAdapter

from .cli import parse_args


def main():
    args = parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print(
            "CRITICAL: OPENAI_API_KEY environment variable is not set.", file=sys.stderr
        )
        raise SystemExit(1)

    storage = FileStorageAdapter(
        output_svg_path=args.output,
        resume=args.resume,
        openai_image_long_side=args.openai_image_long_side,
        base_temp=args.model_temp,
    )

    try:
        run_svg_search(
            image_path=args.image,
            storage=storage,
            seed_svg_path=args.seed_svg,
            max_accepts=args.max_accepts,
            workers=args.workers,
            base_model_temperature=args.model_temp,
            openai_image_long_side=args.openai_image_long_side,
            max_wall_seconds=args.max_wall_seconds,
            log_level=args.log_level,
            scorer_type=args.scorer,
            strategy_type=StrategyType(args.strategy),
            goal=args.goal,
            write_lineage=args.write_lineage,
        )
    except KeyboardInterrupt:
        print("\nSearch interrupted by user. Exiting safely...", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"FATAL: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
