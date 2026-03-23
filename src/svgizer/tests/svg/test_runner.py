import io
import logging

from PIL import Image

from svgizer.diff import DiffScorer, ScorerType, get_scorer
from svgizer.image_utils import (
    downscale_png_bytes,
    png_bytes_to_data_url,
)
from svgizer.search import (
    GeneticPoolStrategy,
    MultiprocessSearchEngine,
    StorageAdapter,
    StrategyType,
)
from svgizer.svg.adapter import SvgStrategyAdapter, is_svg_stale
from svgizer.svg.worker import worker_loop
from svgizer.utils import setup_logger

log = logging.getLogger("main")


def run_svg_search(
    image_path: str,
    storage: StorageAdapter,
    seed_svg_path: str | None,
    max_accepts: int,
    workers: int,
    base_model_temperature: float,
    openai_image_long_side: int,
    max_wall_seconds: float | None,
    log_level: str,
    scorer_type: ScorerType,
    strategy_type: StrategyType,
    goal: str | None,
    # --- Actual Code DI Hooks ---
    engine_cls: type[MultiprocessSearchEngine] = MultiprocessSearchEngine,
    scorer_override: DiffScorer | None = None,
) -> None:
    setup_logger(log_level)

    original_img = Image.open(image_path).convert("RGB")
    original_w, original_h = original_img.size

    buf = io.BytesIO()
    original_img.save(buf, format="PNG")
    original_png_bytes = buf.getvalue()

    # Use override if provided (useful for tests)
    scorer = scorer_override or get_scorer(scorer_type)
    scorer.prepare_reference(original_img)

    storage.initialize()
    initial_nodes = storage.load_resume_nodes()

    # ... [Seed handling logic remains same] ...

    base_strategy = GeneticPoolStrategy(top_k=3, is_stale_fn=is_svg_stale)
    strategy = SvgStrategyAdapter(base_strategy, openai_image_long_side, False)

    # Use the injected engine class
    engine = engine_cls(workers=workers, strategy=strategy, storage=storage)

    model_png = downscale_png_bytes(original_png_bytes, openai_image_long_side)
    worker_params = {
        "openai_original_data_url": png_bytes_to_data_url(model_png),
        "original_png_bytes": original_png_bytes,
        "original_w": original_w,
        "original_h": original_h,
        "log_level": log_level,
        "scorer_type": scorer_type,
        "goal": goal,
    }

    engine.start_workers(worker_loop, worker_params)
    best_node = engine.run(initial_nodes, max_accepts, max_wall_seconds)

    if best_node and best_node.state.payload.svg:
        storage.save_final_svg(best_node.state.payload.svg)
