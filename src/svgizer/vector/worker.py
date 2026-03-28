import base64
import contextlib
import io
import logging
import multiprocessing as mp
import random
import signal

from PIL import Image

from svgizer.formats.models import VectorResultPayload
from svgizer.image_utils import (
    generate_diff_data_url,
    png_bytes_to_data_url,
    resize_long_side,
)
from svgizer.llm import LLMConfig, get_provider
from svgizer.score.complexity import visual_complexity
from svgizer.search import INVALID_SCORE, Result
from svgizer.search.diversity import simhash
from svgizer.utils import setup_logger


def _use_llm(has_content: bool, llm_rate: float, llm_pressure: float) -> bool:
    if llm_rate <= 0:
        return False
    return not has_content or random.random() < llm_rate * llm_pressure


def worker_loop(task_q: mp.Queue, result_q: mp.Queue, worker_params: dict):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    setup_logger(worker_params["log_level"], log_file=worker_params.get("log_file"))
    log = logging.getLogger("worker")

    try:
        plugin = worker_params["format_plugin"]
        provider_name = worker_params["llm_provider"]
        api_key = worker_params["api_key"]
        model_name = worker_params["llm_model"]
        reasoning = worker_params["reasoning"]
        llm_rate = float(worker_params["llm_rate"])
        llm_in_flight = worker_params.get("llm_in_flight")

        client = get_provider(provider_name, api_key)

        orig_img = Image.open(io.BytesIO(worker_params["original_png_bytes"])).convert(
            "RGB"
        )
        fast_eval_side = 128
        orig_img_fast = resize_long_side(orig_img, fast_eval_side)

    except Exception as e:
        log.critical(f"Worker failed initialization: {e!r}")
        return

    while True:
        try:
            task = task_q.get()
        except (OSError, EOFError, BrokenPipeError):
            break  # queue torn down during shutdown
        if task is None:
            break

        parent = task.parent_state
        has_content = bool(parent.payload.content)

        use_llm = task.force_llm or _use_llm(has_content, llm_rate, task.llm_pressure)
        llm_type = None

        try:
            if (
                task.secondary_parent_state
                and task.secondary_parent_state.payload.content
            ):
                secondary_content = task.secondary_parent_state.payload.content
                content, change_summary = plugin.crossover(
                    parent.payload.content,
                    secondary_content,
                    orig_img_fast,
                )

            elif use_llm:
                llm_type = "llm-generate"
                if llm_in_flight is not None:
                    with llm_in_flight.get_lock():
                        llm_in_flight.value += 1
                try:
                    parent_preview = (
                        parent.payload.raster_preview_data_url
                        or parent.payload.raster_data_url
                    )
                    change_summary = worker_params.get("goal")

                    if has_content:
                        sum_prompt = plugin.build_summarize_prompt(
                            worker_params["image_data_url"],
                            parent_preview,
                            custom_goal=worker_params.get("goal"),
                            previous_summary=parent.payload.change_summary,
                        )
                        sum_config = LLMConfig(model=model_name, reasoning=reasoning)
                        log.debug(f"LLM call [summarize] task={task.task_id}")
                        change_summary = client.generate(sum_prompt, sum_config)

                    diff_data_url = None
                    if parent.payload.raster_data_url:
                        _, encoded = parent.payload.raster_data_url.split(",", 1)
                        cand_bytes = base64.b64decode(encoded)
                        diff_data_url = generate_diff_data_url(
                            worker_params["original_png_bytes"],
                            cand_bytes,
                            worker_params["image_long_side"],
                        )
                    elif has_content:
                        cand_bytes = plugin.rasterize(
                            parent.payload.content,
                            out_w=worker_params["original_w"],
                            out_h=worker_params["original_h"],
                        )
                        diff_data_url = generate_diff_data_url(
                            worker_params["original_png_bytes"],
                            cand_bytes,
                            worker_params["image_long_side"],
                        )

                    gen_config = LLMConfig(model=model_name, reasoning=reasoning)
                    gen_prompt = plugin.build_generate_prompt(
                        worker_params["image_data_url"],
                        task.parent_id,
                        content_prev=parent.payload.content,
                        raster_preview_url=parent_preview if has_content else None,
                        change_summary=change_summary,
                        diff_data_url=diff_data_url,
                    )
                    log.debug(
                        f"LLM call [generate] task={task.task_id} "
                        f"parent={task.parent_id} model={model_name}"
                    )
                    raw = client.generate(gen_prompt, gen_config)
                    content = plugin.extract_from_llm(raw)
                finally:
                    if llm_in_flight is not None:
                        with llm_in_flight.get_lock():
                            llm_in_flight.value -= 1

            else:
                content, change_summary = plugin.mutate(
                    parent.payload.content,
                    orig_img_fast,
                )

            valid, err = plugin.validate(content)
            if not valid:
                raise ValueError(err)

            png = plugin.rasterize(
                content,
                out_w=worker_params["original_w"],
                out_h=worker_params["original_h"],
            )
            complexity = visual_complexity(png)
            signature = simhash(content)

            full_img = Image.open(io.BytesIO(png)).convert("RGB")
            preview_img = resize_long_side(full_img, worker_params["image_long_side"])
            preview_buf = io.BytesIO()
            preview_img.save(preview_buf, format="PNG")
            preview_data_url = png_bytes_to_data_url(preview_buf.getvalue())

            result_q.put(
                Result(
                    task_id=task.task_id,
                    parent_id=task.parent_id,
                    worker_slot=task.worker_slot,
                    valid=True,
                    score=None,
                    payload=VectorResultPayload(
                        content=content,
                        raster_png=png,
                        change_summary=change_summary,
                        raster_preview_data_url=preview_data_url,
                    ),
                    secondary_parent_id=task.secondary_parent_id,
                    complexity=complexity,
                    signature=signature,
                    llm_type=llm_type,
                )
            )

        except Exception as e:
            log.error(f"Task {task.task_id} failed: {e!r}")
            with contextlib.suppress(OSError, EOFError, BrokenPipeError):
                result_q.put(
                    Result(
                        task_id=task.task_id,
                        parent_id=task.parent_id,
                        worker_slot=task.worker_slot,
                        valid=False,
                        score=INVALID_SCORE,
                        payload=VectorResultPayload(None, None, None),
                        invalid_msg=repr(e),
                        secondary_parent_id=task.secondary_parent_id,
                        signature=None,
                        llm_type=llm_type,
                    )
                )
