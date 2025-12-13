from __future__ import annotations

import base64
import difflib
import io
import logging
import os
import queue
import re
import time
from collections import deque
from typing import Optional, List, Any, Tuple

import multiprocessing as mp

import cairosvg
from openai import OpenAI
from PIL import Image

from svgizer.models import BeamState, SearchNode, Task, Result, INVALID_SCORE
from svgizer.diff_scores import pixel_diff_score
from svgizer.openai_iface import (
    summarize_changes,
    call_openai_for_svg,
    extract_svg_fragment,
    is_valid_svg,
)

# Temperature / staleness knobs
TEMP_STEP = 0.3
MAX_TEMP = 1.6
STALENESS_THRESHOLD = 0.995
STALENESS_HITS_BEFORE_TEMP_INCREASE = 1

# Resume file pattern written by this tool:
#   {base_name}_node{node_id:05d}_score{score:.6f}.svg
NODE_FILE_RE = re.compile(r"_node(\d+)_score([0-9.]+)\.svg$")


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


def png_bytes_to_data_url(png_bytes: bytes) -> str:
    b64 = base64.b64encode(png_bytes).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def rasterize_svg_to_png_bytes(svg_text: str) -> bytes:
    return cairosvg.svg2png(bytestring=svg_text.encode("utf-8"))


def is_stale(prev_svg: Optional[str], new_svg: str) -> bool:
    if prev_svg is None:
        return False
    ratio = difflib.SequenceMatcher(None, prev_svg, new_svg).ratio()
    return ratio >= STALENESS_THRESHOLD


def setup_logger(level: str) -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(processName)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _fatal(msg: str) -> str:
    return "FATAL: " + msg


def worker_loop(
    task_q: mp.Queue,
    result_q: mp.Queue,
    original_data_url: str,
    original_png_bytes: bytes,
    log_level: str,
):
    setup_logger(log_level)
    log = logging.getLogger("worker")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # Fatal: cannot proceed.
        log.error("OPENAI_API_KEY environment variable is not set.")
        return

    client = OpenAI(api_key=api_key)
    original_rgb = Image.open(io.BytesIO(original_png_bytes)).convert("RGB")

    while True:
        task: Any = task_q.get()
        if task is None:
            log.info("Worker received shutdown signal.")
            return

        assert isinstance(task, Task)
        parent = task.parent_state

        temperature = min(MAX_TEMP, max(0.0, parent.temperature + (task.candidate_index * 0.05)))
        log.info(
            f"Start task={task.task_id} parent={task.parent_id} cand={task.candidate_index} temp={temperature:.2f}"
        )

        change_summary = None
        if parent.svg and parent.raster_data_url:
            try:
                change_summary = summarize_changes(
                    client=client,
                    original_data_url=original_data_url,
                    iter_index=task.parent_id,
                    rasterized_svg_data_url=parent.raster_data_url,
                )
                log.info(f"Summary task={task.task_id}:\n{change_summary}")
            except Exception as e:
                # Not fatal; continue without summary.
                log.warning(f"Summary failed task={task.task_id}: {e}")

        # OpenAI call
        try:
            raw = call_openai_for_svg(
                client=client,
                original_data_url=original_data_url,
                iter_index=task.parent_id,
                temperature=temperature,
                svg_prev=parent.svg,
                svg_prev_invalid_msg=parent.invalid_msg,
                rasterized_svg_data_url=parent.raster_data_url,
                change_summary=change_summary,
                diversity_hint=f"parent={task.parent_id} cand={task.candidate_index}",
            )
            svg = extract_svg_fragment(raw)
            log.info(f"OpenAI returned task={task.task_id} svg_len={len(svg)}")
        except Exception as e:
            # Fatal by default: user requested cancel on error.
            result_q.put(
                Result(
                    task_id=task.task_id,
                    parent_id=task.parent_id,
                    candidate_index=task.candidate_index,
                    svg=None,
                    valid=False,
                    invalid_msg=_fatal(f"OpenAI call failed: {e}"),
                    raster_png=None,
                    score=INVALID_SCORE,
                    used_temperature=temperature,
                    change_summary=change_summary,
                )
            )
            continue

        # Validate SVG (not fatal; invalid candidates are expected sometimes)
        valid, err = is_valid_svg(svg)
        if not valid:
            log.warning(f"Invalid SVG task={task.task_id}: {err}")
            result_q.put(
                Result(
                    task_id=task.task_id,
                    parent_id=task.parent_id,
                    candidate_index=task.candidate_index,
                    svg=svg,
                    valid=False,
                    invalid_msg=err,
                    raster_png=None,
                    score=INVALID_SCORE,
                    used_temperature=temperature,
                    change_summary=change_summary,
                )
            )
            continue

        # Rasterize and score
        try:
            png = rasterize_svg_to_png_bytes(svg)
            score = pixel_diff_score(original_rgb, png)
            log.info(f"Scored task={task.task_id} score={score:.6f}")
        except Exception as e:
            # Fatal by default: user requested cancel on error.
            result_q.put(
                Result(
                    task_id=task.task_id,
                    parent_id=task.parent_id,
                    candidate_index=task.candidate_index,
                    svg=svg,
                    valid=False,
                    invalid_msg=_fatal(f"Rasterize/score failed: {e}"),
                    raster_png=None,
                    score=INVALID_SCORE,
                    used_temperature=temperature,
                    change_summary=change_summary,
                )
            )
            continue

        result_q.put(
            Result(
                task_id=task.task_id,
                parent_id=task.parent_id,
                candidate_index=task.candidate_index,
                svg=svg,
                valid=True,
                invalid_msg=None,
                raster_png=png,
                score=score,
                used_temperature=temperature,
                change_summary=change_summary,
            )
        )


def _load_resume_nodes(
    base_name: str,
    ext: str,
    log: logging.Logger,
    base_temperature: float,
) -> Tuple[List[SearchNode], Optional[SearchNode], int]:
    """
    Resume by scanning previously written node SVGs:
      {base_name}_node{ID}_score{SCORE}{ext}

    Returns: (accepted_nodes_sorted_by_id, best_seen, max_node_id)
    """
    directory = os.path.dirname(base_name) or "."
    prefix = os.path.basename(base_name) + "_node"

    accepted: List[SearchNode] = []
    best_seen: Optional[SearchNode] = None
    max_id = 0

    for fn in os.listdir(directory):
        if not fn.startswith(prefix) or not fn.endswith(ext):
            continue
        m = NODE_FILE_RE.search(fn)
        if not m:
            continue
        node_id = int(m.group(1))
        score = float(m.group(2))
        path = os.path.join(directory, fn)

        try:
            with open(path, "r", encoding="utf-8") as f:
                svg = f.read()
            # recompute raster_data_url for summaries
            png = rasterize_svg_to_png_bytes(svg)
            raster_data_url = png_bytes_to_data_url(png)
        except Exception as e:
            log.warning(f"Resume: failed to load {fn}: {e}")
            continue

        state = BeamState(
            svg=svg,
            raster_data_url=raster_data_url,
            score=score,
            temperature=base_temperature,
            stale_hits=0,
            invalid_msg=None,
        )
        node = SearchNode(score=score, id=node_id, state=state)
        accepted.append(node)

        if best_seen is None or score < best_seen.score:
            best_seen = node
        if node_id > max_id:
            max_id = node_id

    accepted.sort(key=lambda n: n.id)
    if accepted:
        log.info(f"Resume: loaded {len(accepted)} prior nodes (max_id={max_id}).")
        if best_seen:
            log.info(f"Resume: best prior node={best_seen.id} score={best_seen.score:.6f}")
    return accepted, best_seen, max_id


def run_search(
    image_path: str,
    output_svg_path: str,
    max_iter: int,
    base_temperature: float,
    num_beams: int,
    candidates_per_node: int,
    workers: int,
    log_level: str,
    write_top_k_each: int,
    *,
    # Overall limits
    max_total_tasks: int = 10_000,          # max OpenAI calls enqueued (hard stop)
    max_wall_seconds: Optional[float] = None,  # optional time limit
    # Resume behavior
    resume: bool = True,
) -> None:
    setup_logger(log_level)
    log = logging.getLogger("main")

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY environment variable is not set.")
    if not os.path.isfile(image_path):
        raise SystemExit(f"Input image '{image_path}' does not exist.")

    original_data_url = encode_image_to_data_url(image_path)

    original_img = Image.open(image_path).convert("RGB")
    buf = io.BytesIO()
    original_img.save(buf, format="PNG")
    original_png_bytes = buf.getvalue()

    base_name, ext = os.path.splitext(output_svg_path)
    if not ext:
        ext = ".svg"

    start_time = time.monotonic()

    def time_up() -> bool:
        return max_wall_seconds is not None and (time.monotonic() - start_time) >= max_wall_seconds

    # Bigger queue headroom; still bounded so we can enforce max_total_tasks.
    ctx = mp.get_context("spawn")
    task_q: mp.Queue = ctx.Queue(maxsize=max(512, workers * 256))
    result_q: mp.Queue = ctx.Queue()

    procs: List[mp.Process] = []
    for _ in range(max(1, workers)):
        p = ctx.Process(
            target=worker_loop,
            args=(task_q, result_q, original_data_url, original_png_bytes, log_level),
            daemon=True,
        )
        p.start()
        procs.append(p)

    def shutdown_all(reason: str, terminate: bool = True) -> None:
        log.error(f"Canceling run: {reason}")
        # Try graceful stop
        for _ in procs:
            try:
                task_q.put_nowait(None)
            except Exception:
                pass
        # Then force
        if terminate:
            for p in procs:
                try:
                    p.terminate()
                except Exception:
                    pass
        for p in procs:
            try:
                p.join(timeout=1.0)
            except Exception:
                pass

    # In-memory beam
    best_heap: List[SearchNode] = []
    accepted_nodes: List[SearchNode] = []
    best_seen: Optional[SearchNode] = None

    # IDs
    next_task_id = 1
    next_node_id = 0

    # target accept count (soft); overall hard limits control stop
    target_accepts = max_iter * num_beams

    # Root
    root_state = BeamState(
        svg=None,
        raster_data_url=None,
        score=INVALID_SCORE,
        temperature=base_temperature,
        stale_hits=0,
        invalid_msg=None,
    )
    root_node = SearchNode(score=INVALID_SCORE, id=0, state=root_state)
    best_heap.append(root_node)

    # Resume from prior node files (if present)
    if resume:
        prior_nodes, prior_best, max_id = _load_resume_nodes(base_name, ext, log, base_temperature)
        if prior_nodes:
            accepted_nodes.extend(prior_nodes)
            best_seen = prior_best
            next_node_id = max_id
            # rebuild beam
            best_heap = [root_node] + sorted(prior_nodes, key=lambda n: n.score)[:num_beams]
            best_heap.sort(key=lambda n: n.score)

    # Pending expansions and tasks
    expand_q = deque()       # items: (node_id, BeamState, next_cand)
    pending_tasks = deque()  # items: Task

    def schedule_expand(node_id: int, state: BeamState) -> None:
        expand_q.append((node_id, state, 0))

    def try_enqueue_task(task: Task) -> bool:
        nonlocal next_task_id
        if task.task_id > max_total_tasks:
            return False
        try:
            task_q.put_nowait(task)
            log.info(
                f"Enqueue task={task.task_id} parent={task.parent_id} cand={task.candidate_index} "
                f"temp≈{min(MAX_TEMP, task.parent_state.temperature + (task.candidate_index*0.05)):.2f}"
            )
            return True
        except queue.Full:
            return False

    def pump_enqueues(budget: int = 4096) -> None:
        nonlocal next_task_id
        if time_up():
            return

        # 1) flush pending tasks first
        for _ in range(budget):
            if not pending_tasks:
                break
            t = pending_tasks[0]
            if try_enqueue_task(t):
                pending_tasks.popleft()
                next_task_id += 1
            else:
                return

        # 2) generate tasks from expansions, one at a time
        for _ in range(budget):
            if not expand_q:
                break
            if next_task_id > max_total_tasks:
                return

            parent_id, parent_state, next_cand = expand_q.popleft()
            if next_cand >= candidates_per_node:
                continue

            task = Task(
                task_id=next_task_id,
                parent_id=parent_id,
                parent_state=parent_state,
                candidate_index=next_cand,
            )

            if try_enqueue_task(task):
                next_task_id += 1
                # requeue expansion with next candidate
                if next_cand + 1 < candidates_per_node:
                    expand_q.appendleft((parent_id, parent_state, next_cand + 1))
            else:
                # queue full: hold the task and resume this expansion later
                pending_tasks.append(task)
                expand_q.appendleft((parent_id, parent_state, next_cand))
                return

    def handle_result(res: Result) -> bool:
        nonlocal next_node_id, best_seen

        # Cancel entire run on fatal errors (exceptions inside worker)
        if res.invalid_msg and res.invalid_msg.startswith("FATAL:"):
            shutdown_all(res.invalid_msg, terminate=True)
            raise SystemExit(res.invalid_msg)

        # Non-fatal rejects (invalid SVG, etc.)
        if (not res.valid) or (res.svg is None) or (res.raster_png is None) or (res.score >= INVALID_SCORE):
            log.warning(
                f"Reject task={res.task_id} parent={res.parent_id} cand={res.candidate_index} reason={res.invalid_msg}"
            )
            return False

        # Find parent state for lineage updates (scan recent accepts)
        parent_state = None
        for n in reversed(accepted_nodes[-300:]):
            if n.id == res.parent_id:
                parent_state = n.state
                break
        if parent_state is None and res.parent_id == 0:
            parent_state = root_state

        stale_hits = parent_state.stale_hits if parent_state else 0
        next_temp = parent_state.temperature if parent_state else base_temperature

        if parent_state and is_stale(parent_state.svg, res.svg):
            stale_hits += 1
            if stale_hits >= STALENESS_HITS_BEFORE_TEMP_INCREASE and next_temp < MAX_TEMP:
                next_temp = min(MAX_TEMP, next_temp + TEMP_STEP)
                stale_hits = 0
                log.info(f"Staleness: increasing temp for lineage parent={res.parent_id} -> {next_temp:.2f}")
        else:
            stale_hits = 0

        next_node_id += 1
        state = BeamState(
            svg=res.svg,
            raster_data_url=png_bytes_to_data_url(res.raster_png),
            score=res.score,
            temperature=next_temp,
            stale_hits=stale_hits,
            invalid_msg=None,
        )
        node = SearchNode(score=res.score, id=next_node_id, state=state)
        accepted_nodes.append(node)

        # Maintain beam set
        best_heap.append(node)
        best_heap.sort(key=lambda x: x.score)
        if len(best_heap) > num_beams:
            best_heap[:] = best_heap[:num_beams]

        if best_seen is None or node.score < best_seen.score:
            best_seen = node
            log.info(f"NEW BEST node={node.id} score={node.score:.6f} (from task={res.task_id})")

        # Write candidate (supports resume)
        iter_path = f"{base_name}_node{node.id:05d}_score{node.score:.6f}{ext}"
        try:
            with open(iter_path, "w", encoding="utf-8") as f:
                f.write(node.state.svg or "")
            log.info(f"Wrote {iter_path}")
        except Exception as e:
            shutdown_all(f"Failed to write {iter_path}: {e}", terminate=True)
            raise SystemExit(1)

        # Write TOP snapshot
        if write_top_k_each > 0 and (node.id % write_top_k_each == 0):
            snap = sorted(best_heap, key=lambda x: x.score)
            for rank, bn in enumerate(snap, start=1):
                pth = f"{base_name}_TOP_rank{rank:02d}_node{bn.id:05d}_score{bn.score:.6f}{ext}"
                try:
                    with open(pth, "w", encoding="utf-8") as f:
                        f.write(bn.state.svg or "")
                except Exception:
                    pass
            log.info(f"Wrote TOP snapshot (every {write_top_k_each} accepts)")

        # Schedule expansion (never enqueue directly here)
        schedule_expand(node.id, node.state)
        return True

    # Seed: expand root, plus (if resuming) also re-expand current best beams to continue search
    schedule_expand(0, root_state)
    if resume and best_heap:
        # Re-expand current beam states so the run continues from what you already have
        for n in best_heap:
            if n.id != 0 and n.state.svg:
                schedule_expand(n.id, n.state)

    pump_enqueues(budget=50_000)

    accepted_valid = 0
    while True:
        if time_up():
            log.warning("Stopping: wall-clock limit reached.")
            break

        if next_task_id > max_total_tasks:
            log.warning("Stopping: max_total_tasks reached.")
            break

        if accepted_valid >= target_accepts:
            break

        pump_enqueues(budget=8192)

        try:
            res: Result = result_q.get(timeout=0.2)
        except queue.Empty:
            # If nothing is happening and there is nothing left to enqueue, stop
            if not pending_tasks and not expand_q:
                log.warning("Stopping: no pending work (queues drained).")
                break
            continue

        if handle_result(res):
            accepted_valid += 1
            log.info(f"Accepted valid nodes: {accepted_valid}/{target_accepts}")

    # Choose best (write final)
    if best_seen is None or best_seen.state.svg is None:
        shutdown_all("No valid SVG produced.", terminate=True)
        raise SystemExit("No valid SVG produced.")

    try:
        with open(output_svg_path, "w", encoding="utf-8") as f:
            f.write(best_seen.state.svg)
        log.info(f"Final SVG written to: {output_svg_path} (best score={best_seen.score:.6f})")
    except Exception as e:
        shutdown_all(f"Failed to write final SVG '{output_svg_path}': {e}", terminate=True)
        raise SystemExit(1)

    # Shutdown workers
    shutdown_all("completed", terminate=False)
