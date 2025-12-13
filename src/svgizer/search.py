from __future__ import annotations

import base64
import difflib
import io
import logging
import os
import queue
import random
import re
import time
from typing import Optional, List, Any, Tuple, Dict

import multiprocessing as mp

import cairosvg
from openai import OpenAI
from PIL import Image

from svgizer.models import ChainState, SearchNode, Task, Result, INVALID_SCORE
from svgizer.diff_scores import pixel_diff_score
from svgizer.openai_iface import summarize_changes, call_openai_for_svg, extract_svg_fragment, is_valid_svg

TEMP_STEP = 0.3
MAX_TEMP = 1.6

STALENESS_THRESHOLD = 0.995
STALE_HITS_BEFORE_BUMP = 1

# New (score-first, includes parent):
#   score00000.073180_node00022_parent00018.svg
NODE_FILE_RE_NEW = re.compile(r"^score([0-9.]+)_node(\d+)_parent(\d+)\.svg$")
# Old:
#   output_node00001_score0.123456.svg
NODE_FILE_RE_OLD = re.compile(r"_node(\d+)_score([0-9.]+)\.svg$")


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


def _score_key(score: float) -> str:
    # fixed width so lex sort == numeric sort
    return f"{score:012.6f}"


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
        log.error("OPENAI_API_KEY environment variable is not set.")
        return

    client = OpenAI(api_key=api_key)
    original_rgb = Image.open(io.BytesIO(original_png_bytes)).convert("RGB")

    while True:
        task: Any = task_q.get()
        if task is None:
            return

        assert isinstance(task, Task)
        parent = task.parent_state

        # Small deterministic jitter by worker slot to encourage diversity across workers.
        temperature = min(MAX_TEMP, max(0.0, parent.model_temperature + (task.worker_slot * 0.07)))

        change_summary = None
        if parent.svg and parent.raster_data_url:
            try:
                change_summary = summarize_changes(
                    client=client,
                    original_data_url=original_data_url,
                    iter_index=task.parent_id,
                    rasterized_svg_data_url=parent.raster_data_url,
                )
            except Exception as e:
                log.debug(f"Summary failed task={task.task_id}: {e}")

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
                diversity_hint=f"parent={task.parent_id} worker={task.worker_slot}",
            )
            svg = extract_svg_fragment(raw)
        except Exception as e:
            result_q.put(
                Result(
                    task_id=task.task_id,
                    parent_id=task.parent_id,
                    worker_slot=task.worker_slot,
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

        valid, err = is_valid_svg(svg)
        if not valid:
            result_q.put(
                Result(
                    task_id=task.task_id,
                    parent_id=task.parent_id,
                    worker_slot=task.worker_slot,
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

        try:
            png = rasterize_svg_to_png_bytes(svg)
            score = pixel_diff_score(original_rgb, png)
        except Exception as e:
            result_q.put(
                Result(
                    task_id=task.task_id,
                    parent_id=task.parent_id,
                    worker_slot=task.worker_slot,
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
                worker_slot=task.worker_slot,
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
    nodes_dir: str,
    base_name: str,
    ext: str,
    log: logging.Logger,
    base_model_temperature: float,
) -> Tuple[List[SearchNode], Optional[SearchNode], int]:
    accepted: List[SearchNode] = []
    best_seen: Optional[SearchNode] = None
    max_id = 0

    scan_paths: List[Tuple[str, str]] = []
    if os.path.isdir(nodes_dir):
        scan_paths.append((nodes_dir, "new"))
    out_dir = os.path.dirname(base_name) or "."
    scan_paths.append((out_dir, "old"))

    for directory, mode in scan_paths:
        try:
            filenames = os.listdir(directory)
        except Exception:
            continue

        for fn in filenames:
            if not fn.endswith(ext):
                continue

            path = os.path.join(directory, fn)

            if mode == "new":
                m = NODE_FILE_RE_NEW.match(fn)
                if not m:
                    continue
                score = float(m.group(1))
                node_id = int(m.group(2))
                parent_id = int(m.group(3))
            else:
                m = NODE_FILE_RE_OLD.search(fn)
                if not m:
                    continue
                node_id = int(m.group(1))
                score = float(m.group(2))
                parent_id = 0  # not recoverable from old filenames

            try:
                with open(path, "r", encoding="utf-8") as f:
                    svg = f.read()
                png = rasterize_svg_to_png_bytes(svg)
                raster_data_url = png_bytes_to_data_url(png)
            except Exception as e:
                log.warning(f"Resume: failed to load {path}: {e}")
                continue

            state = ChainState(
                svg=svg,
                raster_data_url=raster_data_url,
                score=score,
                model_temperature=base_model_temperature,
                stale_hits=0,
                invalid_msg=None,
            )
            node = SearchNode(score=score, id=node_id, parent_id=parent_id, state=state)
            accepted.append(node)

            if best_seen is None or node.score < best_seen.score:
                best_seen = node
            max_id = max(max_id, node_id)

    accepted.sort(key=lambda n: n.id)
    return accepted, best_seen, max_id


def run_search(
    image_path: str,
    output_svg_path: str,
    max_accepts: int,
    workers: int,
    base_model_temperature: float,
    max_total_tasks: int,
    max_wall_seconds: Optional[float],
    resume: bool,
    top_k: int,
    write_top_k_each: int,
    write_lineage: bool,
    log_level: str,
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

    out_dir = os.path.dirname(base_name) or "."
    nodes_dir = os.path.join(out_dir, os.path.basename(base_name) + "_nodes")
    os.makedirs(nodes_dir, exist_ok=True)

    lineage_tsv_path = os.path.join(out_dir, os.path.basename(base_name) + "_lineage.tsv")
    lineage_dot_path = os.path.join(out_dir, os.path.basename(base_name) + "_lineage.dot")

    start_time = time.monotonic()

    def time_up() -> bool:
        return max_wall_seconds is not None and (time.monotonic() - start_time) >= max_wall_seconds

    # Multiprocessing: main feeds tasks, workers pull when ready.
    ctx = mp.get_context("spawn")
    task_q: mp.Queue = ctx.Queue(maxsize=max(64, workers * 8))
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
        for _ in procs:
            try:
                task_q.put_nowait(None)
            except Exception:
                pass
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

    def write_lineage_files(node_info: Dict[int, Tuple[int, float, str]]) -> None:
        if not write_lineage:
            return
        try:
            with open(lineage_tsv_path, "w", encoding="utf-8") as f:
                f.write("node_id\tparent_id\tscore\tpath\n")
                for nid in sorted(node_info.keys()):
                    pid, sc, pth = node_info[nid]
                    f.write(f"{nid}\t{pid}\t{sc:.6f}\t{pth}\n")

            with open(lineage_dot_path, "w", encoding="utf-8") as f:
                f.write("digraph lineage {\n")
                f.write('  rankdir="LR";\n')
                for nid in sorted(node_info.keys()):
                    pid, sc, _ = node_info[nid]
                    f.write(f'  n{nid} [label="{nid}\\n{sc:.6f}"];\n')
                    if pid != 0:
                        f.write(f"  n{pid} -> n{nid};\n")
                f.write("}\n")
        except Exception as e:
            log.warning(f"Failed writing lineage files: {e}")

    # Root state
    root_state = ChainState(
        svg=None,
        raster_data_url=None,
        score=INVALID_SCORE,
        model_temperature=base_model_temperature,
        stale_hits=0,
        invalid_msg=None,
    )

    node_states: Dict[int, ChainState] = {0: root_state}
    node_info: Dict[int, Tuple[int, float, str]] = {}

    accepted_nodes: List[SearchNode] = []
    best_node: Optional[SearchNode] = None
    next_node_id = 0

    if resume:
        prior_nodes, prior_best, max_id = _load_resume_nodes(nodes_dir, base_name, ext, log, base_model_temperature)
        if prior_nodes:
            accepted_nodes.extend(prior_nodes)
            best_node = prior_best
            next_node_id = max_id
            for n in prior_nodes:
                node_states[n.id] = n.state

    def choose_parent_id() -> int:
        # Your requested behavior: always refine the current best available.
        if best_node is not None:
            return best_node.id
        return 0

    # Temperature bumping based on staleness (stored on parent lineage)
    def next_lineage_temp(parent_id: int, parent_svg: Optional[str], child_svg: str) -> Tuple[float, int]:
        parent_state = node_states.get(parent_id, root_state)
        next_temp = parent_state.model_temperature
        stale_hits = parent_state.stale_hits

        if parent_svg and is_stale(parent_svg, child_svg):
            stale_hits += 1
            if stale_hits >= STALE_HITS_BEFORE_BUMP and next_temp < MAX_TEMP:
                next_temp = min(MAX_TEMP, next_temp + TEMP_STEP)
                stale_hits = 0
        else:
            stale_hits = 0
        return next_temp, stale_hits

    # Task accounting
    next_task_id = 1
    tasks_enqueued = 0
    tasks_completed = 0
    accepted_valid = 0
    in_flight = 0

    def enqueue_one(worker_slot: int) -> bool:
        nonlocal next_task_id, tasks_enqueued, in_flight
        if next_task_id > max_total_tasks:
            return False

        pid = choose_parent_id()
        pstate = node_states.get(pid, root_state)

        t = Task(
            task_id=next_task_id,
            parent_id=pid,
            parent_state=pstate,
            worker_slot=worker_slot,
        )

        try:
            task_q.put_nowait(t)
        except queue.Full:
            return False

        next_task_id += 1
        tasks_enqueued += 1
        in_flight += 1
        return True

    # Start: exactly `workers` initial tasks (root if no resume-best, otherwise best)
    for slot in range(workers):
        if not enqueue_one(worker_slot=slot):
            break

    while True:
        if time_up():
            log.warning("Stopping: wall-clock limit reached.")
            break
        if accepted_valid >= max_accepts:
            break
        if tasks_completed >= max_total_tasks:
            break
        if in_flight == 0:
            # nothing running and can't enqueue -> stop
            break

        try:
            res: Result = result_q.get(timeout=0.2)
        except queue.Empty:
            continue

        in_flight = max(0, in_flight - 1)
        tasks_completed += 1

        # Fatal worker errors
        if res.invalid_msg and res.invalid_msg.startswith("FATAL:"):
            shutdown_all(res.invalid_msg, terminate=True)
            raise SystemExit(res.invalid_msg)

        # Immediately enqueue a replacement task for the worker slot that just freed up
        # (unless we're stopping).
        if (accepted_valid < max_accepts) and (next_task_id <= max_total_tasks) and (not time_up()):
            enqueue_one(worker_slot=res.worker_slot)

        # Ignore invalid results
        if (not res.valid) or (res.svg is None) or (res.raster_png is None) or (res.score >= INVALID_SCORE):
            continue

        # Create node
        next_node_id += 1

        parent_state = node_states.get(res.parent_id, root_state)
        next_temp, stale_hits = next_lineage_temp(res.parent_id, parent_state.svg, res.svg)

        state = ChainState(
            svg=res.svg,
            raster_data_url=png_bytes_to_data_url(res.raster_png),
            score=res.score,
            model_temperature=next_temp,
            stale_hits=stale_hits,
            invalid_msg=None,
        )
        node = SearchNode(score=res.score, id=next_node_id, parent_id=res.parent_id, state=state)
        accepted_nodes.append(node)
        node_states[node.id] = node.state

        # Write node (score-first)
        fn = f"score{_score_key(node.score)}_node{node.id:05d}_parent{node.parent_id:05d}{ext}"
        iter_path = os.path.join(nodes_dir, fn)
        try:
            with open(iter_path, "w", encoding="utf-8") as f:
                f.write(node.state.svg or "")
        except Exception as e:
            shutdown_all(f"Failed to write {iter_path}: {e}", terminate=True)
            raise SystemExit(1)

        node_info[node.id] = (node.parent_id, node.score, iter_path)

        # Update best + top-k list
        if best_node is None or node.score < best_node.score:
            best_node = node
            log.info(f"NEW BEST node={node.id} score={node.score:.6f}")

        accepted_valid += 1

        # Optional snapshots: write current best K nodes occasionally
        if write_top_k_each > 0 and (accepted_valid % write_top_k_each == 0):
            best_k = sorted(accepted_nodes, key=lambda n: n.score)[: max(1, top_k)]
            for rank, bn in enumerate(best_k, start=1):
                pth = os.path.join(
                    nodes_dir,
                    f"TOP_rank{rank:02d}_score{_score_key(bn.score)}_node{bn.id:05d}_parent{bn.parent_id:05d}{ext}",
                )
                try:
                    with open(pth, "w", encoding="utf-8") as f:
                        f.write(bn.state.svg or "")
                except Exception:
                    pass

        if write_lineage and (accepted_valid % 10 == 0):
            write_lineage_files(node_info)

    if best_node is None or best_node.state.svg is None or best_node.score >= INVALID_SCORE:
        shutdown_all("No valid SVG produced.", terminate=True)
        raise SystemExit("No valid SVG produced.")

    # Final output = best
    try:
        with open(output_svg_path, "w", encoding="utf-8") as f:
            f.write(best_node.state.svg)
        log.info(f"Final SVG written to: {output_svg_path} (best score={best_node.score:.6f})")
    except Exception as e:
        shutdown_all(f"Failed to write final SVG '{output_svg_path}': {e}", terminate=True)
        raise SystemExit(1)

    if write_lineage:
        write_lineage_files(node_info)

    shutdown_all("completed", terminate=False)
