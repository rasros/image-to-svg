#!/usr/bin/env python3
"""
Plot stats for one or more runs.

Usage:
    uv run scripts/plot_run.py <run_dir>
    uv run scripts/plot_run.py output/runs/2024-01-01_12-00-00
    uv run scripts/plot_run.py output/runs/           # overlay all runs
    uv run scripts/plot_run.py output.svg             # all runs for this output

Options:
    --output FILE   Save to FILE instead of showing interactively.
    --top N         Only plot the N most recent runs when given a runs/ dir (default: all).
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ── Data loading ─────────────────────────────────────────────────────────────

def load_stats(run_dir: Path) -> dict:
    """Load wide-format stats.csv. Returns a dict with summary scalars and
    score_history → list of (elapsed, score) derived from new-best rows."""
    path = run_dir / "stats.csv"
    if not path.exists():
        return {}

    rows = []
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        return {}

    last = rows[-1]

    def _float(key, default=0.0):
        try:
            v = last.get(key, "")
            return float(v) if v != "" else default
        except (ValueError, TypeError):
            return default

    tasks = _float("tasks_completed") or 1
    llm   = _float("llm_call_count")
    mut   = _float("mutation_call_count")
    acc   = _float("accepted_count")
    llm_acc = _float("llm_accepted_count")
    llm_inv = _float("llm_invalid_count")
    mut_acc = _float("mutation_accepted_count")

    stats: dict = {
        "strategy":               last.get("strategy", ""),
        "model":                  last.get("model", ""),
        "elapsed_seconds":        _float("elapsed"),
        "best_score":             _float("best_score", float("inf")),
        "tasks_completed":        tasks,
        "accepted_count":         acc,
        "pool_rejected_count":    _float("pool_rejected_count"),
        "invalid_count":          _float("invalid_count"),
        "accept_rate":            acc / tasks,
        "pool_rejected_rate":     _float("pool_rejected_count") / tasks,
        "invalid_rate":           _float("invalid_count") / tasks,
        "llm_call_count":         llm,
        "llm_accepted_count":     llm_acc,
        "llm_invalid_count":      llm_inv,
        "llm_valid_rate":         (llm - llm_inv) / llm if llm else 0.0,
        "llm_accept_rate":        llm_acc / llm if llm else 0.0,
        "mutation_call_count":    mut,
        "mutation_accepted_count": mut_acc,
        "mutation_accept_rate":   mut_acc / mut if mut else 0.0,
        "llm_rate":               _float("llm_rate"),
        "llm_pressure_final":     _float("llm_pressure"),
        "epochs_completed":       _float("epoch"),
        "epoch_patience_config":  _float("epoch_patience"),
        "epoch_diversity_config": _float("epoch_diversity"),
        "epoch_variance_config":  _float("epoch_variance"),
        "pool_diversity_final":   _float("pool_diversity"),
        "pool_score_std_final":   _float("pool_score_std"),
    }

    # Reconstruct score_history from rows where best_score decreased.
    history = []
    prev_best = float("inf")
    for row in rows:
        try:
            elapsed = float(row.get("elapsed", 0) or 0)
            bs_raw = row.get("best_score", "")
            if bs_raw == "":
                continue
            bs = float(bs_raw)
            if bs < prev_best:
                history.append((elapsed, bs))
                prev_best = bs
        except (ValueError, TypeError):
            pass
    stats["score_history"] = history

    return stats


def load_lineage(run_dir: Path) -> list[dict]:
    path = run_dir / "lineage.csv"
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rows.append({
                    "id": int(row["id"]),
                    "parent": int(row["parent"]),
                    "epoch": int(row.get("epoch", 0) or 0),
                    "score": float(row["score"]),
                    "complexity": float(row.get("complexity", 0) or 0),
                })
            except (KeyError, ValueError):
                pass
    return rows


def resolve_run_dirs(path: Path, top: int | None) -> list[Path]:
    if path.suffix == ".svg":
        runs_dir = path.parent / path.stem / "runs"
    elif path.name == "runs" and path.is_dir():
        runs_dir = path
    elif (path / "runs").is_dir():
        runs_dir = path / "runs"
    elif (path / "nodes").is_dir() or (path / "lineage.csv").exists():
        return [path]
    else:
        # recurse
        found = sorted(path.rglob("runs"), key=lambda p: str(p))
        dirs = []
        for rd in found:
            dirs.extend(sorted([d for d in rd.iterdir() if d.is_dir()], key=lambda d: d.name))
        return dirs[-top:] if top else dirs

    run_dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir()], key=lambda d: d.name)
    return run_dirs[-top:] if top else run_dirs


# ── Plot helpers ──────────────────────────────────────────────────────────────

COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def _label(run_dir: Path) -> str:
    return run_dir.name


def plot_score_history(ax, runs: list[tuple[Path, dict]], lineages: list[list[dict]]):
    ax.set_title("Best score over time")
    ax.set_xlabel("Elapsed (s)")
    ax.set_ylabel("Score (lower = better)")
    for i, (run_dir, stats) in enumerate(runs):
        history = stats.get("score_history", [])
        if not history:
            continue
        xs = [0.0] + [t for t, _ in history]
        ys = [history[0][1]] + [s for _, s in history]
        color = COLORS[i % len(COLORS)]
        ax.step(xs, ys, where="post", color=color, label=_label(run_dir), linewidth=1.5)
        ax.scatter([t for t, _ in history], [s for _, s in history],
                   color=color, s=20, zorder=3)
    if len(runs) > 1:
        ax.legend(fontsize=7, loc="upper right")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))


def plot_score_distribution(ax, runs: list[tuple[Path, dict]], lineages: list[list[dict]]):
    ax.set_title("Accepted score distribution")
    ax.set_xlabel("Score")
    ax.set_ylabel("Count")
    for i, (run_dir, _) in enumerate(runs):
        lin = lineages[i]
        scores = [r["score"] for r in lin if r["score"] < float("inf")]
        if not scores:
            continue
        ax.hist(scores, bins=40, alpha=0.6, color=COLORS[i % len(COLORS)],
                label=_label(run_dir), edgecolor="none")
    if len(runs) > 1:
        ax.legend(fontsize=7)


def plot_pareto(ax, runs: list[tuple[Path, dict]], lineages: list[list[dict]]):
    ax.set_title("Score vs complexity (all accepted)")
    ax.set_xlabel("Complexity")
    ax.set_ylabel("Score")
    for i, (run_dir, _) in enumerate(runs):
        lin = lineages[i]
        xs = [r["complexity"] for r in lin if r["score"] < float("inf")]
        ys = [r["score"] for r in lin if r["score"] < float("inf")]
        if not xs:
            continue
        ax.scatter(xs, ys, s=8, alpha=0.4, color=COLORS[i % len(COLORS)],
                   label=_label(run_dir))
    if len(runs) > 1:
        ax.legend(fontsize=7)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))


def plot_score_per_epoch(ax, runs: list[tuple[Path, dict]], lineages: list[list[dict]]):
    ax.set_title("Score by epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    all_epochs = sorted({r["epoch"] for lin in lineages for r in lin})
    width = 0.8 / max(len(runs), 1)
    for i, (run_dir, _) in enumerate(runs):
        lin = lineages[i]
        by_epoch: dict[int, list[float]] = {}
        for r in lin:
            if r["score"] < float("inf"):
                by_epoch.setdefault(r["epoch"], []).append(r["score"])
        epochs = sorted(by_epoch)
        if not epochs:
            continue
        offsets = np.array(epochs, dtype=float) + (i - len(runs) / 2 + 0.5) * width
        medians = [np.median(by_epoch[e]) for e in epochs]
        q25 = [np.percentile(by_epoch[e], 25) for e in epochs]
        q75 = [np.percentile(by_epoch[e], 75) for e in epochs]
        color = COLORS[i % len(COLORS)]
        ax.bar(offsets, medians, width=width * 0.9, color=color, alpha=0.7,
               label=_label(run_dir))
        ax.errorbar(offsets, medians,
                    yerr=[np.subtract(medians, q25), np.subtract(q75, medians)],
                    fmt="none", color="black", linewidth=0.8, capsize=2)
    ax.set_xticks(all_epochs)
    if len(runs) > 1:
        ax.legend(fontsize=7)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))


def plot_task_breakdown(ax, runs: list[tuple[Path, dict]], lineages: list[list[dict]]):
    ax.set_title("Task outcome breakdown")
    labels = ["accepted", "pool_rejected", "invalid"]
    x = np.arange(len(labels))
    width = 0.8 / max(len(runs), 1)
    for i, (run_dir, stats) in enumerate(runs):
        total = stats.get("tasks_completed") or 1
        values = [
            (stats.get("accepted_count") or 0) / total,
            (stats.get("pool_rejected_count") or 0) / total,
            (stats.get("invalid_count") or 0) / total,
        ]
        offsets = x + (i - len(runs) / 2 + 0.5) * width
        ax.bar(offsets, values, width=width * 0.9,
               color=COLORS[i % len(COLORS)], alpha=0.8, label=_label(run_dir))
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Fraction of tasks")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    if len(runs) > 1:
        ax.legend(fontsize=7)


def plot_llm_vs_mutation(ax, runs: list[tuple[Path, dict]], lineages: list[list[dict]]):
    ax.set_title("LLM vs mutation accept rate")
    labels = ["LLM accept", "LLM valid", "Mut accept"]
    x = np.arange(len(labels))
    width = 0.8 / max(len(runs), 1)
    for i, (run_dir, stats) in enumerate(runs):
        values = [
            stats.get("llm_accept_rate") or 0,
            stats.get("llm_valid_rate") or 0,
            stats.get("mutation_accept_rate") or 0,
        ]
        offsets = x + (i - len(runs) / 2 + 0.5) * width
        ax.bar(offsets, values, width=width * 0.9,
               color=COLORS[i % len(COLORS)], alpha=0.8, label=_label(run_dir))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Rate")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    if len(runs) > 1:
        ax.legend(fontsize=7)


def plot_summary_text(ax, runs: list[tuple[Path, dict]], lineages: list[list[dict]]):
    ax.axis("off")
    ax.set_title("Run summary", loc="left")
    lines = []
    for run_dir, stats in runs:
        if not stats:
            continue
        best = stats.get("best_score")
        best_str = f"{best:.6f}" if best is not None else "—"
        lines.append(f"[{_label(run_dir)}]")
        lines.append(f"  best score:    {best_str}")
        lines.append(f"  elapsed:       {stats.get('elapsed_seconds', '—'):.0f}s")
        lines.append(f"  tasks:         {int(stats.get('tasks_completed') or 0):,}")
        lines.append(f"  accepted:      {int(stats.get('accepted_count') or 0):,}  ({stats.get('accept_rate', 0)*100:.1f}%)")
        lines.append(f"  llm calls:     {int(stats.get('llm_call_count') or 0):,}")
        lines.append(f"  mut calls:     {int(stats.get('mutation_call_count') or 0):,}")
        lines.append(f"  epochs:        {int(stats.get('epochs_completed') or 0)}")
        lines.append(f"  diversity:     {stats.get('pool_diversity_final', 0):.3f}")
        lines.append(f"  score std:     {stats.get('pool_score_std_final', 0):.5f}")
        lines.append(f"  model:         {stats.get('model', '—')}")
        lines.append("")
    ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes,
            fontsize=8, verticalalignment="top", fontfamily="monospace")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", help="Run dir, runs/ dir, output.svg, or project dir.")
    parser.add_argument("--output", "-o", default=None,
                        help="Save plot to this file (png/pdf/svg). Default: show interactively.")
    parser.add_argument("--top", type=int, default=None, metavar="N",
                        help="Only plot the N most recent runs.")
    args = parser.parse_args()

    run_dirs = resolve_run_dirs(Path(args.path), args.top)
    if not run_dirs:
        print(f"No run directories found at {args.path}", file=sys.stderr)
        sys.exit(1)

    runs = [(d, load_stats(d)) for d in run_dirs]
    lineages = [load_lineage(d) for d in run_dirs]

    # Filter to runs that have any data at all
    combined = [(r, s, l) for r, s, l in zip(run_dirs, [s for _, s in runs], lineages)
                if s or l]
    if not combined:
        print("No stats.csv or lineage.csv found in the given run directories.",
              file=sys.stderr)
        sys.exit(1)
    run_dirs_f = [r for r, _, _ in combined]
    runs_f = [(r, s) for r, s, _ in combined]
    lineages_f = [l for _, _, l in combined]

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle(
        f"SVGizer run stats — {run_dirs_f[0].parent.name if len(run_dirs_f) == 1 else str(Path(args.path))}",
        fontsize=11, fontweight="bold",
    )
    plt.subplots_adjust(hspace=0.38, wspace=0.32, left=0.06, right=0.98, top=0.92, bottom=0.08)

    plot_score_history(axes[0, 0], runs_f, lineages_f)
    plot_score_distribution(axes[0, 1], runs_f, lineages_f)
    plot_pareto(axes[0, 2], runs_f, lineages_f)
    plot_score_per_epoch(axes[0, 3], runs_f, lineages_f)
    plot_task_breakdown(axes[1, 0], runs_f, lineages_f)
    plot_llm_vs_mutation(axes[1, 1], runs_f, lineages_f)
    plot_summary_text(axes[1, 2], runs_f, lineages_f)
    axes[1, 3].axis("off")  # spare slot

    if args.output:
        out = Path(args.output)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved to {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
