import sys

import pytest

from svgizer.cli import parse_args
from svgizer.search import StrategyType


def test_parse_args(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["svgizer", "input.png", "--workers", "4", "--max-accepts", "10"]
    )
    args = parse_args()
    assert args.image == "input.png"
    assert args.workers == 4
    assert args.max_accepts == 10
    assert args.strategy == StrategyType.NSGA.value


# ---------------------------------------------------------------------------
# max_wall_seconds conversion
# ---------------------------------------------------------------------------


def test_max_wall_seconds_zero_becomes_none(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["svgizer", "img.png", "--max-wall-seconds", "0"])
    args = parse_args()
    assert args.max_wall_seconds is None


def test_max_wall_seconds_negative_becomes_none(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["svgizer", "img.png", "--max-wall-seconds", "-10"]
    )
    args = parse_args()
    assert args.max_wall_seconds is None


def test_max_wall_seconds_positive_kept(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["svgizer", "img.png", "--max-wall-seconds", "120"]
    )
    args = parse_args()
    assert args.max_wall_seconds == 120.0


# ---------------------------------------------------------------------------
# Boundary validation errors
# ---------------------------------------------------------------------------


def test_max_accepts_zero_raises(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["svgizer", "img.png", "--max-accepts", "0"])
    with pytest.raises(SystemExit):
        parse_args()


def test_workers_zero_raises(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["svgizer", "img.png", "--workers", "0"])
    with pytest.raises(SystemExit):
        parse_args()


def test_pool_size_zero_raises(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["svgizer", "img.png", "--pool-size", "0"])
    with pytest.raises(SystemExit):
        parse_args()


def test_image_long_side_negative_raises(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["svgizer", "img.png", "--image-long-side", "-1"])
    with pytest.raises(SystemExit):
        parse_args()


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_default_pool_size(monkeypatch):
    from svgizer.cli import DEFAULT_POOL_SIZE

    monkeypatch.setattr(sys, "argv", ["svgizer", "img.png"])
    args = parse_args()
    assert args.pool_size == DEFAULT_POOL_SIZE


def test_default_llm_rate(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["svgizer", "img.png"])
    args = parse_args()
    assert args.llm_rate == 0.05


def test_default_patience_zero(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["svgizer", "img.png"])
    args = parse_args()
    assert args.patience == 0
