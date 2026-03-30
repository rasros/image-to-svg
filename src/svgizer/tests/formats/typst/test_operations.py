import importlib.util

import pytest
from PIL import Image

from svgizer.formats.typst.operations import (
    _random_block_swap,
    _random_numeric_tweak,
    crossover_with_micro_search,
    mutate_with_micro_search,
)

_TYPST_AVAILABLE = importlib.util.find_spec("typst") is not None

_TYPST_CODE = """#set page(width: auto, height: auto, margin: 0pt)
#rect(width: 100pt, height: 50pt, fill: red)
#circle(radius: 20pt, fill: blue)
"""


def test_random_numeric_tweak_changes_value_keeps_unit():
    code = "#rect(width: 100pt)"
    changed = False
    for _ in range(30):
        result = _random_numeric_tweak(code)
        if result != code:
            changed = True
            assert "pt)" in result
            assert "100" not in result
            break
    assert changed


def test_random_numeric_tweak_handles_percentages():
    code = "#rect(width: 50%)"
    result = _random_numeric_tweak(code)
    assert result != code or result == code  # Could rarely be exactly 1.0 factor
    assert "%" in result


def test_random_numeric_tweak_ignores_no_units():
    code = "#rect(width: 100)"
    result = _random_numeric_tweak(code)
    assert result == code


def test_random_block_swap_swaps_lines():
    code = "#set page()\n#rect()\n#circle()\n#line()"
    changed = False
    for _ in range(30):
        result = _random_block_swap(code)
        if result != code:
            changed = True
            lines = result.split("\n")
            assert lines[0] == "#set page()"  # Ensure index 0 is untouched
            assert set(lines) == {"#set page()", "#rect()", "#circle()", "#line()"}
            break
    assert changed


def test_random_block_swap_ignores_short_code():
    code = "#set page()\n#rect()"
    assert _random_block_swap(code) == code


@pytest.mark.skipif(not _TYPST_AVAILABLE, reason="typst package not installed")
def test_mutate_with_micro_search_returns_typst_string():
    target = Image.new("RGB", (32, 32), color="red")
    result, summary = mutate_with_micro_search(_TYPST_CODE, target, num_trials=3)
    assert isinstance(result, str)
    assert "#set page" in result
    assert summary == "local typst mutation"


@pytest.mark.skipif(not _TYPST_AVAILABLE, reason="typst package not installed")
def test_crossover_with_micro_search_returns_typst_string():
    code_b = (
        """#set page(width: auto, height: auto, margin: 0pt)\n\n#polygon(fill: green)"""
    )
    target = Image.new("RGB", (32, 32), color="green")
    # Add double newlines to code A to allow crossover points
    code_a = _TYPST_CODE.replace("\n", "\n\n")

    result, summary = crossover_with_micro_search(code_a, code_b, target, num_trials=3)
    assert isinstance(result, str)
    assert summary in ("typst block crossover", "local typst mutation")


@pytest.mark.skipif(not _TYPST_AVAILABLE, reason="typst package not installed")
def test_crossover_falls_back_to_mutation_when_no_blocks():
    code_b = "#set page()\n#rect()"  # No double newlines
    target = Image.new("RGB", (32, 32), color="red")
    result, summary = crossover_with_micro_search(
        _TYPST_CODE, code_b, target, num_trials=3
    )
    assert isinstance(result, str)
    assert summary == "local typst mutation"
