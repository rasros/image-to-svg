from svgizer.svg.worker import _use_llm


def test_use_llm_no_svg_always_true():
    """When there is no SVG yet, always call the LLM regardless of rate."""
    for _ in range(20):
        assert _use_llm(has_svg=False, llm_rate=0.0) is True


def test_use_llm_rate_zero_never_calls():
    """With an existing SVG and rate=0, never call the LLM."""
    for _ in range(20):
        assert _use_llm(has_svg=True, llm_rate=0.0) is False


def test_use_llm_rate_one_always_calls():
    """With rate=1.0, always call the LLM even when an SVG exists."""
    for _ in range(20):
        assert _use_llm(has_svg=True, llm_rate=1.0) is True


def test_use_llm_stochastic(monkeypatch):
    """With rate=0.5, roughly half of calls should return True."""
    import random

    values = [0.3, 0.7, 0.3, 0.7, 0.3]
    monkeypatch.setattr(random, "random", lambda: values.pop(0))
    results = [_use_llm(has_svg=True, llm_rate=0.5) for _ in range(5)]
    assert results == [True, False, True, False, True]
