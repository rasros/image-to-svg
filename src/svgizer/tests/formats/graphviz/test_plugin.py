import shutil

import pytest

from svgizer.formats.graphviz.plugin import GraphvizPlugin, _sanitize_dot

_DOT_AVAILABLE = shutil.which("dot") is not None

_DIGRAPH = "digraph G { A -> B }"
_GRAPH_WITH_ARROWS = "graph G { A -> B }"  # invalid: -> in undirected graph
_UNDIRECTED = "graph G { A -- B }"


# ── _sanitize_dot ─────────────────────────────────────────────────────────────


def test_sanitize_noop_for_valid_digraph():
    result = _sanitize_dot(_DIGRAPH)
    assert result == _DIGRAPH


def test_sanitize_upgrades_graph_to_digraph_when_arrow_present():
    result = _sanitize_dot(_GRAPH_WITH_ARROWS)
    assert result.startswith("digraph")


def test_sanitize_preserves_undirected_graph_without_arrows():
    result = _sanitize_dot(_UNDIRECTED)
    assert result == _UNDIRECTED


def test_sanitize_strict_graph_upgraded():
    dot = "strict graph G { A -> B }"
    result = _sanitize_dot(dot)
    assert "digraph" in result
    assert "strict" in result


def test_sanitize_already_digraph_unchanged():
    result = _sanitize_dot("digraph G { A -> B -> C }")
    assert result.count("digraph") == 1


# ── extract_from_llm ──────────────────────────────────────────────────────────


def test_extract_fenced_dot_block():
    plugin = GraphvizPlugin()
    raw = "Here is the graph:\n```dot\ndigraph G { A -> B }\n```\n"
    result = plugin.extract_from_llm(raw)
    assert result == "digraph G { A -> B }"


def test_extract_fenced_case_insensitive():
    plugin = GraphvizPlugin()
    raw = "```DOT\ndigraph G { A -> B }\n```"
    result = plugin.extract_from_llm(raw)
    assert "digraph" in result


def test_extract_raw_digraph_block():
    plugin = GraphvizPlugin()
    raw = 'Sure! digraph G { A -> B [label="edge"] }'
    result = plugin.extract_from_llm(raw)
    assert "digraph" in result
    assert "A -> B" in result


def test_extract_raw_quoted_name():
    plugin = GraphvizPlugin()
    raw = 'digraph "My Graph" { A -> B }'
    result = plugin.extract_from_llm(raw)
    assert "digraph" in result


def test_extract_sanitizes_graph_to_digraph():
    plugin = GraphvizPlugin()
    raw = "```dot\ngraph G { A -> B }\n```"
    result = plugin.extract_from_llm(raw)
    assert result.startswith("digraph")


def test_extract_fallback_returns_stripped_raw():
    plugin = GraphvizPlugin()
    raw = "  no graph here  "
    result = plugin.extract_from_llm(raw)
    assert result == "no graph here"


# ── validate / rasterize (require system graphviz) ───────────────────────────


@pytest.mark.skipif(not _DOT_AVAILABLE, reason="graphviz system binary not installed")
def test_validate_valid_digraph():
    plugin = GraphvizPlugin()
    valid, err = plugin.validate(_DIGRAPH)
    assert valid is True
    assert err is None


@pytest.mark.skipif(not _DOT_AVAILABLE, reason="graphviz system binary not installed")
def test_validate_invalid_dot():
    plugin = GraphvizPlugin()
    valid, err = plugin.validate("this is not dot at all >>>")
    assert valid is False
    assert err is not None


@pytest.mark.skipif(not _DOT_AVAILABLE, reason="graphviz system binary not installed")
def test_rasterize_returns_png_bytes():
    plugin = GraphvizPlugin()
    png = plugin.rasterize(_DIGRAPH, out_w=64, out_h=64)
    assert isinstance(png, bytes)
    assert png[:4] == b"\x89PNG"


@pytest.mark.skipif(not _DOT_AVAILABLE, reason="graphviz system binary not installed")
def test_rasterize_output_dimensions():
    import io

    from PIL import Image

    plugin = GraphvizPlugin()
    png = plugin.rasterize(_DIGRAPH, out_w=128, out_h=96)
    img = Image.open(io.BytesIO(png))
    assert img.size == (128, 96)


@pytest.mark.skipif(not _DOT_AVAILABLE, reason="graphviz system binary not installed")
def test_rasterize_fast_returns_png_bytes():
    plugin = GraphvizPlugin()
    png = plugin.rasterize_fast(_DIGRAPH, long_side=64)
    assert png is not None
    assert png[:4] == b"\x89PNG"


@pytest.mark.skipif(not _DOT_AVAILABLE, reason="graphviz system binary not installed")
def test_rasterize_fast_returns_none_on_invalid():
    plugin = GraphvizPlugin()
    result = plugin.rasterize_fast("not dot code >>>", long_side=64)
    assert result is None
