import pytest
from unittest.mock import MagicMock, patch
from svgizer.search.engine import MultiprocessSearchEngine
from svgizer.models import SearchNode, Result

def test_engine_initialization():
    strategy = MagicMock()
    storage = MagicMock()
    engine = MultiprocessSearchEngine(workers=2, strategy=strategy, storage=storage, scorer_type="simple")

    assert engine.workers == 2
    assert engine.scorer_type == "simple"

@patch("svgizer.search.engine.mp.get_context")
def test_engine_termination_on_max_accepts(mock_context):
    strategy = MagicMock()
    storage = MagicMock()
    # Mock result queue to return a valid result immediately
    mock_res = MagicMock(spec=Result)
    mock_res.valid = True
    mock_res.score = 0.1
    mock_res.parent_id = 0

    engine = MultiprocessSearchEngine(workers=1, strategy=strategy, storage=storage, scorer_type="simple")
    engine.result_q.get = MagicMock(return_value=mock_res)

    # Mock strategy to return a new state
    strategy.create_new_state.return_value = MagicMock()

    nodes = [SearchNode(score=0.5, id=0, parent_id=0, state=MagicMock())]

    # Run engine with max_accepts=1. It should finish after one result.
    best = engine.run(nodes, max_accepts=1, max_wall_seconds=10, openai_image_long_side=512, original_dims=(100, 100))

    assert best.score == 0.1
    assert storage.save_node.called