import pytest
from svgizer.models import SearchNode, ChainState, Result
from svgizer.search.genetic import GeneticPoolStrategy

@pytest.fixture
def strategy():
    return GeneticPoolStrategy(top_k=2, temp_step=0.5, max_temp=2.0, stale_threshold=1)

def test_genetic_selection_logic(strategy):
    # Setup two nodes: node 1 is better (lower score)
    node_low = SearchNode(score=0.1, id=1, parent_id=0, state=None)
    node_high = SearchNode(score=0.9, id=2, parent_id=0, state=None)
    nodes = [node_high, node_low]

    # With elite_prob=0 (progress=1.0), it should always pick the absolute best (node 1)
    selected_id = strategy.select_parent(nodes, progress=1.0)
    assert selected_id == 1

def test_temperature_bumping_on_staleness(strategy):
    parent_state = ChainState(
        svg="<svg>old</svg>",
        model_temperature=0.6,
        stale_hits=0,
        score=0.5,
        raster_data_url=None,
        raster_preview_data_url=None,
        invalid_msg=None
    )

    # Result with identical SVG
    result = Result(
        task_id=1, parent_id=1, worker_slot=0,
        svg="<svg>old</svg>", valid=True, invalid_msg=None,
        raster_png=b"", score=0.5, used_temperature=0.6, change_summary=""
    )

    new_state = strategy.create_new_state(parent_state, result)

    # Temperature should have bumped by 0.5 because stale_threshold is 1
    assert new_state.model_temperature == 1.1
    assert new_state.stale_hits == 0 # Reset after bump