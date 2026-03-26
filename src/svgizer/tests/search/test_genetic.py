import pytest

from svgizer.search import ChainState, GeneticPoolStrategy, Result, SearchNode


@pytest.fixture
def strategy():
    return GeneticPoolStrategy(
        top_k=2,
    )


def test_select_parent_returns_best_on_full_progress(strategy):
    dummy_state = ChainState(score=0.0, payload=None)
    n1 = SearchNode(score=0.1, id=1, parent_id=0, state=dummy_state)
    n2 = SearchNode(score=0.9, id=2, parent_id=0, state=dummy_state)

    selected, secondary = strategy.select_parent([n1, n2], progress=1.0)
    assert selected in [1, 2]
    assert secondary is None or secondary in [1, 2]


def test_create_new_state_propagates_payload(strategy):
    res = Result(
        task_id=1,
        parent_id=1,
        worker_slot=0,
        valid=True,
        score=0.5,
        payload="new_identical_payload",
    )

    new_state = strategy.create_new_state(res)

    assert new_state.score == 0.5
    assert new_state.payload == "new_identical_payload"
