import random
from typing import Generic, TypeVar

from svgizer.search.models import ChainState, Result, SearchNode

TState = TypeVar("TState")


class GreedyHillClimbingStrategy(Generic[TState]):
    """Parallel random beam search.

    `seeds` beams run independently; each epoch is a full restart from fresh
    LLM seeds.  The globally best node is preserved across epochs by the engine.
    """

    def __init__(self, seeds: int = 1):
        self.seeds = seeds

    @property
    def top_k_count(self) -> int:
        return self.seeds

    def select_parent(
        self, nodes: list[SearchNode[TState]], progress: float
    ) -> tuple[int, int | None]:
        """Pick a random beam — spreads work evenly across all active beams."""
        _ = progress
        if not nodes:
            return 0, None
        return random.choice(nodes).id, None

    def should_diversify(self, pool: list[SearchNode[TState]]) -> tuple[bool, float]:
        _ = pool
        return False, 0.0

    def epoch_seeds(
        self, pool: list[SearchNode[TState]], max_seeds: int
    ) -> list[SearchNode[TState]]:
        """Return [] to signal a full restart from the initial empty node."""
        _ = pool, max_seeds
        return []

    def create_new_state(self, result: Result) -> ChainState[TState]:
        return ChainState(
            score=result.score,
            payload=result.payload,
        )
