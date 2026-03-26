from typing import Generic, TypeVar

from svgizer.search.models import ChainState, Result, SearchNode

TState = TypeVar("TState")


class GreedyHillClimbingStrategy(Generic[TState]):
    @property
    def top_k_count(self) -> int:
        return 1

    def select_parent(
        self, nodes: list[SearchNode[TState]], progress: float
    ) -> tuple[int, int | None]:
        _ = progress
        if not nodes:
            return 0, None

        best_node = min(nodes, key=lambda n: n.score)
        return best_node.id, None

    def should_diversify(self, pool: list[SearchNode[TState]]) -> bool:
        _ = pool
        return False

    def epoch_seeds(
        self, pool: list[SearchNode[TState]], max_seeds: int
    ) -> list[SearchNode[TState]]:
        valid = [n for n in pool if n.score < float("inf")]
        if not valid:
            return pool[:max_seeds]
        return sorted(valid, key=lambda n: n.score)[:max_seeds]

    def create_new_state(self, result: Result) -> ChainState[TState]:
        return ChainState(
            score=result.score,
            payload=result.payload,
        )
