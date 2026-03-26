import random
from typing import Generic, TypeVar

from svgizer.search.models import ChainState, Result, SearchNode
from svgizer.search.utils import calculate_elite_prob, choose_from_top_k_weighted

TState = TypeVar("TState")


class GeneticPoolStrategy(Generic[TState]):
    def __init__(
        self,
        top_k: int = 3,
        elite_start: float = 0.70,
        elite_end: float = 0.10,
        crossover_prob: float = 0.25,
    ):
        self.top_k = top_k
        self.elite_start = elite_start
        self.elite_end = elite_end
        self.crossover_prob = crossover_prob

    @property
    def top_k_count(self) -> int:
        return self.top_k

    def select_parent(
        self, nodes: list[SearchNode[TState]], progress: float
    ) -> tuple[int, int | None]:
        if not nodes:
            return 0, None

        best_k = sorted(nodes, key=lambda n: n.score)[: self.top_k]

        if len(best_k) >= 2 and random.random() < self.crossover_prob:
            p1, p2 = random.sample(best_k, 2)
            return p1.id, p2.id

        best_node = best_k[0]
        elite_prob = calculate_elite_prob(progress, self.elite_start, self.elite_end)

        if random.random() < elite_prob:
            return choose_from_top_k_weighted(best_k), None
        return best_node.id, None

    def create_new_state(
        self, parent_state: ChainState[TState], result: Result
    ) -> ChainState[TState]:
        return ChainState(
            score=result.score,
            payload=result.payload,
        )
