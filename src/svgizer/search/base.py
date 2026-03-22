from enum import Enum
from typing import List, Protocol, Tuple, Optional
from svgizer.models import SearchNode, ChainState, Result


class StrategyType(str, Enum):
    GENETIC = "genetic"
    GREEDY = "greedy"


class SearchStrategy(Protocol):
    """Protocol for the 'brains' of the search (selection and evolution)."""

    def select_parent(
        self, nodes: List[SearchNode], progress: float
    ) -> Tuple[int, Optional[int]]:
        """Decides which node(s) to mutate or crossover next."""
        ...

    def create_new_state(self, parent_state: ChainState, result: Result) -> ChainState:
        """Handles temperature bumping, staleness, and state transition."""
        ...

    @property
    def top_k_count(self) -> int: ...
