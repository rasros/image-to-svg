from enum import Enum
from typing import List, Protocol
from svgizer.models import SearchNode, ChainState, Result


class StrategyType(str, Enum):
    GENETIC = "genetic"
    GREEDY = "greedy"


class SearchStrategy(Protocol):
    """Protocol for the 'brains' of the search (selection and evolution)."""

    def select_parent(self, nodes: List[SearchNode], progress: float) -> int:
        """Decides which node to mutate next."""
        ...

    def create_new_state(self, parent_state: ChainState, result: Result) -> ChainState:
        """Handles temperature bumping, staleness, and state transition."""
        ...

    @property
    def top_k_count(self) -> int: ...
