import binascii
from enum import Enum
from typing import Protocol, TypeVar

from svgizer.search.models import ChainState, Result, SearchNode

TState = TypeVar("TState")


def compute_signature(
    text: str | None, num_perms: int = 64, ngram_size: int = 4
) -> tuple[int, ...] | None:
    """Compute a MinHash signature for O(1) Jaccard similarity estimation."""
    if not text:
        return None
    encoded = text.encode("utf-8")
    if len(encoded) < ngram_size:
        ngrams = {encoded}
    else:
        ngrams = {
            encoded[i : i + ngram_size] for i in range(len(encoded) - ngram_size + 1)
        }

    sig = []
    for i in range(num_perms):
        salt = i.to_bytes(4, "little")
        sig.append(min(binascii.crc32(salt + ng) for ng in ngrams))
    return tuple(sig)


def estimate_jaccard(
    sig1: tuple[int, ...] | None, sig2: tuple[int, ...] | None
) -> float:
    """Estimate Jaccard similarity from two MinHash signatures (0.0 to 1.0)."""
    if not sig1 or not sig2 or len(sig1) != len(sig2):
        return 0.0
    matches = sum(1 for a, b in zip(sig1, sig2, strict=False) if a == b)
    return matches / len(sig1)


class StrategyType(str, Enum):
    GREEDY = "greedy"
    NSGA = "nsga"


class SearchStrategy(Protocol[TState]):
    """Protocol for the 'brains' of the search (selection and evolution)."""

    def select_parent(
        self, nodes: list[SearchNode[TState]], progress: float
    ) -> tuple[int, int | None]:
        """Decides which node(s) to mutate or crossover next."""
        ...

    def create_new_state(self, result: Result[TState]) -> ChainState[TState]:
        """Handles state transition."""
        ...

    def should_diversify(self, pool: list[SearchNode]) -> bool:
        """Return True when the pool has converged and fresh LLM seeds are needed."""
        ...

    @property
    def top_k_count(self) -> int: ...


class StorageAdapter(Protocol[TState]):
    """
    Protocol for the 'memory' of the search.
    The SearchEngine uses this to persist every accepted step of the evolution.
    """

    def initialize(self) -> None: ...

    def save_node(self, node: SearchNode[TState]) -> None: ...

    def load_resume_nodes(self, max_nodes: int = 20) -> list[tuple[int, str]]:
        """Returns raw IDs and SVG content for re-hydration."""
        ...

    @property
    def max_node_id(self) -> int:
        """Returns the highest ID currently known to the storage backend."""
        ...
