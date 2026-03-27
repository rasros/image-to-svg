import dataclasses
import threading
import time
from collections import deque


@dataclasses.dataclass
class SearchStats:
    strategy_name: str = ""
    model_name: str = ""
    epoch_patience: int = 0

    epoch: int = 0
    epoch_no_improve: int = 0
    llm_pressure: float = 0.0
    pool_diversity: float = 1.0  # mean pairwise Jaccard distance in active pool (0-1)
    epoch_diversity: float = 0.0  # --epoch-diversity threshold (0 = disabled)

    tasks_completed: int = 0
    accepted_count: int = 0
    pool_rejected_count: int = 0
    invalid_count: int = 0

    llm_rate: float = 0.0  # configured llm_rate, used to compute effective call rate

    llm_call_count: int = 0
    llm_invalid_count: int = 0  # LLM calls that produced invalid/unparseable SVG
    llm_accepted_count: int = 0

    mutation_call_count: int = 0
    mutation_accepted_count: int = 0

    duplicate_count: int = 0  # results rejected as duplicate genomes

    best_score: float = float("inf")
    # (elapsed_seconds, score) on each new-best event; kept for seeding but not graphed
    score_history: deque = dataclasses.field(default_factory=lambda: deque(maxlen=80))

    # Per-accept flag: 1.0 if that accept was a new best, else 0.0
    activity_window: deque = dataclasses.field(default_factory=lambda: deque(maxlen=50))
    # Sampled improvement rate over time (fraction of activity_window that are new bests)
    convergence_history: deque = dataclasses.field(
        default_factory=lambda: deque(maxlen=60)
    )
    _conv_sample_counter: int = dataclasses.field(
        default=0, init=False, repr=False, compare=False
    )

    # Captured from logging by DashboardLogHandler
    recent_events: deque = dataclasses.field(default_factory=lambda: deque(maxlen=8))

    start_time: float = dataclasses.field(default_factory=time.monotonic)
    _lock: threading.Lock = dataclasses.field(
        default_factory=threading.Lock, init=False, repr=False, compare=False
    )

    def elapsed(self) -> float:
        return time.monotonic() - self.start_time

    def accept_rate(self) -> float:
        return (
            self.accepted_count / self.tasks_completed if self.tasks_completed else 0.0
        )

    def pool_rejected_rate(self) -> float:
        return (
            self.pool_rejected_count / self.tasks_completed
            if self.tasks_completed
            else 0.0
        )

    def invalid_rate(self) -> float:
        return (
            self.invalid_count / self.tasks_completed if self.tasks_completed else 0.0
        )

    def llm_valid_rate(self) -> float:
        valid = self.llm_call_count - self.llm_invalid_count
        return valid / self.llm_call_count if self.llm_call_count else 0.0

    def llm_accept_rate(self) -> float:
        return (
            self.llm_accepted_count / self.llm_call_count
            if self.llm_call_count
            else 0.0
        )

    def effective_llm_rate(self) -> float:
        """Actual fraction of tasks that call the LLM (pressure × llm_rate)."""
        return self.llm_pressure * self.llm_rate

    def duplicate_rate(self) -> float:
        return (
            self.duplicate_count / self.tasks_completed if self.tasks_completed else 0.0
        )

    def mutation_accept_rate(self) -> float:
        return (
            self.mutation_accepted_count / self.mutation_call_count
            if self.mutation_call_count
            else 0.0
        )

    def improvement_rate(self) -> float:
        """Fraction of recent accepts that set a new best score."""
        if not self.activity_window:
            return 0.0
        return sum(self.activity_window) / len(self.activity_window)

    def stagnation_fraction(self) -> float:
        if self.epoch_patience <= 0:
            return 0.0
        return min(1.0, self.epoch_no_improve / self.epoch_patience)
