from dataclasses import dataclass
from typing import Any, Protocol

from PIL import Image


class Scorer(Protocol):
    def prepare_reference(self, original_rgb: Image.Image) -> Any: ...

    def score(self, reference: Any, candidate_png: bytes) -> float: ...


@dataclass(frozen=True)
class ScoreConfig:
    target_long_side: int = 256
    w_vision: float = 0.85
    w_color: float = 0.15


DEFAULT_CONFIG = ScoreConfig()
