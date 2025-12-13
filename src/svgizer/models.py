from __future__ import annotations

import dataclasses
from typing import Optional


INVALID_SCORE = 1e9


@dataclasses.dataclass
class BeamState:
    svg: Optional[str]
    raster_data_url: Optional[str]  # rasterized svg as png data-url
    score: float
    temperature: float
    stale_hits: int
    invalid_msg: Optional[str]


@dataclasses.dataclass(order=True)
class SearchNode:
    score: float
    id: int = dataclasses.field(compare=False)
    state: BeamState = dataclasses.field(compare=False)


@dataclasses.dataclass
class Task:
    task_id: int
    parent_id: int
    parent_state: BeamState
    candidate_index: int


@dataclasses.dataclass
class Result:
    task_id: int
    parent_id: int
    candidate_index: int
    svg: Optional[str]
    valid: bool
    invalid_msg: Optional[str]
    raster_png: Optional[bytes]
    score: float
    used_temperature: float
    change_summary: Optional[str]

