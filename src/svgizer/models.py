from __future__ import annotations

import dataclasses

INVALID_SCORE = 1e9


@dataclasses.dataclass
class ChainState:
    svg: str | None
    raster_data_url: str | None  # FULL-res (for lineage / inspection)
    raster_preview_data_url: str | None  # DOWNSCALED (for OpenAI)
    score: float
    model_temperature: float
    stale_hits: int
    invalid_msg: str | None
    change_summary: str | None = None


@dataclasses.dataclass(order=True)
class SearchNode:
    score: float
    id: int = dataclasses.field(compare=False)
    parent_id: int = dataclasses.field(compare=False)
    state: ChainState = dataclasses.field(compare=False)


@dataclasses.dataclass
class Task:
    task_id: int
    parent_id: int
    parent_state: ChainState
    worker_slot: int  # used only for diversity/jitter
    secondary_parent_id: int | None = None
    secondary_parent_state: ChainState | None = None


@dataclasses.dataclass
class Result:
    task_id: int
    parent_id: int
    worker_slot: int
    svg: str | None
    valid: bool
    invalid_msg: str | None
    raster_png: bytes | None
    score: float
    used_temperature: float
    change_summary: str | None
    secondary_parent_id: int | None = None
