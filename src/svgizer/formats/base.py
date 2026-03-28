from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import PIL.Image


class FormatPlugin(Protocol):
    name: str
    file_extension: str  # e.g. ".svg", ".dot"

    def rasterize(self, content: str, out_w: int, out_h: int) -> bytes:
        """Render content to PNG bytes at given dimensions."""
        ...

    def rasterize_fast(self, content: str, long_side: int) -> bytes | None:
        """Fast low-res render for micro-search scoring. Return None on failure."""
        ...

    def validate(self, content: str) -> tuple[bool, str | None]:
        """Return (is_valid, error_message_or_None)."""
        ...

    def extract_from_llm(self, raw: str) -> str:
        """Parse the LLM's raw text response to extract format content."""
        ...

    def build_generate_prompt(
        self,
        image_data_url: str,
        node_index: int,
        content_prev: str | None,
        raster_preview_url: str | None,
        change_summary: str | None,
        diff_data_url: str | None,
    ) -> list[dict]:
        """Build the LLM generation/refinement prompt as content blocks."""
        ...

    def build_summarize_prompt(
        self,
        image_data_url: str,
        raster_preview_url: str | None,
        custom_goal: str | None,
        previous_summary: str | None,
    ) -> list[dict]:
        """Build the LLM summarize/critique prompt as content blocks."""
        ...

    def mutate(self, content: str, orig_img_fast: PIL.Image.Image) -> tuple[str, str]:
        """Mutate existing content. Return (new_content, change_summary)."""
        ...

    def crossover(
        self,
        content_a: str,
        content_b: str,
        orig_img_fast: PIL.Image.Image,
    ) -> tuple[str, str]:
        """Crossover two contents. Return (new_content, change_summary)."""
        ...
