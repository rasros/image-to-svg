import csv
import logging
import os
import re

from svgizer.image_utils import make_preview_data_url, rasterize_svg_to_png_bytes
from svgizer.search import ChainState, SearchNode
from svgizer.svg_adapter import SvgStatePayload

log = logging.getLogger(__name__)


class FileStorageAdapter:
    """
    Implementation of the Search Storage protocol using the local filesystem.
    Handles node serialization, CSV lineage, and state hydration for resume.
    """

    def __init__(
        self,
        output_svg_path: str,
        resume: bool = False,
        img_dims: tuple[int, int] = (512, 512),
        openai_image_long_side: int = 512,
        base_temp: float = 0.6,
    ):
        self.output_svg_path = output_svg_path
        self.resume = resume
        self.img_dims = img_dims
        self.openai_image_long_side = openai_image_long_side
        self.base_temp = base_temp
        self._max_id = 0

        # Path parsing
        base_name, ext = os.path.splitext(output_svg_path)
        self.base_name = os.path.basename(base_name)
        self.ext = ext or ".svg"
        self.out_dir = os.path.dirname(base_name) or "."
        self.nodes_dir = os.path.join(self.out_dir, f"{self.base_name}_nodes")
        self.lineage_csv = os.path.join(self.out_dir, f"{self.base_name}_lineage.csv")

    @property
    def max_node_id(self) -> int:
        return self._max_id

    def initialize(self) -> None:
        os.makedirs(self.nodes_dir, exist_ok=True)

    def save_node(self, node: SearchNode[SvgStatePayload]) -> None:
        """Saves SVG file and updates lineage CSV."""
        self._max_id = max(self._max_id, node.id)

        fn = f"score{node.score:012.6f}_node{node.id:05d}_parent{node.parent_id:05d}{self.ext}"
        path = os.path.join(self.nodes_dir, fn)

        if node.state.payload.svg:
            with open(path, "w", encoding="utf-8") as f:
                f.write(node.state.payload.svg)

        # Append to CSV record
        exists = os.path.isfile(self.lineage_csv)
        try:
            with open(self.lineage_csv, "a", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                if not exists:
                    w.writerow(["id", "parent", "score", "temp", "summary"])
                w.writerow(
                    [
                        node.id,
                        node.parent_id,
                        f"{node.score:.6f}",
                        node.state.model_temperature,
                        node.state.payload.change_summary or "",
                    ]
                )
        except Exception as e:
            log.warning(f"Lineage write failed: {e}")

    def load_resume_nodes(self) -> list[SearchNode[SvgStatePayload]]:
        """Scans filesystem to rebuild the search pool."""
        if not self.resume or not os.path.isdir(self.nodes_dir):
            return []

        nodes = []
        pattern = re.compile(r"^score([0-9.]+)_node(\d+)_parent(\d+)\.svg$")

        for fn in os.listdir(self.nodes_dir):
            match = pattern.match(fn)
            if not match:
                continue

            score, nid, pid = (
                float(match.group(1)),
                int(match.group(2)),
                int(match.group(3)),
            )
            self._max_id = max(self._max_id, nid)

            try:
                with open(os.path.join(self.nodes_dir, fn), encoding="utf-8") as f:
                    svg = f.read()

                # Hydrate preview for the LLM context
                png = rasterize_svg_to_png_bytes(
                    svg, out_w=self.img_dims[0], out_h=self.img_dims[1]
                )
                prev = make_preview_data_url(png, self.openai_image_long_side)

                nodes.append(
                    SearchNode(
                        score=score,
                        id=nid,
                        parent_id=pid,
                        state=ChainState(
                            score=score,
                            model_temperature=self.base_temp,
                            stale_hits=0,
                            payload=SvgStatePayload(svg, None, prev, None, None),
                        ),
                    )
                )
            except Exception as e:
                log.error(f"Failed to load resume node {fn}: {e}")

        return sorted(nodes, key=lambda n: n.id)

    def save_final_svg(self, svg_content: str) -> None:
        """Saves the best result to the primary output path."""
        with open(self.output_svg_path, "w", encoding="utf-8") as f:
            f.write(svg_content)

    def load_seed_svg(self, seed_path: str) -> str:
        """Utility for the pipeline to load user-provided starting points."""
        with open(seed_path, encoding="utf-8") as f:
            return f.read()
