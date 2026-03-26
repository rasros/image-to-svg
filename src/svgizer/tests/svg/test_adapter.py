from svgizer.svg.adapter import SvgResultPayload, SvgStatePayload


def create_payload(svg_text: str | None) -> SvgStatePayload:
    return SvgStatePayload(
        svg=svg_text,
        raster_data_url=None,
        raster_preview_data_url=None,
        change_summary=None,
        invalid_msg=None,
    )


def create_result(svg_text: str | None) -> SvgResultPayload:
    return SvgResultPayload(svg=svg_text, raster_png=None, change_summary=None)


def test_payload_creation():
    payload = create_payload("<svg></svg>")
    assert payload.svg == "<svg></svg>"
