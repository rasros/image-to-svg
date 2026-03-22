import pytest
from unittest.mock import MagicMock, patch
from svgizer.search.run import run_search

@patch("svgizer.search.run.MultiprocessSearchEngine")
@patch("svgizer.search.run.get_scorer")
@patch("svgizer.search.run.Image.open")
def test_run_search_initializes_correctly(mock_img, mock_get_scorer, mock_engine_cls):
    storage = MagicMock()
    storage.load_resume_nodes.return_value = ([], None, 0)
    storage.write_lineage_enabled = False

    # Mock engine behavior
    mock_engine = mock_engine_cls.return_value
    mock_engine.run.return_value = MagicMock()

    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
        run_search(
            image_path="dummy.png",
            storage=storage,
            seed_svg_path=None,
            max_accepts=1,
            workers=1,
            base_model_temperature=0.6,
            openai_image_long_side=512,
            max_wall_seconds=None,
            log_level="INFO",
            scorer_type="simple"
        )

    # Verify storage was initialized and engine was started
    assert storage.initialize.called
    assert mock_engine.start_workers.called
    assert mock_engine.run.called