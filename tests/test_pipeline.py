"""tests/test_pipeline.py"""
import json
import pytest
from pathlib import Path
from analyzer.pipeline import Pipeline

FIXTURES = Path(__file__).parent / "fixtures"

def test_run_all_analyzers():
    pipe = Pipeline()
    result = pipe.run(str(FIXTURES / "grid_layout.png"))
    assert "metadata" in result
    assert "colors" in result
    assert "spatial" in result
    assert "edges" in result
    assert "gradients" in result
    assert "textures" in result

def test_metadata_fields():
    pipe = Pipeline()
    result = pipe.run(str(FIXTURES / "grid_layout.png"))
    meta = result["metadata"]
    assert meta["filename"] == "grid_layout.png"
    assert meta["dimensions"] == [300, 200]
    assert "analyzed_at" in meta
    assert "analyzers_run" in meta
    assert "duration_ms" in meta

def test_run_selected_analyzers():
    pipe = Pipeline(analyzers=["color", "edge"])
    result = pipe.run(str(FIXTURES / "grid_layout.png"))
    assert "colors" in result
    assert "edges" in result
    assert "spatial" not in result
    assert "gradients" not in result

def test_output_is_json_serializable():
    pipe = Pipeline()
    result = pipe.run(str(FIXTURES / "grid_layout.png"))
    serialized = json.dumps(result)
    assert isinstance(serialized, str)
    roundtrip = json.loads(serialized)
    assert roundtrip["metadata"]["filename"] == "grid_layout.png"

def test_invalid_analyzer_name():
    pipe = Pipeline(analyzers=["color", "nonexistent"])
    with pytest.raises(ValueError, match="Unknown analyzer"):
        pipe.run(str(FIXTURES / "grid_layout.png"))

def test_edge_gets_spatial_context():
    """When spatial runs before edge, edge contours should include confidence tags."""
    pipe = Pipeline()  # default order: color, spatial, edge, ...
    result = pipe.run(str(FIXTURES / "grid_layout.png"))
    for contour in result["edges"]["contours"]:
        assert "confidence" in contour
        assert contour["confidence"] in ("coarse", "fine")

def test_edge_without_spatial_still_works():
    """Edge should work fine when spatial doesn't run first."""
    pipe = Pipeline(analyzers=["edge"])
    result = pipe.run(str(FIXTURES / "grid_layout.png"))
    assert "contours" in result["edges"]
    for contour in result["edges"]["contours"]:
        assert contour["confidence"] == "coarse"
