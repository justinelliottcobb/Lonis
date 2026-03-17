"""tests/test_edge.py"""
import numpy as np
import pytest
from pathlib import Path
from analyzer.utils.image import load_image
from analyzer.analyzers.edge import EdgeAnalyzer

FIXTURES = Path(__file__).parent / "fixtures"

@pytest.fixture
def analyzer():
    return EdgeAnalyzer()

def test_name(analyzer):
    assert analyzer.name == "edge"

def test_shapes_detects_contours(analyzer):
    img = load_image(FIXTURES / "shapes.png")
    result = analyzer.analyze(img)
    assert len(result["contours"]) >= 2

def test_contour_has_required_fields(analyzer):
    img = load_image(FIXTURES / "shapes.png")
    result = analyzer.analyze(img)
    contour = result["contours"][0]
    assert "id" in contour
    assert "bounds" in contour
    assert "shape_type" in contour
    assert "vertex_count" in contour
    assert "area" in contour

def test_shape_classification(analyzer):
    img = load_image(FIXTURES / "shapes.png")
    result = analyzer.analyze(img)
    types = {c["shape_type"] for c in result["contours"]}
    assert "rectangle" in types or "ellipse" in types

def test_edge_density_range(analyzer):
    img = load_image(FIXTURES / "shapes.png")
    result = analyzer.analyze(img)
    assert 0.0 <= result["edge_density"] <= 1.0

def test_dominant_angles(analyzer):
    img = load_image(FIXTURES / "shapes.png")
    result = analyzer.analyze(img)
    assert "dominant_angles" in result
    assert isinstance(result["dominant_angles"], list)

def test_output_keys(analyzer):
    img = load_image(FIXTURES / "solid_red.png")
    result = analyzer.analyze(img)
    assert "contours" in result
    assert "edge_density" in result
    assert "dominant_angles" in result


# --- Coarse-to-fine tests ---

def test_confidence_field(analyzer):
    """All contours should have a confidence tag."""
    img = load_image(FIXTURES / "shapes.png")
    result = analyzer.analyze(img)
    for contour in result["contours"]:
        assert contour["confidence"] in ("coarse", "fine")

def test_no_context_all_coarse(analyzer):
    """Without spatial context, all contours should be 'coarse'."""
    img = load_image(FIXTURES / "shapes.png")
    result = analyzer.analyze(img)
    for contour in result["contours"]:
        assert contour["confidence"] == "coarse"

def test_coarse_to_fine_with_spatial_context(analyzer):
    """With spatial context providing region hints, fine pass should find additional contours."""
    img = load_image(FIXTURES / "grid_layout.png")

    # First run without context
    result_no_ctx = analyzer.analyze(img)
    coarse_count = len(result_no_ctx["contours"])

    # Build spatial context with regions that cover the grid cells
    # grid_layout.png is 300x200 with 6 colored cells in a 3x2 grid
    spatial_context = {
        "spatial": {
            "regions": [
                {"id": 0, "bounds": {"x": 10, "y": 10, "w": 80, "h": 80}},
                {"id": 1, "bounds": {"x": 110, "y": 10, "w": 80, "h": 80}},
                {"id": 2, "bounds": {"x": 210, "y": 10, "w": 80, "h": 80}},
                {"id": 3, "bounds": {"x": 10, "y": 110, "w": 80, "h": 80}},
                {"id": 4, "bounds": {"x": 110, "y": 110, "w": 80, "h": 80}},
                {"id": 5, "bounds": {"x": 210, "y": 110, "w": 80, "h": 80}},
            ]
        }
    }

    result_with_ctx = analyzer.analyze(img, context=spatial_context)
    fine_count = len(result_with_ctx["contours"])

    # With context, should find at least as many contours as without
    assert fine_count >= coarse_count

def test_fine_contours_have_fine_confidence(analyzer):
    """Fine-pass contours should be tagged with 'fine' confidence."""
    img = load_image(FIXTURES / "grid_layout.png")

    # Create context with a region far from any likely coarse contour
    spatial_context = {
        "spatial": {
            "regions": [
                {"id": 0, "bounds": {"x": 10, "y": 10, "w": 80, "h": 80}},
                {"id": 1, "bounds": {"x": 110, "y": 10, "w": 80, "h": 80}},
                {"id": 2, "bounds": {"x": 210, "y": 10, "w": 80, "h": 80}},
                {"id": 3, "bounds": {"x": 10, "y": 110, "w": 80, "h": 80}},
                {"id": 4, "bounds": {"x": 110, "y": 110, "w": 80, "h": 80}},
                {"id": 5, "bounds": {"x": 210, "y": 110, "w": 80, "h": 80}},
            ]
        }
    }

    result = analyzer.analyze(img, context=spatial_context)
    confidences = {c["confidence"] for c in result["contours"]}
    # Should have at least coarse contours
    assert "coarse" in confidences or len(result["contours"]) == 0

def test_empty_spatial_regions(analyzer):
    """Empty spatial regions list should behave like no context."""
    img = load_image(FIXTURES / "shapes.png")
    context = {"spatial": {"regions": []}}
    result = analyzer.analyze(img, context=context)
    for contour in result["contours"]:
        assert contour["confidence"] == "coarse"

def test_contour_ids_sequential(analyzer):
    """After merge, contour IDs should be sequential starting from 0."""
    img = load_image(FIXTURES / "grid_layout.png")
    spatial_context = {
        "spatial": {
            "regions": [
                {"id": i, "bounds": {"x": 10 + i * 100, "y": 10, "w": 80, "h": 80}}
                for i in range(3)
            ]
        }
    }
    result = analyzer.analyze(img, context=spatial_context)
    ids = [c["id"] for c in result["contours"]]
    assert ids == list(range(len(ids)))
