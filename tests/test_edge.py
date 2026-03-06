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
