"""tests/test_spatial.py"""
import numpy as np
import pytest
from pathlib import Path
from analyzer.utils.image import load_image
from analyzer.analyzers.spatial import SpatialAnalyzer

FIXTURES = Path(__file__).parent / "fixtures"

@pytest.fixture
def analyzer():
    return SpatialAnalyzer()

def test_name(analyzer):
    assert analyzer.name == "spatial"

def test_grid_layout_detects_columns(analyzer):
    img = load_image(FIXTURES / "grid_layout.png")
    result = analyzer.analyze(img)
    grid = result["grid"]
    assert grid["columns"] == 3
    assert grid["rows"] == 2

def test_grid_layout_has_regions(analyzer):
    img = load_image(FIXTURES / "grid_layout.png")
    result = analyzer.analyze(img)
    regions = result["regions"]
    assert len(regions) >= 6

def test_region_has_required_fields(analyzer):
    img = load_image(FIXTURES / "grid_layout.png")
    result = analyzer.analyze(img)
    region = result["regions"][0]
    assert "id" in region
    assert "bounds" in region
    bounds = region["bounds"]
    assert all(k in bounds for k in ("x", "y", "w", "h"))
    assert "dominant_color" in region
    assert "area_percentage" in region

def test_output_has_required_keys(analyzer):
    img = load_image(FIXTURES / "solid_red.png")
    result = analyzer.analyze(img)
    assert "grid" in result
    assert "regions" in result
