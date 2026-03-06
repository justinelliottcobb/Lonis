"""tests/test_texture.py"""
import numpy as np
import pytest
from pathlib import Path
from analyzer.utils.image import load_image
from analyzer.analyzers.texture import TextureAnalyzer

FIXTURES = Path(__file__).parent / "fixtures"

@pytest.fixture
def analyzer():
    return TextureAnalyzer()

def test_name(analyzer):
    assert analyzer.name == "texture"

def test_solid_is_smooth(analyzer):
    img = load_image(FIXTURES / "solid_red.png")
    result = analyzer.analyze(img)
    assert result["global"]["roughness"] < 0.1
    assert result["global"]["uniformity"] > 0.9

def test_checkerboard_is_rough(analyzer):
    img = load_image(FIXTURES / "checkerboard.png")
    result = analyzer.analyze(img)
    assert result["global"]["roughness"] > 0.3

def test_output_keys(analyzer):
    img = load_image(FIXTURES / "solid_red.png")
    result = analyzer.analyze(img)
    assert "global" in result
    assert "roughness" in result["global"]
    assert "glossiness" in result["global"]
    assert "uniformity" in result["global"]

def test_values_in_range(analyzer):
    img = load_image(FIXTURES / "checkerboard.png")
    result = analyzer.analyze(img)
    g = result["global"]
    assert 0.0 <= g["roughness"] <= 1.0
    assert 0.0 <= g["glossiness"] <= 1.0
    assert 0.0 <= g["uniformity"] <= 1.0
