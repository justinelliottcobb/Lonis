"""tests/test_gradient.py"""
import numpy as np
import pytest
from pathlib import Path
from analyzer.utils.image import load_image
from analyzer.analyzers.gradient import GradientAnalyzer

FIXTURES = Path(__file__).parent / "fixtures"

@pytest.fixture
def analyzer():
    return GradientAnalyzer()

def test_name(analyzer):
    assert analyzer.name == "gradient"

def test_horizontal_gradient_direction(analyzer):
    img = load_image(FIXTURES / "gradient_h.png")
    result = analyzer.analyze(img)
    # Horizontal gradient should be ~0 or ~180 degrees
    assert result["global_direction"] in range(-15, 15) or result["global_direction"] in range(165, 195)

def test_luminance_range(analyzer):
    img = load_image(FIXTURES / "gradient_h.png")
    result = analyzer.analyze(img)
    lum = result["luminance_range"]
    assert lum["min"] < 20
    assert lum["max"] > 230
    assert "mean" in lum

def test_solid_image_low_gradient(analyzer):
    img = load_image(FIXTURES / "solid_red.png")
    result = analyzer.analyze(img)
    assert len(result["transitions"]) == 0 or result["transitions"][0]["strength"] < 0.1

def test_output_keys(analyzer):
    img = load_image(FIXTURES / "gradient_h.png")
    result = analyzer.analyze(img)
    assert "global_direction" in result
    assert "luminance_range" in result
    assert "transitions" in result

def test_transition_fields(analyzer):
    img = load_image(FIXTURES / "gradient_h.png")
    result = analyzer.analyze(img)
    if result["transitions"]:
        t = result["transitions"][0]
        assert "direction" in t
        assert "from_color" in t
        assert "to_color" in t
        assert "strength" in t
