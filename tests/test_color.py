"""tests/test_color.py"""
import numpy as np
import pytest
from pathlib import Path
from analyzer.utils.image import load_image
from analyzer.analyzers.color import ColorAnalyzer

FIXTURES = Path(__file__).parent / "fixtures"

@pytest.fixture
def analyzer():
    return ColorAnalyzer()

def test_name(analyzer):
    assert analyzer.name == "color"

def test_solid_red_dominant(analyzer):
    img = load_image(FIXTURES / "solid_red.png")
    result = analyzer.analyze(img)
    dominant = result["dominant"]
    assert len(dominant) >= 1
    assert dominant[0]["hex"] == "#ff0000"
    assert dominant[0]["percentage"] == pytest.approx(100.0, abs=1.0)

def test_two_color_palette(analyzer):
    img = load_image(FIXTURES / "two_color.png")
    result = analyzer.analyze(img)
    hexes = {c["hex"] for c in result["dominant"]}
    assert "#ff0000" in hexes
    assert "#0000ff" in hexes

def test_two_color_percentages(analyzer):
    img = load_image(FIXTURES / "two_color.png")
    result = analyzer.analyze(img)
    for c in result["dominant"]:
        assert c["percentage"] == pytest.approx(50.0, abs=5.0)

def test_output_has_required_keys(analyzer):
    img = load_image(FIXTURES / "solid_red.png")
    result = analyzer.analyze(img)
    assert "dominant" in result
    assert "palette" in result
    assert "harmony" in result

def test_dominant_color_fields(analyzer):
    img = load_image(FIXTURES / "solid_red.png")
    result = analyzer.analyze(img)
    color = result["dominant"][0]
    assert "hex" in color
    assert "rgb" in color
    assert "hsl" in color
    assert "percentage" in color
    assert len(color["rgb"]) == 3
    assert len(color["hsl"]) == 3

def test_harmony_fields(analyzer):
    img = load_image(FIXTURES / "two_color.png")
    result = analyzer.analyze(img)
    harmony = result["harmony"]
    assert "temperature" in harmony
    assert "contrast_ratio" in harmony

def test_with_mask(analyzer):
    img = load_image(FIXTURES / "two_color.png")
    # Mask: only left half (red pixels)
    mask = np.zeros((10, 20), dtype=bool)
    mask[:, :10] = True
    result = analyzer.analyze(img, mask=mask)
    assert result["dominant"][0]["hex"] == "#ff0000"
