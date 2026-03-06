"""tests/test_utils.py"""
import numpy as np
import pytest
from pathlib import Path
from analyzer.utils.image import load_image

FIXTURES = Path(__file__).parent / "fixtures"

def test_load_image_returns_rgb_array():
    img = load_image(FIXTURES / "solid_red.png")
    assert isinstance(img, np.ndarray)
    assert img.dtype == np.uint8
    assert img.shape == (10, 10, 3)

def test_load_image_correct_values():
    img = load_image(FIXTURES / "solid_red.png")
    assert np.all(img[:, :, 0] == 255)  # R
    assert np.all(img[:, :, 1] == 0)    # G
    assert np.all(img[:, :, 2] == 0)    # B

def test_load_image_nonexistent_raises():
    with pytest.raises(FileNotFoundError):
        load_image(Path("/nonexistent/image.png"))

def test_load_image_string_path():
    img = load_image(str(FIXTURES / "solid_red.png"))
    assert img.shape == (10, 10, 3)
