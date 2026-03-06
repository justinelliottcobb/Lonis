"""analyzer/utils/image.py"""
from pathlib import Path
import numpy as np
from PIL import Image

def load_image(path: str | Path) -> np.ndarray:
    """Load an image file and return as RGB uint8 numpy array.

    Args:
        path: Path to image file (PNG, JPG, WEBP, etc.)

    Returns:
        numpy array of shape (H, W, 3), dtype uint8, RGB color space.

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If file cannot be decoded as an image.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    try:
        img = Image.open(path).convert("RGB")
        return np.array(img, dtype=np.uint8)
    except Exception as e:
        raise ValueError(f"Cannot decode image: {path} — {e}") from e

def get_image_metadata(path: str | Path) -> dict:
    """Return basic metadata about an image file."""
    path = Path(path)
    img = Image.open(path)
    return {
        "filename": path.name,
        "dimensions": [img.width, img.height],
        "format": img.format or path.suffix.lstrip(".").upper(),
    }
