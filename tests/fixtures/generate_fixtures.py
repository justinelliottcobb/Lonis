"""Generate small deterministic test images for analyzer tests."""
import numpy as np
from PIL import Image
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent

def generate_solid_red():
    """10x10 solid red image."""
    img = np.full((10, 10, 3), [255, 0, 0], dtype=np.uint8)
    Image.fromarray(img).save(FIXTURES_DIR / "solid_red.png")

def generate_two_color():
    """20x10 image: left half red, right half blue."""
    img = np.zeros((10, 20, 3), dtype=np.uint8)
    img[:, :10] = [255, 0, 0]
    img[:, 10:] = [0, 0, 255]
    Image.fromarray(img).save(FIXTURES_DIR / "two_color.png")

def generate_gradient():
    """100x100 horizontal black-to-white gradient."""
    grad = np.tile(np.linspace(0, 255, 100, dtype=np.uint8), (100, 1))
    img = np.stack([grad, grad, grad], axis=2)
    Image.fromarray(img).save(FIXTURES_DIR / "gradient_h.png")

def generate_shapes():
    """200x200 white bg with a black rectangle and black circle."""
    img = np.full((200, 200, 3), 255, dtype=np.uint8)
    # Rectangle: top-left (20,20) to (80,60)
    img[20:60, 20:80] = [0, 0, 0]
    # Circle: center (140,100), radius 30
    y, x = np.ogrid[:200, :200]
    circle_mask = ((x - 140)**2 + (y - 100)**2) <= 30**2
    img[circle_mask] = [0, 0, 0]
    Image.fromarray(img).save(FIXTURES_DIR / "shapes.png")

def generate_textured():
    """100x100 image with checkerboard pattern (textured)."""
    block = 10
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    for i in range(0, 100, block):
        for j in range(0, 100, block):
            if (i // block + j // block) % 2 == 0:
                img[i:i+block, j:j+block] = [200, 200, 200]
            else:
                img[i:i+block, j:j+block] = [50, 50, 50]
    Image.fromarray(img).save(FIXTURES_DIR / "checkerboard.png")

def generate_grid_layout():
    """300x200 dark bg with 3x2 grid of colored rectangles."""
    img = np.full((200, 300, 3), 20, dtype=np.uint8)
    colors = [
        [200, 120, 50], [50, 170, 130], [190, 50, 100],
        [60, 60, 180], [200, 180, 40], [100, 200, 100]
    ]
    for idx, (row, col) in enumerate([(r, c) for r in range(2) for c in range(3)]):
        x0 = 15 + col * 95
        y0 = 15 + row * 95
        img[y0:y0+75, x0:x0+75] = colors[idx]
    Image.fromarray(img).save(FIXTURES_DIR / "grid_layout.png")

def generate_all():
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    generate_solid_red()
    generate_two_color()
    generate_gradient()
    generate_shapes()
    generate_textured()
    generate_grid_layout()
    print(f"Generated fixtures in {FIXTURES_DIR}")

if __name__ == "__main__":
    generate_all()
