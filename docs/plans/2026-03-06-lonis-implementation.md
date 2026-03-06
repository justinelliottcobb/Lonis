# Lonis v1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a CLI tool that analyzes bitmap images and outputs precise, structured design data (colors, spatial layout, edges, gradients, textures) as JSON.

**Architecture:** Modular Python pipeline — each analyzer is an independent module inheriting from BaseAnalyzer, orchestrated by a Pipeline class, invoked via CLI. Each analyzer takes an RGB numpy array + optional mask and returns a dict.

**Tech Stack:** Python 3.12, Pillow, opencv-python-headless, numpy, scikit-learn, scipy, pytest

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `analyzer/__init__.py`
- Create: `analyzer/analyzers/__init__.py`
- Create: `analyzer/utils/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/fixtures/generate_fixtures.py`

**Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "lonis"
version = "0.1.0"
description = "Bitmap-to-structured-design-data analyzer"
requires-python = ">=3.11"
dependencies = [
    "Pillow>=10.0",
    "opencv-python-headless>=4.8",
    "numpy>=1.24",
    "scikit-learn>=1.3",
    "scipy>=1.11",
]

[project.scripts]
lonis = "analyzer.cli:main"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

**Step 2: Create requirements.txt**

```
Pillow>=10.0
opencv-python-headless>=4.8
numpy>=1.24
scikit-learn>=1.3
scipy>=1.11
pytest>=7.0
```

**Step 3: Create empty __init__.py files**

Create empty `analyzer/__init__.py`, `analyzer/analyzers/__init__.py`, `analyzer/utils/__init__.py`, `tests/__init__.py`.

**Step 4: Create test fixture generator**

```python
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
```

**Step 5: Install dependencies and generate fixtures**

Run: `cd /home/sisawat/working/design/lonis && pip install -e ".[dev]" 2>/dev/null; pip install -r requirements.txt`
Run: `python3 tests/fixtures/generate_fixtures.py`

**Step 6: Verify setup**

Run: `python3 -c "from analyzer import __init__; print('OK')"`
Expected: `OK`

**Step 7: Commit**

```bash
git add -A
git commit -m "feat: project scaffolding with dependencies and test fixtures"
```

---

### Task 2: Image Utils & BaseAnalyzer

**Files:**
- Create: `analyzer/utils/image.py`
- Create: `analyzer/analyzers/base.py`
- Create: `tests/test_utils.py`

**Step 1: Write failing test for image loading**

```python
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
```

**Step 2: Run test to verify it fails**

Run: `cd /home/sisawat/working/design/lonis && python3 -m pytest tests/test_utils.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'analyzer.utils.image'`

**Step 3: Implement image utils**

```python
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
```

**Step 4: Implement BaseAnalyzer**

```python
"""analyzer/analyzers/base.py"""
from abc import ABC, abstractmethod
import numpy as np

class BaseAnalyzer(ABC):
    """Abstract base class for all Lonis analyzers.

    The mask parameter enables future region-based analysis (Phase 2)
    without changing the analyzer interface.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this analyzer, e.g. 'color'."""

    @abstractmethod
    def analyze(self, image: np.ndarray, mask: np.ndarray | None = None) -> dict:
        """Run analysis on an image.

        Args:
            image: RGB numpy array, uint8, shape (H, W, 3).
            mask:  Optional boolean mask, shape (H, W). True = include pixel.

        Returns:
            Dict of analysis results for this analyzer's domain.
        """

    def _apply_mask(self, image: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
        """Return only the masked pixels as a (N, 3) array.

        If mask is None, returns all pixels flattened.
        """
        if mask is not None:
            return image[mask]
        return image.reshape(-1, 3)
```

**Step 5: Run tests to verify they pass**

Run: `cd /home/sisawat/working/design/lonis && python3 -m pytest tests/test_utils.py -v`
Expected: 4 passed

**Step 6: Commit**

```bash
git add analyzer/utils/image.py analyzer/analyzers/base.py tests/test_utils.py
git commit -m "feat: image loading utils and BaseAnalyzer abstract class"
```

---

### Task 3: ColorAnalyzer

**Files:**
- Create: `analyzer/analyzers/color.py`
- Create: `tests/test_color.py`

**Step 1: Write failing tests**

```python
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
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/sisawat/working/design/lonis && python3 -m pytest tests/test_color.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement ColorAnalyzer**

```python
"""analyzer/analyzers/color.py"""
import colorsys
import numpy as np
from sklearn.cluster import KMeans
from .base import BaseAnalyzer

class ColorAnalyzer(BaseAnalyzer):
    """Extracts dominant colors, palette, and color harmony from an image."""

    def __init__(self, max_colors: int = 8, sample_limit: int = 10000):
        self.max_colors = max_colors
        self.sample_limit = sample_limit

    @property
    def name(self) -> str:
        return "color"

    def analyze(self, image: np.ndarray, mask: np.ndarray | None = None) -> dict:
        pixels = self._apply_mask(image, mask)

        # Subsample for performance if needed
        if len(pixels) > self.sample_limit:
            indices = np.random.default_rng(42).choice(len(pixels), self.sample_limit, replace=False)
            sampled = pixels[indices]
        else:
            sampled = pixels

        # Determine number of clusters (don't exceed unique colors)
        unique_colors = np.unique(sampled.reshape(-1, 3), axis=0)
        n_clusters = min(self.max_colors, len(unique_colors))

        # KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(sampled)

        # Build palette sorted by frequency
        labels, counts = np.unique(kmeans.labels_, return_counts=True)
        total = counts.sum()
        centers = kmeans.cluster_centers_.astype(int)

        palette = []
        for idx in np.argsort(-counts):
            r, g, b = centers[idx]
            palette.append({
                "hex": f"#{r:02x}{g:02x}{b:02x}",
                "rgb": [int(r), int(g), int(b)],
                "hsl": list(self._rgb_to_hsl(r, g, b)),
                "percentage": round(float(counts[idx]) / total * 100, 1),
            })

        harmony = self._analyze_harmony(palette)

        return {
            "dominant": palette,
            "palette": [c["hex"] for c in palette],
            "harmony": harmony,
        }

    @staticmethod
    def _rgb_to_hsl(r: int, g: int, b: int) -> tuple[int, int, int]:
        h, l, s = colorsys.rgb_to_hls(r / 255.0, g / 255.0, b / 255.0)
        return round(h * 360), round(s * 100), round(l * 100)

    @staticmethod
    def _analyze_harmony(palette: list[dict]) -> dict:
        if not palette:
            return {"temperature": "neutral", "contrast_ratio": 0.0}

        # Temperature: based on hue distribution
        warm_weight = 0.0
        cool_weight = 0.0
        for c in palette:
            h = c["hsl"][0]
            pct = c["percentage"]
            # Warm: roughly 0-60 and 300-360 (reds, oranges, yellows)
            # Cool: roughly 120-270 (greens, blues, purples)
            if h < 60 or h > 300:
                warm_weight += pct
            elif 120 <= h <= 270:
                cool_weight += pct
            else:
                warm_weight += pct * 0.5
                cool_weight += pct * 0.5

        if warm_weight > cool_weight * 1.5:
            temperature = "warm-dominant"
        elif cool_weight > warm_weight * 1.5:
            temperature = "cool-dominant"
        else:
            temperature = "balanced"

        # Contrast: luminance ratio between lightest and darkest
        luminances = [c["hsl"][2] for c in palette]
        lightest = max(luminances) / 100.0 + 0.05
        darkest = min(luminances) / 100.0 + 0.05
        contrast = round(lightest / darkest, 1) if darkest > 0 else 0.0

        return {
            "temperature": temperature,
            "contrast_ratio": contrast,
        }
```

**Step 4: Run tests to verify they pass**

Run: `cd /home/sisawat/working/design/lonis && python3 -m pytest tests/test_color.py -v`
Expected: 8 passed

**Step 5: Commit**

```bash
git add analyzer/analyzers/color.py tests/test_color.py
git commit -m "feat: ColorAnalyzer with KMeans palette extraction and harmony analysis"
```

---

### Task 4: SpatialAnalyzer

**Files:**
- Create: `analyzer/analyzers/spatial.py`
- Create: `tests/test_spatial.py`

**Step 1: Write failing tests**

```python
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
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/sisawat/working/design/lonis && python3 -m pytest tests/test_spatial.py -v`
Expected: FAIL

**Step 3: Implement SpatialAnalyzer**

```python
"""analyzer/analyzers/spatial.py"""
import cv2
import numpy as np
from .base import BaseAnalyzer

class SpatialAnalyzer(BaseAnalyzer):
    """Detects spatial layout grid and distinct visual regions."""

    def __init__(self, min_region_area_pct: float = 0.5):
        self.min_region_area_pct = min_region_area_pct

    @property
    def name(self) -> str:
        return "spatial"

    def analyze(self, image: np.ndarray, mask: np.ndarray | None = None) -> dict:
        h, w = image.shape[:2]
        total_area = h * w

        regions = self._detect_regions(image, mask)
        grid = self._detect_grid(regions, w, h)

        return {
            "grid": grid,
            "regions": [
                {
                    "id": i,
                    "bounds": {"x": r[0], "y": r[1], "w": r[2], "h": r[3]},
                    "dominant_color": r[4],
                    "area_percentage": round(r[2] * r[3] / total_area * 100, 1),
                }
                for i, r in enumerate(regions)
                if (r[2] * r[3]) / total_area * 100 >= self.min_region_area_pct
            ],
        }

    def _detect_regions(self, image: np.ndarray, mask: np.ndarray | None) -> list:
        """Find distinct color regions using contour detection on edges."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Adaptive threshold to find distinct areas
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions = []
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            if mask is not None and not mask[y:y+ch, x:x+cw].any():
                continue
            # Dominant color in this region
            region_pixels = image[y:y+ch, x:x+cw].reshape(-1, 3)
            avg = region_pixels.mean(axis=0).astype(int)
            hex_color = f"#{avg[0]:02x}{avg[1]:02x}{avg[2]:02x}"
            regions.append((int(x), int(y), int(cw), int(ch), hex_color))

        # Sort by position: top-to-bottom, left-to-right
        regions.sort(key=lambda r: (r[1], r[0]))
        return regions

    def _detect_grid(self, regions: list, img_w: int, img_h: int) -> dict:
        """Attempt to detect a grid pattern from region positions."""
        if len(regions) < 2:
            return {"columns": 1, "rows": 1, "cell_size": [img_w, img_h], "gutters": [0, 0]}

        xs = sorted(set(r[0] for r in regions))
        ys = sorted(set(r[1] for r in regions))

        # Cluster x positions to find columns
        cols = self._cluster_positions(xs, threshold=img_w * 0.05)
        rows = self._cluster_positions(ys, threshold=img_h * 0.05)

        n_cols = len(cols)
        n_rows = len(rows)

        # Estimate cell size from median region dimensions
        widths = [r[2] for r in regions]
        heights = [r[3] for r in regions]
        cell_w = int(np.median(widths)) if widths else img_w
        cell_h = int(np.median(heights)) if heights else img_h

        # Estimate gutters
        if n_cols > 1:
            col_positions = sorted(cols)
            gutter_x = int(np.median(np.diff(col_positions)) - cell_w)
            gutter_x = max(0, gutter_x)
        else:
            gutter_x = 0

        if n_rows > 1:
            row_positions = sorted(rows)
            gutter_y = int(np.median(np.diff(row_positions)) - cell_h)
            gutter_y = max(0, gutter_y)
        else:
            gutter_y = 0

        return {
            "columns": n_cols,
            "rows": n_rows,
            "cell_size": [cell_w, cell_h],
            "gutters": [gutter_x, gutter_y],
        }

    @staticmethod
    def _cluster_positions(positions: list[int], threshold: float) -> list[int]:
        """Group nearby positions into clusters, return cluster centers."""
        if not positions:
            return []
        clusters = [[positions[0]]]
        for p in positions[1:]:
            if p - clusters[-1][-1] <= threshold:
                clusters[-1].append(p)
            else:
                clusters.append([p])
        return [int(np.mean(c)) for c in clusters]
```

**Step 4: Run tests to verify they pass**

Run: `cd /home/sisawat/working/design/lonis && python3 -m pytest tests/test_spatial.py -v`
Expected: 5 passed

**Step 5: Commit**

```bash
git add analyzer/analyzers/spatial.py tests/test_spatial.py
git commit -m "feat: SpatialAnalyzer with grid detection and region extraction"
```

---

### Task 5: EdgeAnalyzer

**Files:**
- Create: `analyzer/analyzers/edge.py`
- Create: `tests/test_edge.py`

**Step 1: Write failing tests**

```python
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
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/sisawat/working/design/lonis && python3 -m pytest tests/test_edge.py -v`
Expected: FAIL

**Step 3: Implement EdgeAnalyzer**

```python
"""analyzer/analyzers/edge.py"""
import cv2
import numpy as np
from .base import BaseAnalyzer

class EdgeAnalyzer(BaseAnalyzer):
    """Detects contours, classifies shapes, and measures edge characteristics."""

    @property
    def name(self) -> str:
        return "edge"

    def analyze(self, image: np.ndarray, mask: np.ndarray | None = None) -> dict:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        if mask is not None:
            gray = cv2.bitwise_and(gray, gray, mask=mask.astype(np.uint8) * 255)

        edges = cv2.Canny(gray, 50, 150)
        contours_raw, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        h, w = gray.shape
        min_area = h * w * 0.001  # Ignore tiny contours

        contours = []
        for i, cnt in enumerate(contours_raw):
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            x, y, cw, ch = cv2.boundingRect(cnt)
            shape_type = self._classify_shape(cnt)
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            contours.append({
                "id": len(contours),
                "bounds": {"x": int(x), "y": int(y), "w": int(cw), "h": int(ch)},
                "shape_type": shape_type,
                "vertex_count": len(approx),
                "area": int(area),
            })

        # Edge density: fraction of pixels that are edges
        edge_density = round(float(np.count_nonzero(edges)) / (h * w), 3)

        # Dominant edge angles via Hough lines
        dominant_angles = self._detect_dominant_angles(edges)

        return {
            "contours": contours,
            "edge_density": edge_density,
            "dominant_angles": dominant_angles,
        }

    @staticmethod
    def _classify_shape(contour: np.ndarray) -> str:
        """Classify a contour as rectangle, ellipse, triangle, or irregular."""
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
        vertices = len(approx)

        if vertices == 3:
            return "triangle"
        elif vertices == 4:
            # Check if approximately rectangular
            x, y, w, h = cv2.boundingRect(approx)
            aspect = w / h if h > 0 else 0
            area_ratio = cv2.contourArea(approx) / (w * h) if w * h > 0 else 0
            if 0.85 < area_ratio:
                return "rectangle"
            return "quadrilateral"
        elif vertices > 6:
            # Check circularity
            area = cv2.contourArea(contour)
            circularity = 4 * np.pi * area / (peri * peri) if peri > 0 else 0
            if circularity > 0.7:
                return "ellipse"
        return "irregular"

    @staticmethod
    def _detect_dominant_angles(edges: np.ndarray) -> list[int]:
        """Detect dominant edge angles using Hough Line Transform."""
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=20, maxLineGap=10)
        if lines is None:
            return []

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = int(np.degrees(np.arctan2(y2 - y1, x2 - x1))) % 180
            angles.append(angle)

        if not angles:
            return []

        # Cluster angles into bins of 15 degrees
        bins = np.arange(0, 195, 15)
        hist, _ = np.histogram(angles, bins=bins)
        # Return top 3 angle bins with significant counts
        threshold = max(hist) * 0.3
        dominant = [int(bins[i] + 7) for i in np.argsort(-hist)[:3] if hist[np.argsort(-hist)][list(np.argsort(-hist)).index(i)] >= threshold]
        return sorted(set(dominant))
```

**Step 4: Run tests**

Run: `cd /home/sisawat/working/design/lonis && python3 -m pytest tests/test_edge.py -v`
Expected: 7 passed

**Step 5: Commit**

```bash
git add analyzer/analyzers/edge.py tests/test_edge.py
git commit -m "feat: EdgeAnalyzer with contour detection, shape classification, and angle analysis"
```

---

### Task 6: GradientAnalyzer

**Files:**
- Create: `analyzer/analyzers/gradient.py`
- Create: `tests/test_gradient.py`

**Step 1: Write failing tests**

```python
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
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/sisawat/working/design/lonis && python3 -m pytest tests/test_gradient.py -v`
Expected: FAIL

**Step 3: Implement GradientAnalyzer**

```python
"""analyzer/analyzers/gradient.py"""
import cv2
import numpy as np
from .base import BaseAnalyzer

class GradientAnalyzer(BaseAnalyzer):
    """Analyzes gradient directions, color transitions, and luminance distribution."""

    @property
    def name(self) -> str:
        return "gradient"

    def analyze(self, image: np.ndarray, mask: np.ndarray | None = None) -> dict:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)

        if mask is not None:
            gray = gray * mask.astype(np.float32)

        h, w = gray.shape

        # Sobel gradients
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = np.sqrt(gx**2 + gy**2)
        angle = np.arctan2(gy, gx)

        # Global direction: weighted average of gradient angles
        weights = magnitude.flatten()
        if weights.sum() > 0:
            angles_flat = np.degrees(angle.flatten())
            global_dir = int(np.average(angles_flat, weights=weights)) % 360
        else:
            global_dir = 0

        # Luminance range
        valid_pixels = gray[mask] if mask is not None else gray.flatten()
        lum_range = {
            "min": int(valid_pixels.min()),
            "max": int(valid_pixels.max()),
            "mean": int(valid_pixels.mean()),
        }

        # Detect major transitions
        transitions = self._detect_transitions(image, gray, magnitude, h, w)

        return {
            "global_direction": global_dir,
            "luminance_range": lum_range,
            "transitions": transitions,
        }

    def _detect_transitions(self, image: np.ndarray, gray: np.ndarray,
                            magnitude: np.ndarray, h: int, w: int) -> list[dict]:
        """Detect major color transitions across the image."""
        transitions = []

        # Check horizontal transition (left-to-right)
        left_strip = image[:, :w//4].reshape(-1, 3).mean(axis=0).astype(int)
        right_strip = image[:, 3*w//4:].reshape(-1, 3).mean(axis=0).astype(int)
        h_diff = float(np.linalg.norm(left_strip.astype(float) - right_strip.astype(float))) / 441.67  # max RGB distance

        if h_diff > 0.05:
            transitions.append({
                "region": [0, 0, w, h],
                "direction": "left-to-right",
                "from_color": f"#{left_strip[0]:02x}{left_strip[1]:02x}{left_strip[2]:02x}",
                "to_color": f"#{right_strip[0]:02x}{right_strip[1]:02x}{right_strip[2]:02x}",
                "strength": round(h_diff, 2),
            })

        # Check vertical transition (top-to-bottom)
        top_strip = image[:h//4, :].reshape(-1, 3).mean(axis=0).astype(int)
        bot_strip = image[3*h//4:, :].reshape(-1, 3).mean(axis=0).astype(int)
        v_diff = float(np.linalg.norm(top_strip.astype(float) - bot_strip.astype(float))) / 441.67

        if v_diff > 0.05:
            transitions.append({
                "region": [0, 0, w, h],
                "direction": "top-to-bottom",
                "from_color": f"#{top_strip[0]:02x}{top_strip[1]:02x}{top_strip[2]:02x}",
                "to_color": f"#{bot_strip[0]:02x}{bot_strip[1]:02x}{bot_strip[2]:02x}",
                "strength": round(v_diff, 2),
            })

        return transitions
```

**Step 4: Run tests**

Run: `cd /home/sisawat/working/design/lonis && python3 -m pytest tests/test_gradient.py -v`
Expected: 6 passed

**Step 5: Commit**

```bash
git add analyzer/analyzers/gradient.py tests/test_gradient.py
git commit -m "feat: GradientAnalyzer with direction detection, luminance mapping, and transitions"
```

---

### Task 7: TextureAnalyzer

**Files:**
- Create: `analyzer/analyzers/texture.py`
- Create: `tests/test_texture.py`

**Step 1: Write failing tests**

```python
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
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/sisawat/working/design/lonis && python3 -m pytest tests/test_texture.py -v`
Expected: FAIL

**Step 3: Implement TextureAnalyzer**

```python
"""analyzer/analyzers/texture.py"""
import cv2
import numpy as np
from scipy import ndimage
from .base import BaseAnalyzer

class TextureAnalyzer(BaseAnalyzer):
    """Classifies surface texture properties: roughness, glossiness, uniformity."""

    @property
    def name(self) -> str:
        return "texture"

    def analyze(self, image: np.ndarray, mask: np.ndarray | None = None) -> dict:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

        if mask is not None:
            gray = gray * mask.astype(np.float32)
            valid = mask
        else:
            valid = np.ones(gray.shape, dtype=bool)

        global_tex = self._compute_texture_metrics(gray, valid)
        return {"global": global_tex}

    def _compute_texture_metrics(self, gray: np.ndarray, valid: np.ndarray) -> dict:
        """Compute roughness, glossiness, and uniformity metrics."""
        valid_pixels = gray[valid]
        if len(valid_pixels) == 0:
            return {"roughness": 0.0, "glossiness": 0.0, "uniformity": 1.0}

        # Roughness: based on local variance (Laplacian response)
        laplacian = ndimage.laplace(gray)
        lap_values = np.abs(laplacian[valid])
        roughness = float(np.clip(lap_values.mean() / 0.3, 0.0, 1.0))

        # Uniformity: inverse of coefficient of variation
        mean_val = valid_pixels.mean()
        if mean_val > 0.001:
            cv = valid_pixels.std() / mean_val
            uniformity = float(np.clip(1.0 - cv, 0.0, 1.0))
        else:
            uniformity = 1.0

        # Glossiness: based on high-value peaks (specular highlights)
        high_threshold = 0.85
        high_ratio = float((valid_pixels > high_threshold).sum()) / len(valid_pixels)
        # Also consider local contrast
        local_std = ndimage.uniform_filter(gray**2, size=5) - ndimage.uniform_filter(gray, size=5)**2
        local_std = np.sqrt(np.clip(local_std, 0, None))
        contrast = float(local_std[valid].mean())
        glossiness = float(np.clip(high_ratio * 2 + contrast, 0.0, 1.0))

        return {
            "roughness": round(roughness, 2),
            "glossiness": round(glossiness, 2),
            "uniformity": round(uniformity, 2),
        }
```

**Step 4: Run tests**

Run: `cd /home/sisawat/working/design/lonis && python3 -m pytest tests/test_texture.py -v`
Expected: 5 passed

**Step 5: Commit**

```bash
git add analyzer/analyzers/texture.py tests/test_texture.py
git commit -m "feat: TextureAnalyzer with roughness, glossiness, and uniformity metrics"
```

---

### Task 8: Pipeline Orchestrator

**Files:**
- Create: `analyzer/pipeline.py`
- Create: `tests/test_pipeline.py`

**Step 1: Write failing tests**

```python
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
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/sisawat/working/design/lonis && python3 -m pytest tests/test_pipeline.py -v`
Expected: FAIL

**Step 3: Implement Pipeline**

```python
"""analyzer/pipeline.py"""
import time
from datetime import datetime, timezone
from pathlib import Path

from .utils.image import load_image, get_image_metadata
from .analyzers.color import ColorAnalyzer
from .analyzers.spatial import SpatialAnalyzer
from .analyzers.edge import EdgeAnalyzer
from .analyzers.gradient import GradientAnalyzer
from .analyzers.texture import TextureAnalyzer

ANALYZER_REGISTRY = {
    "color": ColorAnalyzer,
    "spatial": SpatialAnalyzer,
    "edge": EdgeAnalyzer,
    "gradient": GradientAnalyzer,
    "texture": TextureAnalyzer,
}

# Map analyzer names to output keys
OUTPUT_KEYS = {
    "color": "colors",
    "spatial": "spatial",
    "edge": "edges",
    "gradient": "gradients",
    "texture": "textures",
}

class Pipeline:
    """Orchestrates running analyzers on an image and combining results."""

    def __init__(self, analyzers: list[str] | None = None):
        if analyzers is None:
            self.analyzer_names = list(ANALYZER_REGISTRY.keys())
        else:
            for name in analyzers:
                if name not in ANALYZER_REGISTRY:
                    raise ValueError(f"Unknown analyzer: '{name}'. Available: {list(ANALYZER_REGISTRY.keys())}")
            self.analyzer_names = analyzers

    def run(self, image_path: str) -> dict:
        """Load image, run selected analyzers, return combined result dict."""
        start = time.monotonic()

        path = Path(image_path)
        image = load_image(path)
        metadata = get_image_metadata(path)
        metadata["analyzed_at"] = datetime.now(timezone.utc).isoformat()
        metadata["analyzers_run"] = self.analyzer_names

        result = {"metadata": metadata}

        for name in self.analyzer_names:
            analyzer = ANALYZER_REGISTRY[name]()
            output_key = OUTPUT_KEYS[name]
            result[output_key] = analyzer.analyze(image)

        elapsed = time.monotonic() - start
        result["metadata"]["duration_ms"] = round(elapsed * 1000)

        return result
```

**Step 4: Run tests**

Run: `cd /home/sisawat/working/design/lonis && python3 -m pytest tests/test_pipeline.py -v`
Expected: 5 passed

**Step 5: Commit**

```bash
git add analyzer/pipeline.py tests/test_pipeline.py
git commit -m "feat: Pipeline orchestrator with analyzer registry and metadata"
```

---

### Task 9: CLI Entry Point

**Files:**
- Create: `analyzer/cli.py`
- Create: `tests/test_cli.py`

**Step 1: Write failing tests**

```python
"""tests/test_cli.py"""
import json
import subprocess
import sys
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"
CLI_MODULE = [sys.executable, "-m", "analyzer.cli"]

def test_analyze_outputs_json():
    result = subprocess.run(
        [*CLI_MODULE, "analyze", str(FIXTURES / "grid_layout.png")],
        capture_output=True, text=True, cwd=str(Path(__file__).parent.parent)
    )
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert "metadata" in data
    assert "colors" in data

def test_analyze_with_output_file(tmp_path):
    outfile = tmp_path / "result.json"
    result = subprocess.run(
        [*CLI_MODULE, "analyze", str(FIXTURES / "grid_layout.png"), "-o", str(outfile)],
        capture_output=True, text=True, cwd=str(Path(__file__).parent.parent)
    )
    assert result.returncode == 0
    assert outfile.exists()
    data = json.loads(outfile.read_text())
    assert "metadata" in data

def test_analyze_with_only_flag():
    result = subprocess.run(
        [*CLI_MODULE, "analyze", str(FIXTURES / "grid_layout.png"), "--only", "color,edge"],
        capture_output=True, text=True, cwd=str(Path(__file__).parent.parent)
    )
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert "colors" in data
    assert "edges" in data
    assert "spatial" not in data

def test_analyze_nonexistent_file():
    result = subprocess.run(
        [*CLI_MODULE, "analyze", "/nonexistent/file.png"],
        capture_output=True, text=True, cwd=str(Path(__file__).parent.parent)
    )
    assert result.returncode != 0

def test_verbose_flag():
    result = subprocess.run(
        [*CLI_MODULE, "analyze", str(FIXTURES / "solid_red.png"), "-v"],
        capture_output=True, text=True, cwd=str(Path(__file__).parent.parent)
    )
    assert result.returncode == 0
    # Verbose output goes to stderr
    assert "Analyzing" in result.stderr or "Running" in result.stderr
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/sisawat/working/design/lonis && python3 -m pytest tests/test_cli.py -v`
Expected: FAIL

**Step 3: Implement CLI**

```python
"""analyzer/cli.py"""
import argparse
import json
import sys
from pathlib import Path

from .pipeline import Pipeline

def main():
    parser = argparse.ArgumentParser(
        prog="lonis",
        description="Lonis — Bitmap-to-structured-design-data analyzer",
    )
    subparsers = parser.add_subparsers(dest="command")

    analyze = subparsers.add_parser("analyze", help="Analyze an image file")
    analyze.add_argument("image", help="Path to image file")
    analyze.add_argument("-o", "--output", help="Write JSON to file instead of stdout")
    analyze.add_argument("--only", help="Comma-separated list of analyzers to run (e.g., color,edge)")
    analyze.add_argument("-v", "--verbose", action="store_true", help="Print progress to stderr")

    args = parser.parse_args()

    if args.command != "analyze":
        parser.print_help()
        sys.exit(1)

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: file not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    analyzers = None
    if args.only:
        analyzers = [a.strip() for a in args.only.split(",")]

    if args.verbose:
        names = analyzers or ["color", "spatial", "edge", "gradient", "texture"]
        print(f"Analyzing {image_path.name} with: {', '.join(names)}", file=sys.stderr)

    try:
        pipe = Pipeline(analyzers=analyzers)
        result = pipe.run(str(image_path))
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    output = json.dumps(result, indent=2)

    if args.output:
        Path(args.output).write_text(output)
        if args.verbose:
            print(f"Written to {args.output}", file=sys.stderr)
    else:
        print(output)

if __name__ == "__main__":
    main()
```

Also create `analyzer/__main__.py` so `python -m analyzer.cli` works:

```python
"""analyzer/__main__.py"""
from .cli import main
main()
```

Wait — the CLI tests invoke `python -m analyzer.cli`, so we need `analyzer/cli.py` to be the entry point when called as `python -m analyzer.cli`. Actually, the subprocess calls `python -m analyzer.cli`, which runs `analyzer/cli.py` as `__main__`. The `if __name__ == "__main__"` block handles that.

**Step 4: Run tests**

Run: `cd /home/sisawat/working/design/lonis && python3 -m pytest tests/test_cli.py -v`
Expected: 5 passed

**Step 5: Commit**

```bash
git add analyzer/cli.py tests/test_cli.py
git commit -m "feat: CLI entry point with analyze command, --only filter, and verbose mode"
```

---

### Task 10: Integration Test with Real Image

**Files:**
- Create: `tests/test_integration.py`

**Step 1: Write integration test**

```python
"""tests/test_integration.py — Run full pipeline on the gem-boxes reference image."""
import json
import pytest
from pathlib import Path
from analyzer.pipeline import Pipeline

GEM_BOXES = Path("/home/sisawat/working/design/claude-to-figma/gem-boxes-mj.png")

@pytest.mark.skipif(not GEM_BOXES.exists(), reason="gem-boxes-mj.png not available")
class TestGemBoxesIntegration:

    def test_full_pipeline(self):
        pipe = Pipeline()
        result = pipe.run(str(GEM_BOXES))
        assert result["metadata"]["filename"] == "gem-boxes-mj.png"
        assert len(result["colors"]["dominant"]) >= 3
        assert len(result["spatial"]["regions"]) >= 1
        assert result["textures"]["global"]["roughness"] >= 0.0

    def test_dominant_colors_make_sense(self):
        pipe = Pipeline(analyzers=["color"])
        result = pipe.run(str(GEM_BOXES))
        # The image is mostly dark — dominant color should be dark
        top_color = result["colors"]["dominant"][0]
        r, g, b = top_color["rgb"]
        assert max(r, g, b) < 100, f"Expected dark dominant color, got {top_color}"

    def test_output_to_file(self, tmp_path):
        pipe = Pipeline()
        result = pipe.run(str(GEM_BOXES))
        out = tmp_path / "gem-analysis.json"
        out.write_text(json.dumps(result, indent=2))
        loaded = json.loads(out.read_text())
        assert loaded["metadata"]["filename"] == "gem-boxes-mj.png"
        # Print for manual inspection
        print(f"\nFull analysis written to: {out}")
        print(f"Colors found: {len(loaded['colors']['dominant'])}")
        print(f"Regions found: {len(loaded['spatial']['regions'])}")
        print(f"Contours found: {len(loaded['edges']['contours'])}")
```

**Step 2: Run integration test**

Run: `cd /home/sisawat/working/design/lonis && python3 -m pytest tests/test_integration.py -v -s`
Expected: 3 passed (or skipped if image not found)

**Step 3: Run ALL tests to confirm nothing is broken**

Run: `cd /home/sisawat/working/design/lonis && python3 -m pytest -v`
Expected: All tests pass

**Step 4: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: integration test with gem-boxes reference image"
```

---

### Task 11: Final Polish & Installable Package

**Files:**
- Modify: `analyzer/analyzers/__init__.py` — export all analyzers
- Verify: `pyproject.toml` — `lonis` CLI entry point works

**Step 1: Update analyzers __init__.py**

```python
"""analyzer/analyzers/__init__.py"""
from .color import ColorAnalyzer
from .spatial import SpatialAnalyzer
from .edge import EdgeAnalyzer
from .gradient import GradientAnalyzer
from .texture import TextureAnalyzer

__all__ = ["ColorAnalyzer", "SpatialAnalyzer", "EdgeAnalyzer", "GradientAnalyzer", "TextureAnalyzer"]
```

**Step 2: Install as editable package and verify CLI**

Run: `cd /home/sisawat/working/design/lonis && pip install -e .`
Run: `lonis analyze /home/sisawat/working/design/claude-to-figma/gem-boxes-mj.png | python3 -m json.tool | head -30`
Expected: Pretty-printed JSON output with metadata, colors, etc.

**Step 3: Run full test suite one final time**

Run: `cd /home/sisawat/working/design/lonis && python3 -m pytest -v`
Expected: All tests pass

**Step 4: Final commit**

```bash
git add -A
git commit -m "feat: Lonis v0.1.0 — complete image analysis pipeline with CLI"
```
