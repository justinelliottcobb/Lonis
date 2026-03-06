# Lonis

Bitmap-to-structured-design-data analyzer. Named after Achewood's Lonis Edison, inventor of "Perceived Goods."

## What It Does

Lonis extracts precise, measured design data from bitmap images and outputs it as structured JSON. It solves the problem of vision encoders *interpreting* rather than *measuring* — so you get exact hex codes, spatial relationships, and texture properties instead of guesses.

```
image → [Color] → [Spatial] → [Edge] → [Gradient] → [Texture] → JSON
```

## Install

```bash
git clone git@github.com:justinelliottcobb/Lonis.git
cd Lonis
pip install -e .
```

## Usage

```bash
# Full analysis
lonis analyze photo.png

# Output to file
lonis analyze photo.png -o analysis.json

# Select specific analyzers
lonis analyze photo.png --only color,spatial

# Verbose with timing
lonis analyze photo.png -v
```

## Analyzers

| Analyzer | What It Measures |
|----------|-----------------|
| **color** | Dominant colors via KMeans clustering, hex/RGB/HSL values, palette, color temperature, contrast ratio |
| **spatial** | Grid detection (columns, rows, gutters), distinct visual regions with bounding boxes |
| **edge** | Contour detection, shape classification (rectangle, ellipse, triangle), edge density, dominant angles |
| **gradient** | Global gradient direction, luminance range (min/max/mean), color transitions with strength |
| **texture** | Surface roughness, glossiness, uniformity — from Laplacian and local contrast analysis |

## Output

```json
{
  "metadata": {
    "filename": "photo.png",
    "dimensions": [1680, 720],
    "format": "PNG",
    "analyzed_at": "2026-03-06T18:26:47Z",
    "analyzers_run": ["color", "spatial", "edge", "gradient", "texture"],
    "duration_ms": 510
  },
  "colors": {
    "dominant": [
      { "hex": "#1a1820", "rgb": [26, 24, 32], "hsl": [252, 14, 11], "percentage": 42.3 }
    ],
    "palette": ["#1a1820", ...],
    "harmony": { "temperature": "warm-dominant", "contrast_ratio": 8.2 }
  },
  "spatial": {
    "grid": { "columns": 3, "rows": 2, "cell_size": [196, 203], "gutters": [12, 8] },
    "regions": [
      { "id": 0, "bounds": { "x": 42, "y": 18, "w": 120, "h": 140 }, "dominant_color": "#c87432", "area_percentage": 3.2 }
    ]
  },
  "edges": {
    "contours": [
      { "id": 0, "bounds": { "x": 42, "y": 18, "w": 120, "h": 140 }, "shape_type": "rectangle", "vertex_count": 4, "area": 16800 }
    ],
    "edge_density": 0.23,
    "dominant_angles": [0, 90]
  },
  "gradients": {
    "global_direction": 180,
    "luminance_range": { "min": 4, "max": 218, "mean": 42 },
    "transitions": [
      { "direction": "top-to-bottom", "from_color": "#2a2830", "to_color": "#0e0d12", "strength": 0.4 }
    ]
  },
  "textures": {
    "global": { "roughness": 0.3, "glossiness": 0.7, "uniformity": 0.4 }
  }
}
```

## Dependencies

- **Pillow** — image loading
- **opencv-python-headless** — edge detection, contours, gradients
- **numpy** — array operations
- **scikit-learn** — KMeans color clustering
- **scipy** — texture analysis (Laplacian, uniform filters)

## Roadmap

- **Phase 1** (current): Pipeline of 5 pixel-level analyzers + CLI
- **Phase 1b**: MCP server wrapper for use with Claude Code
- **Phase 2**: Region-parallel analysis via SAM segmentation
- **Phase 3**: Semantic analysis with pluggable providers (CLIP, Florence-2, Claude API)

## Tests

```bash
pytest -v          # 48 tests
pytest -v -s tests/test_integration.py  # real image test (needs gem-boxes-mj.png)
```
