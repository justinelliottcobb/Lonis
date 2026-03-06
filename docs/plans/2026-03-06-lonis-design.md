# Lonis — Bitmap-to-Structured-Design-Data Analyzer

Named after Achewood's Lonis Edison, inventor of "Perceived Goods."

## Problem

When Claude receives a reference image for Figma design work, the image is tokenized through a vision encoder that loses precise color values, exact spatial relationships, gradient details, and texture qualities. The result is an *interpretation* rather than *measurement* — leading to guessed hex codes and approximate proportions.

## Solution

A modular image analysis tool that extracts precise, structured design data from bitmap reference images. Outputs raw measured data as JSON, consumed by Claude (or other tools) before creating designs in Figma.

## Architecture

### Phase 1: Pipeline (Current)

```
image → [ColorAnalyzer] → [SpatialAnalyzer] → [EdgeDetector] → [GradientMapper] → [TextureClassifier] → JSON
```

Each analyzer runs independently on the full image. Modular, testable, easy to extend.

### Phase 2: Region-Parallel (Future)

```
image → [Segmenter (SAM)] → regions[]
  → per-region: [Color] + [Edge] + [Gradient] + [Texture] + [Semantic (CLIP)] → JSON
```

Same analyzer interfaces, applied per-region with masks. The `BaseAnalyzer` interface supports this without modification via its `mask` parameter.

### Phase 3: Semantic (Future)

Pluggable provider system for semantic analysis:
- **SAM**: Object segmentation (also enables Phase 2)
- **CLIP**: Region classification and attribute tagging
- **Florence-2**: Dense captioning and object detection
- **Claude API**: High-level scene understanding via vision API
- Providers are composable — run any combination and merge results

## Project Structure

```
lonis/
├── analyzer/                    # Python core
│   ├── __init__.py
│   ├── cli.py                   # CLI entry point
│   ├── pipeline.py              # Orchestrates analyzers
│   ├── analyzers/
│   │   ├── __init__.py
│   │   ├── base.py              # Abstract base — analyze(image, mask?) -> dict
│   │   ├── color.py             # Dominant colors, palette, harmony, distribution
│   │   ├── spatial.py           # Grid detection, region positions, density
│   │   ├── edge.py              # Contours, shapes, edge density, angles
│   │   ├── gradient.py          # Direction, transitions, luminance mapping
│   │   └── texture.py           # Roughness, glossiness, pattern classification
│   └── utils/
│       ├── __init__.py
│       └── image.py             # Image loading, region extraction, format handling
├── mcp-server/                  # TypeScript MCP wrapper (Phase 1b)
│   ├── package.json
│   ├── tsconfig.json
│   └── server.ts                # Calls Python CLI, returns JSON via MCP
├── tests/
│   ├── test_color.py
│   ├── test_spatial.py
│   ├── test_edge.py
│   ├── test_gradient.py
│   ├── test_texture.py
│   └── fixtures/                # Test images
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Module Interfaces

### BaseAnalyzer

```python
from abc import ABC, abstractmethod
import numpy as np

class BaseAnalyzer(ABC):
    """Base class for all analyzers.

    The mask parameter enables future region-based analysis (Phase 2)
    without changing the analyzer interface.
    """

    @abstractmethod
    def analyze(self, image: np.ndarray, mask: np.ndarray | None = None) -> dict:
        """
        Args:
            image: RGB numpy array, uint8, shape (H, W, 3)
            mask:  Optional boolean mask, shape (H, W). True = analyze this pixel.

        Returns:
            Dict of analysis results for this analyzer's domain.
        """
        pass
```

### Pipeline

```python
class Pipeline:
    def __init__(self, analyzers: list[str] | None = None):
        """
        Args:
            analyzers: List of analyzer names to run.
                       None = run all available.
                       e.g., ["color", "spatial", "edge"]
        """

    def run(self, image_path: str) -> dict:
        """Load image, run selected analyzers, return combined JSON."""
```

### SemanticAnalyzer (Future — Phase 3)

```python
class SemanticAnalyzer(BaseAnalyzer):
    def __init__(self, providers: list[str] = ["clip"]):
        """
        Args:
            providers: List of semantic backends to run.
                       ["clip", "sam", "florence", "claude"]
                       Results are merged across providers.
        """
```

## Output Schema

```json
{
  "metadata": {
    "filename": "gem-boxes-mj.png",
    "dimensions": [1376, 608],
    "format": "PNG",
    "analyzed_at": "2026-03-06T10:30:00Z",
    "analyzers_run": ["color", "spatial", "edge", "gradient", "texture"],
    "duration_ms": 1240
  },
  "colors": {
    "dominant": [
      {
        "hex": "#1a1820",
        "rgb": [26, 24, 32],
        "hsl": [252, 14, 11],
        "percentage": 42.3
      }
    ],
    "palette": [],
    "harmony": {
      "temperature": "warm-dominant",
      "contrast_ratio": 8.2
    }
  },
  "spatial": {
    "grid": {
      "columns": 7,
      "rows": 3,
      "cell_size": [196, 203],
      "gutters": [12, 8]
    },
    "regions": [
      {
        "id": 0,
        "bounds": { "x": 42, "y": 18, "w": 120, "h": 140 },
        "dominant_color": "#c87432",
        "area_percentage": 3.2
      }
    ]
  },
  "edges": {
    "contours": [
      {
        "id": 0,
        "bounds": { "x": 42, "y": 18, "w": 120, "h": 140 },
        "shape_type": "rectangle",
        "vertex_count": 4,
        "area": 16800
      }
    ],
    "edge_density": 0.23,
    "dominant_angles": [0, 90]
  },
  "gradients": {
    "global_direction": 180,
    "luminance_range": { "min": 4, "max": 218, "mean": 42 },
    "transitions": [
      {
        "region": [0, 0, 1376, 608],
        "direction": "top-to-bottom",
        "from_color": "#2a2830",
        "to_color": "#0e0d12",
        "strength": 0.4
      }
    ]
  },
  "textures": {
    "global": {
      "roughness": 0.3,
      "glossiness": 0.7,
      "uniformity": 0.4
    },
    "regions": [
      {
        "bounds": { "x": 42, "y": 18, "w": 120, "h": 140 },
        "roughness": 0.1,
        "glossiness": 0.9,
        "pattern": "smooth-reflective"
      }
    ]
  }
}
```

Future `semantic` key (Phase 3):

```json
"semantic": {
  "scene_description": "Arrangement of translucent gem cases on dark surface",
  "objects": [
    {
      "id": 0,
      "bounds": { "x": 42, "y": 18, "w": 120, "h": 140 },
      "label": "rectangular translucent container",
      "attributes": ["amber-tinted", "hinged lid", "glossy"],
      "confidence": 0.89,
      "provider": "clip"
    }
  ],
  "materials": ["glass", "crystal", "metal filigree"],
  "style_tags": ["luxury", "moody lighting", "dark background"],
  "composition": "grid arrangement, centered, symmetrical",
  "providers_used": ["sam", "clip"]
}
```

## CLI Interface

```bash
# Full analysis (all pixel-level analyzers)
lonis analyze gem-boxes-mj.png

# Output to file
lonis analyze gem-boxes-mj.png -o analysis.json

# Select specific analyzers
lonis analyze gem-boxes-mj.png --only color,spatial

# Verbose with timing
lonis analyze gem-boxes-mj.png -v

# Future: semantic analysis
lonis analyze gem-boxes-mj.png --semantic clip,sam

# Future: region-parallel mode
lonis analyze gem-boxes-mj.png --segment --semantic sam,clip
```

## MCP Server (Phase 1b)

TypeScript wrapper that:
1. Receives `analyze_image` tool call with image path
2. Spawns `python3 -m analyzer.cli analyze <path> --json`
3. Parses and returns the JSON result

Registered as an MCP server in Claude Code config alongside the Figma MCP.

## Dependencies

### Phase 1 (v1)
- `Pillow` — image loading and basic manipulation
- `opencv-python-headless` — edge detection, contours, gradient analysis
- `numpy` — array operations
- `scikit-learn` — KMeans clustering for color palette extraction
- `scipy` — spatial analysis, frequency domain texture analysis

### Phase 3 (Future)
- `segment-anything` (SAM) — object segmentation
- `transformers` + `torch` — CLIP, Florence-2
- `anthropic` — Claude API for semantic analysis

## v1 Scope

Deliver the five pixel-level analyzers, CLI, and JSON output. No ML models, no MCP wrapper. The tool should work end-to-end:

```bash
pip install -e .
lonis analyze /path/to/gem-boxes-mj.png -o analysis.json
```

And produce a complete, accurate JSON file that Claude can read before creating Figma designs.
