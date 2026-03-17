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
            self.analyzer_names = analyzers

    def run(self, image_path: str) -> dict:
        """Load image, run selected analyzers, return combined result dict."""
        # Validate analyzer names at run time so the test can catch it
        for name in self.analyzer_names:
            if name not in ANALYZER_REGISTRY:
                raise ValueError(
                    f"Unknown analyzer: '{name}'. "
                    f"Available: {list(ANALYZER_REGISTRY.keys())}"
                )

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
            result[output_key] = analyzer.analyze(image, context=result)

        elapsed = time.monotonic() - start
        result["metadata"]["duration_ms"] = round(elapsed * 1000)

        return result
