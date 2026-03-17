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

    def analyze(self, image: np.ndarray, mask: np.ndarray | None = None, context: dict | None = None) -> dict:
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
