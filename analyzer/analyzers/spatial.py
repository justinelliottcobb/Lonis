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

        contours, _ = cv2.findContours(
            closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        regions = []
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            if mask is not None and not mask[y : y + ch, x : x + cw].any():
                continue
            # Dominant color in this region
            region_pixels = image[y : y + ch, x : x + cw].reshape(-1, 3)
            avg = region_pixels.mean(axis=0).astype(int)
            hex_color = f"#{avg[0]:02x}{avg[1]:02x}{avg[2]:02x}"
            regions.append((int(x), int(y), int(cw), int(ch), hex_color))

        # Sort by position: top-to-bottom, left-to-right
        regions.sort(key=lambda r: (r[1], r[0]))
        return regions

    def _detect_grid(self, regions: list, img_w: int, img_h: int) -> dict:
        """Attempt to detect a grid pattern from region positions."""
        if len(regions) < 2:
            return {
                "columns": 1,
                "rows": 1,
                "cell_size": [img_w, img_h],
                "gutters": [0, 0],
            }

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
