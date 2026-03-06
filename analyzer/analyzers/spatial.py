"""analyzer/analyzers/spatial.py"""
import cv2
import numpy as np
from .base import BaseAnalyzer


class SpatialAnalyzer(BaseAnalyzer):
    """Detects spatial layout grid and distinct visual regions.

    Uses bilateral filtering to preserve color transitions while smoothing
    noise, then Canny edge detection to find where sharp gradient changes
    occur in space. This measures spatial color transitions — not objects.
    """

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
        """Find distinct visual regions via gradient transition detection.

        Bilateral filter preserves color boundaries while smoothing noise,
        allowing Canny to find clean edges without aggressive morphological
        closing that would merge adjacent regions.
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Bilateral filter: preserves edges, smooths noise — key improvement
        # over Gaussian blur which smears edges and requires heavy closing
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)

        # Lower Canny thresholds catch subtle transitions in dark images
        edges = cv2.Canny(filtered, 15, 60)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(
            closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        regions = []
        min_area = h * w * self.min_region_area_pct / 100
        for cnt in contours:
            if cv2.contourArea(cnt) < min_area:
                continue
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
        return self._merge_overlapping(regions)

    @staticmethod
    def _merge_overlapping(regions: list) -> list:
        """Merge regions whose bounding boxes overlap significantly (IoU > 0.5)."""
        if len(regions) <= 1:
            return regions
        merged = []
        used = set()
        for i, r1 in enumerate(regions):
            if i in used:
                continue
            best = r1
            for j in range(i + 1, len(regions)):
                if j in used:
                    continue
                r2 = regions[j]
                ix1 = max(r1[0], r2[0])
                iy1 = max(r1[1], r2[1])
                ix2 = min(r1[0] + r1[2], r2[0] + r2[2])
                iy2 = min(r1[1] + r1[3], r2[1] + r2[3])
                if ix2 > ix1 and iy2 > iy1:
                    inter = (ix2 - ix1) * (iy2 - iy1)
                    union = r1[2] * r1[3] + r2[2] * r2[3] - inter
                    if union > 0 and inter / union > 0.5:
                        if r2[2] * r2[3] > best[2] * best[3]:
                            best = r2
                        used.add(j)
            merged.append(best)
        merged.sort(key=lambda r: (r[1], r[0]))
        return merged

    def _detect_grid(self, regions: list, img_w: int, img_h: int) -> dict:
        """Detect grid pattern from region positions.

        Filters out canvas-spanning regions (>25% of image area) before
        inferring grid structure, since these are background, not cells.
        """
        img_area = img_w * img_h
        cell_regions = [r for r in regions if r[2] * r[3] < img_area * 0.25]

        if len(cell_regions) < 2:
            return {
                "columns": 1,
                "rows": 1,
                "cell_size": [img_w, img_h],
                "gutters": [0, 0],
            }

        xs = sorted(set(r[0] for r in cell_regions))
        ys = sorted(set(r[1] for r in cell_regions))

        # Cluster positions — use median cell height for row threshold
        # to avoid splitting rows with slight y-offsets
        heights = [r[3] for r in cell_regions]
        cell_h_est = int(np.median(heights)) if heights else img_h // 2
        widths = [r[2] for r in cell_regions]
        cell_w_est = int(np.median(widths)) if widths else img_w // 2

        cols = self._cluster_positions(xs, threshold=cell_w_est * 0.5)
        rows = self._cluster_positions(ys, threshold=cell_h_est * 0.5)

        n_cols = len(cols)
        n_rows = len(rows)

        cell_w = cell_w_est
        cell_h = cell_h_est

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
