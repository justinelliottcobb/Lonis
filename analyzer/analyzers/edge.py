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
        contours_raw, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

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
            contours.append(
                {
                    "id": len(contours),
                    "bounds": {"x": int(x), "y": int(y), "w": int(cw), "h": int(ch)},
                    "shape_type": shape_type,
                    "vertex_count": len(approx),
                    "area": int(area),
                }
            )

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
            x, y, w, h = cv2.boundingRect(approx)
            area_ratio = cv2.contourArea(approx) / (w * h) if w * h > 0 else 0
            if 0.85 < area_ratio:
                return "rectangle"
            return "quadrilateral"
        elif vertices > 6:
            area = cv2.contourArea(contour)
            circularity = 4 * np.pi * area / (peri * peri) if peri > 0 else 0
            if circularity > 0.7:
                return "ellipse"
        return "irregular"

    @staticmethod
    def _detect_dominant_angles(edges: np.ndarray) -> list[int]:
        """Detect dominant edge angles using Hough Line Transform."""
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=30, minLineLength=20, maxLineGap=10
        )
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
        sorted_indices = np.argsort(-hist)[:3]
        threshold = max(hist) * 0.3
        dominant = [
            int(bins[i] + 7) for i in sorted_indices if hist[i] >= threshold
        ]
        return sorted(set(dominant))
