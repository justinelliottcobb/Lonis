"""analyzer/analyzers/edge.py"""
import cv2
import numpy as np
from .base import BaseAnalyzer


class EdgeAnalyzer(BaseAnalyzer):
    """Detects contours, classifies shapes, and measures edge characteristics.

    Uses coarse-to-fine detection when spatial context is available:
    1. Coarse pass: Canny(50, 150) on full image
    2. Fine pass: bilateral filter + Canny(15, 60) on spatial regions
       that the coarse pass missed
    """

    @property
    def name(self) -> str:
        return "edge"

    def analyze(self, image: np.ndarray, mask: np.ndarray | None = None, context: dict | None = None) -> dict:
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        if mask is not None:
            gray = cv2.bitwise_and(gray, gray, mask=mask.astype(np.uint8) * 255)

        # --- Coarse pass: global Canny at standard thresholds ---
        coarse_edges = cv2.Canny(gray, 50, 150)
        coarse_contours = self._find_contours(coarse_edges, h, w, min_area_frac=0.001)

        # Tag coarse contours
        for c in coarse_contours:
            c["confidence"] = "coarse"

        # --- Fine pass: targeted detection in uncovered spatial regions ---
        fine_contours = []
        fine_edges = np.zeros_like(coarse_edges)

        spatial_regions = self._get_spatial_regions(context)
        if spatial_regions:
            uncovered = self._find_uncovered_regions(spatial_regions, coarse_contours)
            for region in uncovered:
                region_contours, region_edges = self._fine_pass_region(
                    image, gray, region, h, w
                )
                fine_contours.extend(region_contours)
                # Accumulate fine edge pixels for density/angle computation
                fine_edges = cv2.bitwise_or(fine_edges, region_edges)

        # Tag fine contours
        for c in fine_contours:
            c["confidence"] = "fine"

        # --- Merge and deduplicate ---
        all_contours = self._merge_contours(coarse_contours + fine_contours)

        # Re-number IDs
        for i, c in enumerate(all_contours):
            c["id"] = i

        # Combined edge map for density and angles
        combined_edges = cv2.bitwise_or(coarse_edges, fine_edges)
        edge_density = round(float(np.count_nonzero(combined_edges)) / (h * w), 3)
        dominant_angles = self._detect_dominant_angles(combined_edges)

        return {
            "contours": all_contours,
            "edge_density": edge_density,
            "dominant_angles": dominant_angles,
        }

    def _find_contours(self, edges: np.ndarray, h: int, w: int, min_area_frac: float) -> list[dict]:
        """Find and classify contours from an edge map."""
        contours_raw, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        min_area = h * w * min_area_frac

        contours = []
        for cnt in contours_raw:
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
                "vertices": [[int(pt[0][0]), int(pt[0][1])] for pt in approx],
                "area": int(area),
            })
        return contours

    @staticmethod
    def _get_spatial_regions(context: dict | None) -> list[dict]:
        """Extract spatial regions from pipeline context."""
        if context is None:
            return []
        spatial = context.get("spatial")
        if spatial is None:
            return []
        return spatial.get("regions", [])

    @staticmethod
    def _find_uncovered_regions(spatial_regions: list[dict], contours: list[dict]) -> list[dict]:
        """Find spatial regions not covered by any existing contour (IoU > 0.3)."""
        uncovered = []
        for region in spatial_regions:
            rb = region["bounds"]
            covered = False
            for c in contours:
                cb = c["bounds"]
                # Compute IoU
                ix1 = max(rb["x"], cb["x"])
                iy1 = max(rb["y"], cb["y"])
                ix2 = min(rb["x"] + rb["w"], cb["x"] + cb["w"])
                iy2 = min(rb["y"] + rb["h"], cb["y"] + cb["h"])
                if ix2 > ix1 and iy2 > iy1:
                    inter = (ix2 - ix1) * (iy2 - iy1)
                    union = rb["w"] * rb["h"] + cb["w"] * cb["h"] - inter
                    if union > 0 and inter / union > 0.3:
                        covered = True
                        break
            if not covered:
                uncovered.append(region)
        return uncovered

    def _fine_pass_region(
        self, image: np.ndarray, gray: np.ndarray, region: dict, img_h: int, img_w: int
    ) -> tuple[list[dict], np.ndarray]:
        """Run fine-grained edge detection on a single spatial region.

        Uses bilateral filtering + lower Canny thresholds to catch
        subtle edges that the coarse pass missed.
        """
        b = region["bounds"]
        pad = 10

        # Clamp to image bounds
        x1 = max(0, b["x"] - pad)
        y1 = max(0, b["y"] - pad)
        x2 = min(img_w, b["x"] + b["w"] + pad)
        y2 = min(img_h, b["y"] + b["h"] + pad)

        roi_gray = gray[y1:y2, x1:x2]
        if roi_gray.size == 0:
            return [], np.zeros((img_h, img_w), dtype=np.uint8)

        # Bilateral filter preserves edges while smoothing noise
        filtered = cv2.bilateralFilter(roi_gray, 9, 75, 75)
        roi_edges = cv2.Canny(filtered, 15, 60)

        # Light morphological closing to connect nearby edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        roi_edges = cv2.morphologyEx(roi_edges, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Find contours in ROI
        roi_h, roi_w = roi_edges.shape
        region_area = b["w"] * b["h"]
        min_area = region_area * 0.05  # 5% of region area

        contours_raw, _ = cv2.findContours(
            roi_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        contours = []
        for cnt in contours_raw:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            x, y, cw, ch = cv2.boundingRect(cnt)
            shape_type = self._classify_shape(cnt)
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            # Translate coordinates back to full image space
            contours.append({
                "id": 0,
                "bounds": {"x": int(x + x1), "y": int(y + y1), "w": int(cw), "h": int(ch)},
                "shape_type": shape_type,
                "vertex_count": len(approx),
                "vertices": [[int(pt[0][0] + x1), int(pt[0][1] + y1)] for pt in approx],
                "area": int(area),
            })

        # Place ROI edges back into full-size edge map
        full_edges = np.zeros((img_h, img_w), dtype=np.uint8)
        full_edges[y1:y2, x1:x2] = roi_edges

        return contours, full_edges

    @staticmethod
    def _merge_contours(contours: list[dict]) -> list[dict]:
        """Merge overlapping contours (IoU > 0.5), keeping the larger one."""
        if len(contours) <= 1:
            return contours

        merged = []
        used = set()
        for i, c1 in enumerate(contours):
            if i in used:
                continue
            best = c1
            b1 = c1["bounds"]
            for j in range(i + 1, len(contours)):
                if j in used:
                    continue
                b2 = contours[j]["bounds"]
                ix1 = max(b1["x"], b2["x"])
                iy1 = max(b1["y"], b2["y"])
                ix2 = min(b1["x"] + b1["w"], b2["x"] + b2["w"])
                iy2 = min(b1["y"] + b1["h"], b2["y"] + b2["h"])
                if ix2 > ix1 and iy2 > iy1:
                    inter = (ix2 - ix1) * (iy2 - iy1)
                    a1 = b1["w"] * b1["h"]
                    a2 = b2["w"] * b2["h"]
                    union = a1 + a2 - inter
                    if union > 0 and inter / union > 0.5:
                        if a2 > a1:
                            best = contours[j]
                        used.add(j)
            merged.append(best)

        # Sort by position: top-to-bottom, left-to-right
        merged.sort(key=lambda c: (c["bounds"]["y"], c["bounds"]["x"]))
        return merged

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
