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
        left_strip = image[:, :w // 4].reshape(-1, 3).mean(axis=0).astype(int)
        right_strip = image[:, 3 * w // 4:].reshape(-1, 3).mean(axis=0).astype(int)
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
        top_strip = image[:h // 4, :].reshape(-1, 3).mean(axis=0).astype(int)
        bot_strip = image[3 * h // 4:, :].reshape(-1, 3).mean(axis=0).astype(int)
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
