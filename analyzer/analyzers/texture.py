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

    def analyze(self, image: np.ndarray, mask: np.ndarray | None = None, context: dict | None = None) -> dict:
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
        local_std = (
            ndimage.uniform_filter(gray**2, size=5)
            - ndimage.uniform_filter(gray, size=5) ** 2
        )
        local_std = np.sqrt(np.clip(local_std, 0, None))
        contrast = float(local_std[valid].mean())
        glossiness = float(np.clip(high_ratio * 2 + contrast, 0.0, 1.0))

        return {
            "roughness": round(roughness, 2),
            "glossiness": round(glossiness, 2),
            "uniformity": round(uniformity, 2),
        }
