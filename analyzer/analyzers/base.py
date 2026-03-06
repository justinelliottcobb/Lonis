"""analyzer/analyzers/base.py"""
from abc import ABC, abstractmethod
import numpy as np

class BaseAnalyzer(ABC):
    """Abstract base class for all Lonis analyzers.

    The mask parameter enables future region-based analysis (Phase 2)
    without changing the analyzer interface.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this analyzer, e.g. 'color'."""

    @abstractmethod
    def analyze(self, image: np.ndarray, mask: np.ndarray | None = None) -> dict:
        """Run analysis on an image.

        Args:
            image: RGB numpy array, uint8, shape (H, W, 3).
            mask:  Optional boolean mask, shape (H, W). True = include pixel.

        Returns:
            Dict of analysis results for this analyzer's domain.
        """

    def _apply_mask(self, image: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
        """Return only the masked pixels as a (N, 3) array.

        If mask is None, returns all pixels flattened.
        """
        if mask is not None:
            return image[mask]
        return image.reshape(-1, 3)
