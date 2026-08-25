import numpy as np
from scipy import ndimage


class EdgeAwareDenoise:
    """Phone ISP YUV-domain NR: kills the painterly mid-texture wash (2-6px daubs),
    preserves strong edges; runs post-gamma like real phone pipelines."""

    def __init__(self, chroma_sigma: float = 0.6, luma_flat_gain: float = 0.45, flat_threshold: float = 8.5e-3, wash_radius: int = 3, passes: int = 1) -> None:
        self.chroma_sigma = chroma_sigma
        self.luma_flat_gain = luma_flat_gain
        self.flat_threshold = flat_threshold
        self.wash_radius = wash_radius
        self.passes = passes

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        y = rgb @ np.array([0.2126, 0.7152, 0.0722])
        cb = rgb[..., 2] - y
        cr = rgb[..., 0] - y
        cb_s = self._blur(cb, self.chroma_sigma)
        cr_s = self._blur(cr, self.chroma_sigma)
        for _ in range(self.passes):
            detail = np.abs(y - self._blur(y, 1.2))
            flat = self._soft(detail, self.flat_threshold)
            box = ndimage.uniform_filter(y, size=2 * self.wash_radius + 1)
            y = y - (self.luma_flat_gain * flat) * (y - box)
        self.last_flat = flat
        r = np.clip(y + cr_s, 0.0, 1.0)
        b = np.clip(y + cb_s, 0.0, 1.0)
        g = np.clip((y - 0.2126 * r - 0.0722 * b) / 0.7152, 0.0, 1.0)
        return np.stack([r, g, b], axis=-1)

    @staticmethod
    def _soft(detail: np.ndarray, threshold: float) -> np.ndarray:
        span = threshold * 1.2
        return np.clip((threshold + span - detail) / (2.0 * span), 0.0, 1.0)

    @staticmethod
    def _blur(channel: np.ndarray, sigma: float) -> np.ndarray:
        from scipy import ndimage

        return ndimage.gaussian_filter(channel, sigma)
