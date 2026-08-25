import numpy as np
from scipy import ndimage


class Motion:
    """Hand tremor: subpixel line PSF at a random angle, like a real handheld exposure."""

    def __init__(self, min_px: float = 0.9, max_px: float = 1.6) -> None:
        self.min_px = min_px
        self.max_px = max_px

    def apply(self, img: np.ndarray) -> np.ndarray:
        length = np.random.uniform(self.min_px, self.max_px)
        angle = np.random.uniform(0.0, np.pi)
        size = int(np.ceil(length)) * 2 + 1
        c = size // 2
        yy, xx = np.mgrid[0:size, 0:size] - c
        along = xx * np.cos(angle) + yy * np.sin(angle)
        across = -xx * np.sin(angle) + yy * np.cos(angle)
        kernel = np.exp(-(across**2) / 0.35) * np.clip(1.0 - np.abs(along) / max(length, 1.01), 0.0, 1.0)
        kernel = kernel / kernel.sum()
        if img.ndim == 3:
            kernel = kernel[..., None]
        return np.clip(ndimage.convolve(img, kernel, mode="nearest"), 0.0, 1.0)
