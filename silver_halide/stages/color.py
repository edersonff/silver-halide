import numpy as np
from scipy import ndimage


def to_gamma(img: np.ndarray) -> np.ndarray:
    return np.where(img <= 0.0031308, 12.92 * img, 1.055 * np.power(img, 1 / 2.4) - 0.055)


def to_linear(img: np.ndarray) -> np.ndarray:
    return np.where(img <= 0.04045, img / 12.92, np.power((img + 0.055) / 1.055, 2.4))


class ColorIsp:
    """Phone-style ISP: WB happens in the pipeline; here tone, saturation, then sharpen."""

    def __init__(self, saturation: float = 1.06, contrast: float = 0.05, sharpen_amount: float = 0.3) -> None:
        self.saturation = saturation
        self.contrast = contrast
        self.sharpen_amount = sharpen_amount

    def tone(self, linear: np.ndarray) -> np.ndarray:
        g = to_gamma(np.clip(linear, 0.0, 1.0))
        g = 0.5 + (g - 0.5) * (1.0 + self.contrast)
        luma = g @ np.array([0.2126, 0.7152, 0.0722])
        return np.clip(luma[..., None] + (g - luma[..., None]) * self.saturation, 0.0, 1.0)

    def sharpen(self, gamma_img: np.ndarray) -> np.ndarray:
        soft = ndimage.gaussian_filter(gamma_img, sigma=(1.1, 1.1, 0))
        return np.clip(gamma_img + self.sharpen_amount * (gamma_img - soft), 0.0, 1.0)
