import numpy as np
from scipy import ndimage


def to_gamma(img: np.ndarray) -> np.ndarray:
    return np.where(img <= 0.0031308, 12.92 * img, 1.055 * np.power(img, 1 / 2.4) - 0.055)


def to_linear(img: np.ndarray) -> np.ndarray:
    return np.where(img <= 0.04045, img / 12.92, np.power((img + 0.055) / 1.055, 2.4))


class ColorIsp:
    """Phone-style ISP: local-ish tone curve (shadow lift, highlight roll-off), saturation, masked sharpen."""

    def __init__(self, saturation: float = 1.03, shadow_lift: float = 0.035, highlight_roll: float = 0.6, sharpen_amount: float = 0.28) -> None:
        self.saturation = saturation
        self.shadow_lift = shadow_lift
        self.highlight_roll = highlight_roll
        self.sharpen_amount = sharpen_amount

    def tone(self, linear: np.ndarray) -> np.ndarray:
        x = np.clip(linear, 0.0, 1.0)
        x = x + self.shadow_lift * np.exp(-x / 0.10) * 0.35
        x = x * (1.0 - self.highlight_roll * np.clip((x - 0.75) / 0.25, 0.0, 1.0) ** 2 * 0.18)
        g = to_gamma(np.clip(x, 0.0, 1.0))
        luma = g @ np.array([0.2126, 0.7152, 0.0722])
        return np.clip(luma[..., None] + (g - luma[..., None]) * self.saturation, 0.0, 1.0)

    def sharpen(self, gamma_img: np.ndarray) -> np.ndarray:
        detail = gamma_img - ndimage.gaussian_filter(gamma_img, sigma=(0.9, 0.9, 0))
        edge = np.abs(ndimage.gaussian_filter(gamma_img @ np.array([0.2126, 0.7152, 0.0722]), 1.0))
        mask = np.clip(edge / (np.percentile(edge, 85) + 1e-12), 0.0, 1.0)[..., None]
        delta = self.sharpen_amount * detail * (0.35 + 0.65 * mask)
        delta = np.clip(delta, -0.012, 0.012)
        return np.clip(gamma_img + delta, 0.0, 1.0)
