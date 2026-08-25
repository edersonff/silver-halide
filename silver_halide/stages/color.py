import numpy as np
from scipy import ndimage

W = np.array([0.2126, 0.7152, 0.0722])


def to_gamma(img: np.ndarray) -> np.ndarray:
    return np.where(img <= 0.0031308, 12.92 * img, 1.055 * np.power(img, 1 / 2.4) - 0.055)


def to_linear(img: np.ndarray) -> np.ndarray:
    return np.where(img <= 0.04045, img / 12.92, np.power((img + 0.055) / 1.055, 2.4))


class ColorIsp:
    """Neutral phone ISP: symmetric micro-contrast S-curve anchored at 0/0.5/1,
    mean-luma preserving (what it lifts it gives back), no taste change."""

    def __init__(self, s_curve: float = 0.10, saturation: float = 1.0, sharpen_amount: float = 0.28) -> None:
        self.s_curve = s_curve
        self.saturation = saturation
        self.sharpen_amount = sharpen_amount

    def tone(self, linear: np.ndarray) -> np.ndarray:
        x = np.clip(linear, 0.0, 1.0)
        s = x + self.s_curve * 4.0 * x * (1.0 - x) * (x - 0.5)
        g = to_gamma(np.clip(s, 0.0, 1.0))
        before = float((to_gamma(x) @ W).mean())
        after = float((g @ W).mean())
        if after > 1e-9:
            g = g * (before / after)
        luma = g @ W
        return np.clip(luma[..., None] + (g - luma[..., None]) * self.saturation, 0.0, 1.0)

    def sharpen(self, gamma_img: np.ndarray) -> np.ndarray:
        detail = gamma_img - ndimage.gaussian_filter(gamma_img, sigma=(0.9, 0.9, 0))
        edge = np.abs(ndimage.gaussian_filter(gamma_img @ W, 1.0))
        mask = np.clip(edge / (np.percentile(edge, 85) + 1e-12), 0.0, 1.0)[..., None]
        delta = self.sharpen_amount * detail * (0.35 + 0.65 * mask)
        delta = np.clip(delta, -0.012, 0.012)
        return np.clip(gamma_img + delta, 0.0, 1.0)
