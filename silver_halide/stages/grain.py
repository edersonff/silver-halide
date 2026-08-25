import numpy as np
from scipy import ndimage


class Grain:
    """Filmgrainer-lineage grain: blurred white noise, channel powers, shadow bias."""

    def __init__(self, amount: float = 0.007, blur: float = 0.75, shadow_bias: float = 0.6) -> None:
        self.amount = amount
        self.blur = blur
        self.shadow_bias = shadow_bias

    def apply(self, gamma_img: np.ndarray) -> np.ndarray:
        h, w = gamma_img.shape[:2]
        n = np.random.normal(0.0, 1.0, size=(h, w, 1))
        n = ndimage.gaussian_filter(n, sigma=self.blur)
        n = n / (n.std() + 1e-12)
        luma = gamma_img @ np.array([0.2126, 0.7152, 0.0722])
        base = self.shadow_bias + (1.0 - self.shadow_bias) * (1.0 - luma)
        highlight = 1.0 - 0.7 * np.clip((luma - 0.88) / 0.12, 0.0, 1.0)
        weight = base * highlight
        powers = np.array([0.55, 1.0, 1.3])
        return np.clip(gamma_img + n * self.amount * weight[..., None] * powers, 0.0, 1.0)
