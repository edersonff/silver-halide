import numpy as np
from scipy import ndimage


class Grain:
    """Grain whose spectrum and amplitude follow local structure: smooth in flats,
    crisper near detail, louder near edges where sharpening excites it."""

    def __init__(self, amount: float = 0.0018, patch_sigma: float = 24.0, patch_gain: float = 0.35) -> None:
        self.amount = amount
        self.patch_sigma = patch_sigma
        self.patch_gain = patch_gain

    def apply(self, gamma_img: np.ndarray) -> np.ndarray:
        h, w = gamma_img.shape[:2]
        luma = gamma_img @ np.array([0.2126, 0.7152, 0.0722])
        detail = np.abs(luma - ndimage.gaussian_filter(luma, 1.4))
        detail_n = np.clip(detail / (np.percentile(detail, 90) + 1e-12), 0.0, 1.0)
        detail_soft = ndimage.gaussian_filter(detail_n, 2.0)

        white = np.random.normal(0.0, 1.0, size=(h, w, 1))
        soft = ndimage.gaussian_filter(np.random.normal(0.0, 1.0, size=(h, w, 1)), sigma=0.7)
        soft = soft / (soft.std() + 1e-12)
        noise = detail_soft[..., None] * white + (1.0 - detail_soft[..., None]) * soft

        field = ndimage.gaussian_filter(np.random.normal(0.0, 1.0, size=(h, w, 1)), sigma=self.patch_sigma)
        field = field / (field.std() + 1e-12)
        amplitude = 1.0 + self.patch_gain * np.clip(field, -2.0, 2.0) / 2.0

        local_luma = ndimage.gaussian_filter(luma, sigma=6.0)
        weight = 0.50 + 0.75 * np.clip(local_luma, 0.0, 1.0)
        edge = np.clip(detail_n / (np.percentile(detail_n, 97) + 1e-12), 0.0, 1.0)
        edge_boost = 1.0 + 0.45 * ndimage.gaussian_filter(edge, 1.0)

        powers = np.array([0.6, 1.0, 1.25])
        return np.clip(
            gamma_img + noise * self.amount * amplitude * (weight * edge_boost)[..., None] * powers,
            0.0,
            1.0,
        )
