import numpy as np
from scipy import ndimage


class Optics:
    """Field-dependent lens: quadratic lateral CA, asymmetric vignette, faint astigmatic blur."""

    def __init__(self, ca: float = 0.0035, vignette: float = 0.22, blur: float = 0.35) -> None:
        self.ca = ca
        self.vignette = vignette
        self.blur = blur

    def apply(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        cx, cy = w / 2 + w * 0.012, h / 2 - h * 0.008
        r = np.sqrt(((xx - cx) / (w / 2)) ** 2 + ((yy - cy) / (h / 2)) ** 2) / 1.35
        ca_map = self.ca * np.clip(r, 0.0, 1.4) ** 1.6
        out = np.stack(
            [
                self._rescale(img[..., 0], 1.0 + ca_map, cy, cx),
                img[..., 1],
                self._rescale(img[..., 2], 1.0 - ca_map, cy, cx),
            ],
            axis=-1,
        )
        out = ndimage.gaussian_filter(out, sigma=(self.blur, self.blur * 1.2, 0))
        ell = 1.0 + 0.06 * ((xx - cx) / (w / 2))
        falloff = 1.0 - self.vignette * np.clip(r * ell, 0.0, 1.0) ** 2.2
        return np.clip(out * falloff[..., None], 0.0, 1.0)

    def _rescale(self, channel: np.ndarray, factor: np.ndarray, cy: float, cx: float) -> np.ndarray:
        h, w = channel.shape
        yy, xx = np.mgrid[0:h, 0:w]
        coords = np.array([(yy - cy) / factor + cy, (xx - cx) / factor + cx])
        return ndimage.map_coordinates(channel, coords, order=1, mode="nearest")
