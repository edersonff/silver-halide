import numpy as np
from scipy import ndimage

W = np.array([0.2126, 0.7152, 0.0722])


class MicroTexture:
    """Transplants real micro-texture (high-frequency residue of a real photo
    patch) into painterly regions. Continuous lift, no mask edges: each pixel
    gets exactly the texture energy it is missing up to the target."""

    def __init__(self, exemplar: np.ndarray, target_rms: float = 0.0128, max_add: float = 0.0032, seed: int = 7) -> None:
        luma = exemplar @ W
        self.hf = exemplar - np.stack([ndimage.gaussian_filter(exemplar[..., c], 1.5) for c in range(3)], axis=-1)
        self.hf_rms = float(np.sqrt((self.hf @ W).var()))
        self.target = target_rms
        self.max_add = max_add
        self.seed = seed

    def _tile(self, h: int, w: int) -> np.ndarray:
        rng = np.random.default_rng(self.seed * 31 + 5)
        c, eh, ew = 3, self.hf.shape[0], self.hf.shape[1]
        oy, ox = int(rng.integers(0, eh)), int(rng.integers(0, ew))
        out = np.zeros((h, w, c))
        for i in range(0, h, eh):
            for j in range(0, w, ew):
                sy = (oy + i) % eh
                sx = (ox + j) % ew
                tile = np.roll(np.roll(self.hf, sy, axis=0), sx, axis=1)
                out[i : i + eh, j : j + ew] = tile[: h - i, : w - j]
        return out

    def apply(self, gamma_img: np.ndarray) -> np.ndarray:
        luma = gamma_img @ W
        hp = luma - ndimage.gaussian_filter(luma, 1.5)
        current = np.sqrt(ndimage.uniform_filter(hp * hp, size=16))
        lift = np.clip(self.target - current, 0.0, self.max_add)
        lift = ndimage.gaussian_filter(lift, sigma=3.0)
        field = self._tile(*gamma_img.shape[:2])
        scale = lift / (self.hf_rms + 1e-12)
        return np.clip(gamma_img + field * scale[..., None], 0.0, 1.0)
