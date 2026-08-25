from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy import ndimage

from .stages.cfa import Cfa
from .stages.color import ColorIsp, to_linear
from .stages.denoise import EdgeAwareDenoise
from .stages.encoder import Encoder
from .stages.grain import Grain
from .stages.optics import Optics
from .stages.sensor import PRESETS, ChromaFloor, Sensor


@dataclass(frozen=True)
class Recipe:
    strength: str = "natural"
    ca: float = 0.0030
    vignette: float = 0.18
    blur: float = 0.35
    grain: float = 0.0018
    quality: int = 89
    supersample: float = 1.25
    seed: int = 7


class Develop:
    """Acquisition chain: capture oversampled, read the sensor, process, deliver small."""

    def __init__(self, recipe: Recipe) -> None:
        self.recipe = recipe
        self.sensor = Sensor(PRESETS[recipe.strength], seed=recipe.seed)
        self.optics = Optics(ca=recipe.ca, vignette=recipe.vignette, blur=recipe.blur)
        self.cfa = Cfa()
        self.denoise = EdgeAwareDenoise()
        self.isp = ColorIsp()
        self.grain = Grain(amount=recipe.grain)
        self.chroma = ChromaFloor()
        self.encoder = Encoder(quality=recipe.quality)

    def run(self, source: str, target: str) -> dict:
        np.random.seed(self.recipe.seed)
        with Image.open(source) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
        h, w = rgb.shape[:2]
        up = self.recipe.supersample
        big = ndimage.zoom(rgb, (up, up, 1), order=3)

        linear = to_linear(big)
        linear = self.white_balance(linear)
        linear = self.optics.apply(linear)
        mosaic = self.cfa.mosaic(linear)
        mosaic = self.sensor.apply(mosaic)
        linear = self.cfa.demosaic(mosaic)
        linear = self.denoise.apply(linear)
        gamma = self.isp.tone(linear)

        small = ndimage.zoom(gamma, (h / gamma.shape[0], w / gamma.shape[1], 1), order=3)
        small = np.clip(small, 0.0, 1.0)
        small = self.chroma.apply(small, iso=self.sensor.p.iso)
        small = self.grain.apply(small)
        small = self.isp.sharpen(small)
        self.encoder.save(small, target)
        return {"width": w, "height": h, "strength": self.recipe.strength}

    def white_balance(self, linear: np.ndarray) -> np.ndarray:
        t = np.random.uniform(-0.004, 0.004)
        gains = np.array([1.0 + t, 1.0, 1.0 - t])
        return np.clip(linear * gains, 0.0, 1.0)
