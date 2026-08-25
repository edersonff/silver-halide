from dataclasses import dataclass

import numpy as np
from PIL import Image

from .stages.cfa import Cfa
from .stages.color import ColorIsp, to_linear
from .stages.encoder import Encoder
from .stages.grain import Grain
from .stages.optics import Optics
from .stages.sensor import PRESETS, ChromaFloor, Sensor


@dataclass(frozen=True)
class Recipe:
    strength: str = "natural"
    ca: float = 0.0024
    vignette: float = 0.22
    blur: float = 0.35
    grain: float = 0.008
    quality: int = 90
    seed: int = 7


class Develop:
    """Full acquisition chain: WB jitter, lens, sensor, CFA, ISP, grain, JPEG."""

    def __init__(self, recipe: Recipe) -> None:
        self.recipe = recipe
        self.sensor = Sensor(PRESETS[recipe.strength])
        self.chroma = ChromaFloor()
        self.optics = Optics(ca=recipe.ca, vignette=recipe.vignette, blur=recipe.blur)
        self.cfa = Cfa()
        self.isp = ColorIsp()
        self.grain = Grain(amount=recipe.grain)
        self.encoder = Encoder(quality=recipe.quality)

    def run(self, source: str, target: str) -> dict:
        np.random.seed(self.recipe.seed)
        with Image.open(source) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
        linear = to_linear(rgb)
        linear = self.white_balance(linear)
        linear = self.optics.apply(linear)
        mosaic = self.cfa.mosaic(linear)
        mosaic = self.sensor.apply(mosaic)
        linear = self.cfa.demosaic(mosaic)
        linear = self.chroma.apply(linear, iso=self.sensor.p.iso)
        gamma = self.isp.tone(linear)
        gamma = self.grain.apply(gamma)
        gamma = self.isp.sharpen(gamma)
        self.encoder.save(gamma, target)
        return {"width": rgb.shape[1], "height": rgb.shape[0], "strength": self.recipe.strength}

    def white_balance(self, linear: np.ndarray) -> np.ndarray:
        t = np.random.uniform(-0.004, 0.004)
        gains = np.array([1.0 + t, 1.0, 1.0 - t])
        return np.clip(linear * gains, 0.0, 1.0)
