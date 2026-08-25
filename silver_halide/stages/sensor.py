from dataclasses import dataclass

import numpy as np
from scipy.stats import tukeylambda


@dataclass(frozen=True)
class SensorParams:
    iso: int = 100
    full_well: int = 12000
    read_sigma: float = 5e-4
    row_sigma: float = 6e-5
    prnu_sigma: float = 0.0015
    black_level: float = 0.004
    tukey_lambda: float = 0.11
    adc_bits: int = 10


class Sensor:
    """Formation model on the Bayer plane: PRNU, black level, shot, read, row FPN, ADC."""

    def __init__(self, params: SensorParams, seed: int = 7) -> None:
        self.p = params
        self.gain = params.iso / 100.0
        self.shot_capacity = params.full_well / self.gain
        rng = np.random.default_rng(seed * 7919 + 13)
        self.prnu = rng.normal(0.0, params.prnu_sigma)

    def apply(self, x: np.ndarray) -> np.ndarray:
        x = np.clip(x - self.p.black_level, 0.0, 1.0)
        sites = x * (1.0 + self.prnu)
        shot = np.random.poisson(sites * self.shot_capacity).astype(np.float64)
        shot /= self.shot_capacity
        u = np.random.uniform(0.0, 1.0, size=x.shape)
        read = tukeylambda.ppf(u, self.p.tukey_lambda) * self.p.read_sigma * self.gain**0.5
        row_shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rows = tukeylambda.ppf(
            np.random.uniform(0.0, 1.0, size=row_shape),
            self.p.tukey_lambda,
        ) * self.p.row_sigma * self.gain
        out = shot + read + rows + self.p.black_level
        q = 2**self.p.adc_bits
        return np.clip(np.round(out * q) / q, 0.0, 1.0)


PRESETS = {
    "subtle": SensorParams(iso=50, read_sigma=3e-4, row_sigma=3e-5, prnu_sigma=0.0010),
    "natural": SensorParams(),
    "harsh": SensorParams(iso=800, read_sigma=1.6e-3, row_sigma=3e-4, prnu_sigma=0.004),
}


class ChromaFloor:
    """Shadow chroma noise a phone ISP leaves behind after chroma denoise."""

    def __init__(self, sigma: float = 4e-4) -> None:
        self.sigma = sigma

    def apply(self, rgb: np.ndarray, iso: int = 400) -> np.ndarray:
        from scipy import ndimage

        luma = rgb @ np.array([0.2126, 0.7152, 0.0722])
        local = ndimage.gaussian_filter(luma, sigma=6.0)
        weight = np.clip(1.0 - local, 0.0, 1.0)
        field = ndimage.gaussian_filter(np.random.normal(0.0, 1.0, size=luma.shape), sigma=24.0)
        field = field / (field.std() + 1e-12)
        amplitude = 1.0 + 0.3 * np.clip(field, -2.0, 2.0) / 2.0
        noise = np.random.normal(0.0, 1.0, size=rgb.shape)
        gain = self.sigma * (iso / 100.0) ** 0.5
        out = rgb + noise * gain * (weight * amplitude)[..., None]
        return np.clip(out, 0.0, 1.0)
