from dataclasses import dataclass

import numpy as np
from scipy.stats import tukeylambda


@dataclass(frozen=True)
class SensorParams:
    iso: int = 400
    full_well: int = 5000
    read_sigma: float = 1.8e-3
    row_sigma: float = 0.22e-3
    tukey_lambda: float = 0.11
    adc_bits: int = 10


class Sensor:
    """ELD-style formation model: shot Poisson, Tukey read noise, row FPN, ADC quantize."""

    def __init__(self, params: SensorParams) -> None:
        self.p = params
        self.gain = params.iso / 100.0
        self.shot_capacity = params.full_well / self.gain

    def apply(self, x: np.ndarray) -> np.ndarray:
        x = np.clip(x, 0.0, 1.0)
        shot = np.random.poisson(x * self.shot_capacity).astype(np.float64)
        shot /= self.shot_capacity
        u = np.random.uniform(0.0, 1.0, size=x.shape)
        read = tukeylambda.ppf(u, self.p.tukey_lambda) * self.p.read_sigma * self.gain**0.5
        row_shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rows = tukeylambda.ppf(
            np.random.uniform(0.0, 1.0, size=row_shape),
            self.p.tukey_lambda,
        ) * self.p.row_sigma * self.gain
        out = shot + read + rows
        q = 2**self.p.adc_bits
        return np.clip(np.round(out * q) / q, 0.0, 1.0)


PRESETS = {
    "subtle": SensorParams(iso=100, read_sigma=1.1e-3, row_sigma=0.10e-3),
    "natural": SensorParams(iso=400),
    "harsh": SensorParams(iso=1600, read_sigma=2.6e-3, row_sigma=0.45e-3),
}


class ChromaFloor:
    """Shadow chroma noise a phone ISP leaves behind after chroma denoise."""

    def __init__(self, sigma: float = 0.0012) -> None:
        self.sigma = sigma

    def apply(self, rgb: np.ndarray, iso: int = 400) -> np.ndarray:
        luma = rgb @ np.array([0.2126, 0.7152, 0.0722])
        weight = 1.0 - luma
        noise = np.random.normal(0.0, 1.0, size=rgb.shape)
        gain = self.sigma * (iso / 100.0) ** 0.5
        out = rgb + noise * gain * weight[..., None]
        return np.clip(out, 0.0, 1.0)
