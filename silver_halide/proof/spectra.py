import base64
import io

import numpy as np
from PIL import Image
from scipy import ndimage


def load_gray(path: str) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.float64) / 255.0


def radial_slope(gray: np.ndarray) -> float:
    """Slope of log power vs log frequency; natural photos sit near -2, AI renders run steeper."""
    f = np.fft.fftshift(np.fft.fft2(gray - gray.mean()))
    power = np.abs(f) ** 2
    h, w = power.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((yy - h / 2) ** 2 + (xx - w / 2) ** 2)
    top = min(h, w) / 2
    bins = np.arange(1, top, 2.0)
    dig = np.digitize(r.ravel(), bins)
    mean_r = ndimage.mean(r.ravel(), dig, np.arange(1, len(bins) + 1))
    mean_p = ndimage.mean(power.ravel(), dig, np.arange(1, len(bins) + 1))
    keep = mean_p > 0
    slope = np.polyfit(np.log(mean_r[keep]), np.log(mean_p[keep]), 1)[0]
    return float(slope)


def noise_variance(gray: np.ndarray) -> dict:
    """Local std of the high-pass residual: the texture a sensor leaves behind."""
    high = gray - ndimage.gaussian_filter(gray, sigma=1.4)
    local_var = ndimage.uniform_filter(high * high, size=31)
    return {"mean": float(local_var.mean()), "p95": float(np.percentile(local_var, 95))}


def metrics(path: str) -> dict:
    gray = load_gray(path)
    return {"slope": radial_slope(gray), **noise_variance(gray)}
