import sys

import numpy as np
from PIL import Image
from scipy import ndimage

W = np.array([0.2126, 0.7152, 0.0722])


def load(path):
    return np.asarray(Image.open(path).convert("L"), dtype=np.float64) / 255.0


def hf_energy_map(gray, sigma=1.5, win=16):
    hp = gray - ndimage.gaussian_filter(gray, sigma)
    return np.sqrt(ndimage.uniform_filter(hp * hp, size=win))


def texture_profile(path, bands=((0.002, 0.006), (0.006, 0.015), (0.015, 0.05), (0.05, 1.0))):
    """HF energy stats inside detail-level bands: who lives in each texture class."""
    gray = load(path)
    e = hf_energy_map(gray)
    out = {}
    for lo, hi in bands:
        m = (e >= lo) & (e < hi)
        if m.sum() < 500:
            out[f"{lo}-{hi}"] = None
            continue
        out[f"{lo}-{hi}"] = {"share": float(m.mean()), "energy_mean": float(e[m].mean())}
    return out


def repetition_score(path, patch=128):
    """Peakiness of the autocorrelation of the high-pass field: tiled/repeated
    AI textures show periodic peaks; organic texture is diffuse."""
    gray = load(path)
    hp = gray - ndimage.gaussian_filter(gray, 1.5)
    hp = hp - hp.mean()
    f = np.fft.fft2(hp)
    power = np.abs(f) ** 2
    h, w = power.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((yy - h / 2) ** 2 + (xx - w / 2) ** 2)
    ring = (r > 8) & (r < min(h, w) / 2.5)
    p = power[ring]
    return float((p.max() / (np.median(p) + 1e-30)))


def anisotropy(path):
    """HF energy orientation balance: organic photos spread energy across
    orientations; rendered hair/edges concentrate on dominant directions."""
    gray = load(path)
    hp = gray - ndimage.gaussian_filter(gray, 1.5)
    gx = ndimage.sobel(hp, axis=1)
    gy = ndimage.sobel(hp, axis=0)
    ang = np.arctan2(gy, gx)
    mag = np.sqrt(gx * gx + gy * gy)
    hist, _ = np.histogram(ang[mag > np.percentile(mag, 75)], bins=12, range=(-np.pi, np.pi))
    hist = hist / (hist.sum() + 1e-12)
    return float(-(hist * np.log(hist + 1e-12)).sum() / np.log(12))


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(f"== {p.split('/')[-1]}")
        print(f"  repetition peak/median: {repetition_score(p):8.1f}")
        print(f"  orientation entropy   : {anisotropy(p):8.3f} (1.0 = isotropic)")
        for band, v in texture_profile(p).items():
            if v is None:
                print(f"  hf band {band:>14}: (too few pixels)")
            else:
                print(f"  hf band {band:>14}: share {v['share']*100:5.1f}%  energy {v['energy_mean']:.5f}")
