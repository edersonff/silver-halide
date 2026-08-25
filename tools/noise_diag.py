import sys

import numpy as np
from PIL import Image
from scipy import ndimage


def noise_fields(path):
    im = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0
    luma = im @ np.array([0.2126, 0.7152, 0.0722])
    high = luma - ndimage.gaussian_filter(luma, 1.5)
    rows = []
    for lo, hi in [(0.05, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 0.97)]:
        m = (luma[2:-2, 2:-2] >= lo) & (luma[2:-2, 2:-2] < hi)
        if m.sum() < 400:
            rows.append((lo, hi, None, None, None))
            continue
        v = high[2:-2, 2:-2][m]
        rows.append((lo, hi, float(v.std()), float(v.skew()) if hasattr(v, "skew") else None, int(m.sum())))
    return rows


def chroma_luma_ratio(path):
    im = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0
    luma = im @ np.array([0.2126, 0.7152, 0.0722])
    resid = im - luma[..., None]
    chroma = np.linalg.norm(resid, axis=-1)
    hp = lambda x: x - ndimage.gaussian_filter(x, 1.5)
    m = (luma > 0.05) & (luma < 0.6)
    return float(hp(chroma)[m].std() / (hp(luma)[m].std() + 1e-12))


def residual_autocorr(path):
    im = np.asarray(Image.open(path).convert("L"), dtype=np.float64) / 255.0
    high = im - ndimage.gaussian_filter(im, 1.5)
    f = np.fft.fft2(high - high.mean())
    power = np.abs(f) ** 2
    h, w = power.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((yy - h / 2) ** 2 + (xx - w / 2) ** 2)
    low = power[(r > 2) & (r < h / 8)].mean()
    highb = power[(r > h / 4) & (r < h / 2.2)].mean()
    return float(np.log10((low + 1e-30) / (highb + 1e-30)))


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(f"== {p.split('/')[-1]}")
        print("  chroma/luma noise ratio:", round(chroma_luma_ratio(p), 3))
        print("  residual LF/HF (log10):", round(residual_autocorr(p), 3))
        for lo, hi, std, _, n in noise_fields(p):
            if std is None:
                print(f"  luma {lo:.2f}-{hi:.2f}: (few pixels)")
            else:
                print(f"  luma {lo:.2f}-{hi:.2f}: noise std {std:.5f}  (n={n})")
