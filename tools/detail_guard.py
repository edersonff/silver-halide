"""Guard metric: detail retention. Buying histogram bands by destroying
texture is the v20 disease; this forbids it."""

import sys

import numpy as np
from PIL import Image
from scipy import ndimage


def hf_map(path):
    g = np.asarray(Image.open(path).convert("L"), dtype=np.float64) / 255.0
    hp = g - ndimage.gaussian_filter(g, 1.5)
    return g, hp, np.sqrt(ndimage.uniform_filter(hp * hp, size=16))


def detail_retention(src_path, out_path):
    _, _, r0 = hf_map(src_path)
    _, _, r1 = hf_map(out_path)
    rich = r0 >= 0.015
    return float(r1[rich].mean() / r0[rich].mean())


def bands(path):
    _, _, r = hf_map(path)
    edges = [(0, 0.002), (0.002, 0.006), (0.006, 0.015), (0.015, 0.05), (0.05, 1.0)]
    return [float(((r >= lo) & (r < hi)).mean()) for lo, hi in edges]


if __name__ == "__main__":
    src, out = sys.argv[1], sys.argv[2]
    ret = detail_retention(src, out)
    b = bands(out)
    names = ["ultra", "smooth", "wash", "rich", "edge"]
    print(f"detail retention (rich band vs source): {ret*100:.1f}%  [guard: >= 85%]")
    print("bands: " + " ".join(f"{n} {v*100:.1f}%" for n, v in zip(names, b)))
