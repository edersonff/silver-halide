import sys

import numpy as np
from PIL import Image
from scipy import ndimage


def flat_noise_stats(path, patch=64):
    im = np.asarray(Image.open(path).convert("L"), dtype=np.float64) / 255.0
    hp = im - ndimage.gaussian_filter(im, 1.5)
    local = ndimage.uniform_filter(hp * hp, size=9)
    flat = local < np.percentile(local, 12)
    best = np.argwhere(flat)
    if len(best) == 0:
        return None
    h, w = im.shape
    luma = im
    stds, specs, lums = [], [], []
    step = max(len(best) // 400, 1)
    for y, x in best[::step]:
        if y + patch > h or x + patch > w or y < patch or x < patch:
            continue
        win = im[y : y + patch, x : x + patch]
        if win.std() > 0.06:
            continue
        res = hp[y : y + patch, x : x + patch]
        stds.append(res.std())
        lums.append(win.mean())
        f = np.fft.fftshift(np.fft.fft2(res))
        p = np.abs(f) ** 2
        yy, xx = np.mgrid[0:patch, 0:patch]
        r = np.sqrt((yy - patch / 2) ** 2 + (xx - patch / 2) ** 2)
        specs.append((p[(r > 2) & (r < patch / 8)].mean(), p[(r > patch / 4) & (r < patch / 2.2)].mean()))
    if not stds:
        return None
    lf = float(np.mean([s[0] for s in specs]))
    hf = float(np.mean([s[1] for s in specs]))
    order = np.argsort(lums)
    return {
        "flat_patches": len(stds),
        "noise_std_mean": float(np.mean(stds)),
        "lf_hf_log10": float(np.log10((lf + 1e-30) / (hf + 1e-30))),
        "luma_points": [(round(lums[i], 2), round(stds[i], 5)) for i in order[:: max(len(order) // 8, 1)]],
    }


if __name__ == "__main__":
    for p in sys.argv[1:]:
        s = flat_noise_stats(p)
        print(f"== {p.split('/')[-1]}")
        if s is None:
            print("  no flat patches found")
            continue
        print(f"  flat patches: {s['flat_patches']}  noise std: {s['noise_std_mean']:.5f}  LF/HF: {s['lf_hf_log10']:.2f}")
        print("  luma->noise:", s["luma_points"])
