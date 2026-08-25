"""Generation-defect locator: finds content-level generative tells.

Painted skin (real pores absent), painterly hair masses, general texture
deficit. Thresholds calibrated against real iPhone captures.
"""

import json

import numpy as np
from PIL import Image
from scipy import ndimage

W = np.array([0.2126, 0.7152, 0.0722])


def _masks(rgb):
    y = rgb @ W
    r, g, b = rgb[..., 0] * 255, rgb[..., 1] * 255, rgb[..., 2] * 255
    cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
    skin = (cb > 77) & (cb < 127) & (cr > 133) & (cr < 177) & (y > 0.15)
    hair = (y < 0.16) & (np.abs(cr - 128) < 22)
    return y, skin, hair


def _rms_map(y):
    hp = y - ndimage.gaussian_filter(y, 1.5)
    return np.sqrt(ndimage.uniform_filter(hp * hp, size=16))


def audit(path, overlay=None):
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
    y, skin, hair = _masks(rgb)
    rms = _rms_map(y)
    total = y.size

    fine = y - ndimage.gaussian_filter(y, 0.8)
    m2 = ndimage.uniform_filter(fine * fine, 24) + 1e-12
    m4 = ndimage.uniform_filter(fine**4, 24)
    kurt = m4 / (m2 * m2)

    painted = skin & (kurt < 3.6)
    flat_hair = hair & (rms < 0.0016)
    deficit = (rms < 0.0012) & ~skin & ~hair

    report = {
        "painted_skin_share": float(painted.sum() / total),
        "painted_skin_of_skin": float(painted.sum() / max(skin.sum(), 1)),
        "skin_median_kurtosis": float(np.median(kurt[skin])) if skin.any() else None,
        "flat_hair_share": float(flat_hair.sum() / total),
        "texture_deficit_share": float(deficit.sum() / total),
        "skin_share": float(skin.sum() / total),
        "note": "painted_skin_of_skin: >0.35 suggests generated/retouched skin (real-photo false-positive ~0.10); calibrated on iPhone captures, heuristic",
    }
    if overlay:
        tint = rgb.copy()
        tint[painted] = tint[painted] * 0.6 + np.array([0.9, 0.1, 0.1]) * 0.4
        tint[flat_hair] = tint[flat_hair] * 0.6 + np.array([0.1, 0.3, 0.9]) * 0.4
        Image.fromarray((np.clip(tint, 0, 1) * 255).astype(np.uint8)).save(overlay)
    return report


if __name__ == "__main__":
    import sys

    overlay = sys.argv[2] if len(sys.argv) > 2 else None
    print(json.dumps(audit(sys.argv[1], overlay), indent=1))
