import io
import os

import numpy as np
from PIL import Image

MARKER = "silver-halide"


class Encoder:
    """Phone-style JPEG: quality 94, 4:2:0 chroma subsampling, honest software marker."""

    def __init__(self, quality: int = 94) -> None:
        self.quality = quality

    def save(self, gamma_img: np.ndarray, path: str) -> None:
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        arr = (np.clip(gamma_img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        image = Image.fromarray(arr, "RGB")
        exif = Image.Exif()
        exif[0x0131] = MARKER
        image.save(path, "JPEG", quality=self.quality, subsampling=2, exif=exif.tobytes())

    def encoded(self, gamma_img: np.ndarray) -> bytes:
        buffer = io.BytesIO()
        arr = (np.clip(gamma_img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        image = Image.fromarray(arr, "RGB")
        image.save(buffer, "JPEG", quality=self.quality, subsampling=2)
        return buffer.getvalue()


def was_processed(path: str) -> bool:
    try:
        exif = Image.open(path).getexif()
    except OSError:
        return False
    return exif.get(0x0131) == MARKER
