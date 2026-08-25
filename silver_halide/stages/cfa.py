import numpy as np
from colour_demosaicing import demosaicing_CFA_Bayer_Malvar2004, mosaicing_CFA_Bayer


class Cfa:
    """Bayer RGGB acquisition: mosaic to sensor plane, demosaic back with Malvar."""

    def mosaic(self, img: np.ndarray) -> np.ndarray:
        return mosaicing_CFA_Bayer(np.clip(img, 0.0, 1.0), pattern="RGGB")

    def demosaic(self, mosaic: np.ndarray) -> np.ndarray:
        rgb = demosaicing_CFA_Bayer_Malvar2004(mosaic, pattern="RGGB")
        return np.clip(rgb, 0.0, 1.0)
