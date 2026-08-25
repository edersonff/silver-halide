import base64
import io

import numpy as np
from PIL import Image

from .spectra import metrics


def _crop_b64(path: str, box: tuple[int, int, int, int]) -> str:
    with Image.open(path) as image:
        crop = image.convert("RGB").crop(box)
    buffer = io.BytesIO()
    crop.save(buffer, "PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def _flat_box(path: str, size: int = 200) -> tuple[int, int, int, int]:
    with Image.open(path) as image:
        w, h = image.size
        data = np.asarray(image.convert("L"), dtype=np.float64)
    best = None
    box = (0, 0, size, size)
    step = max(size // 2, 1)
    for y in range(0, max(h - size, 1), step):
        for x in range(0, max(w - size, 1), step):
            v = float(data[y : y + size, x : x + size].std())
            if best is None or v < best:
                best, box = v, (x, y, x + size, y + size)
    return box


def gallery(before: str, after: str, out: str) -> dict:
    with Image.open(before) as image:
        w, h = image.size
    size = min(220, w // 3, h // 3)
    center = (w // 2 - size, h // 2 - size, w // 2 + size, h // 2 + size)
    flat = _flat_box(before, size)
    edge = (0, 0, size * 2, size * 2)
    rows = []
    for label, box in (("center", center), ("flat/shadow", flat), ("corner (lens)", edge)):
        rows.append(
            f"<tr><th>{label}</th><td><img src='data:image/png;base64,{_crop_b64(before, box)}'></td>"
            f"<td><img src='data:image/png;base64,{_crop_b64(after, box)}'></td></tr>"
        )
    m0, m1 = metrics(before), metrics(after)
    table = (
        f"<tr><th>spectral slope</th><td>{m0['slope']:.2f}</td><td>{m1['slope']:.2f}</td></tr>"
        f"<tr><th>noise mean</th><td>{m0['mean']:.6f}</td><td>{m1['mean']:.6f}</td></tr>"
        f"<tr><th>noise p95</th><td>{m0['p95']:.6f}</td><td>{m1['p95']:.6f}</td></tr>"
    )
    html = f"""<!doctype html><html><head><meta charset='utf-8'><title>silver-halide proof</title>
<style>body{{font-family:system-ui;background:#111;color:#eee;margin:2rem}}
table{{border-collapse:collapse}}td,th{{border:1px solid #333;padding:6px;text-align:center}}
img{{width:100%;image-rendering:pixelated;display:block}}h1{{font-size:1.2rem}}</style></head>
<body><h1>silver-halide — before / after</h1>
<table>{rows}</table>
<h1>measured</h1><table><tr><th>metric</th><th>before</th><th>after</th></tr>{table}</table></body></html>"""
    with open(out, "w") as handle:
        handle.write(html)
    return {"before": m0, "after": m1, "gallery": out}
