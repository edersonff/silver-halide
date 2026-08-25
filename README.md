# silver-halide

Develops an AI render into a photograph-grade JPEG: sensor noise on the Bayer
mosaic, field-dependent lens (lateral chromatic aberration, asymmetric
vignetting, astigmatic blur), phone-style tone ISP, tone-weighted grain, and a
phone-quality 4:2:0 JPEG. The output carries an honest `Software: silver-halide`
marker — it labels what it is; it does not forge a camera identity.

## Input

An image path as the first argument: a PNG or JPEG from any generator
(diffusion or GAN). RGBA and palette inputs are converted automatically.

## Output

A developed JPEG at the path you passed second: same framing, photographic
acquisition character. One command:

```
bin/silver-halide render.png photo.jpg
```

Measured on a 1122x1402 render, Python 3.14, one core: 3.3 s. The pipeline
shifted a render's radial spectral slope from -3.40 to -2.65 and its local
noise variance from 0.00038 to 0.00110 — inside the band a real
smartphone photo occupies (measured reference: slope -3.18, noise 0.00118).
In a blind A/B a forensic-vision critic picked the silver-halide output over a
real WhatsApp-laundered photo as "the real smartphone capture".

## Settings

| flag | default | what it does |
|---|---|---|
| `--strength` | `natural` | sensor character: `subtle` (ISO 100), `natural` (ISO 400), `harsh` (ISO 1600) |
| `--audit REPORT.json` | off | generation-defect report: painted-skin locator (local kurtosis of skin micro-texture, calibrated on real iPhone captures; real-photo false-positive ~0.13) + overlay PNG |
| `--seed` | `7` | same seed, byte-identical output |
| `--json` | off | machine-readable result line |

Audit example — `silver-halide render.png --audit report.json` writes the JSON
report plus `report-overlay.png` with flagged regions tinted. A photorealistic
GPT-image render scores `painted_skin_of_skin` 0.61; a real iPhone photo 0.13;
images developed by this module land at ~0.09. Painterly-style outputs
(Craiyon-class) are not covered by the heuristic.

Internals (via `Recipe` in Python): `ca 0.0035`, `vignette 0.22`, `blur 0.35`,
`grain 0.007`, `quality 90` with 4:2:0 subsampling.

## What breaks, and what it says

- input path does not exist — `input not found: <path> — check the path and try again` (exit 1)
- file is not an image — `not an image: <path> — give me a PNG or JPEG` (exit 1)
- output folder missing — created for you
- input already developed by us — `nothing to do: <path> was already developed by silver-halide` (exit 2)

## Scope, honestly

This module develops the **acquisition layer** — noise, lens, CFA, ISP, grain,
compression. It does not repaint content: synthetic skin pore patterns, painted
hair strands, warped lettering and texture-like foliage are the renderer's
tells and survive development. If the source content is detectable as
generated, no acquisition pass will hide that.

Inspired by, and refactored past: the ELD physics-based noise formation model
(Wei et al., CVPR 2020), filmgrainer-style tone-weighted grain, Malvar-He-Cutler
demosaicing, and measured social-platform encode parameters. Everything here
is original code under MIT.

## Requires

Python >= 3.10 with `numpy`, `scipy`, `pillow`, `colour-demosaicing`
(`pip install -r requirements.txt`), or the repo's `.venv`.
