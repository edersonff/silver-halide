import subprocess
import sys
from pathlib import Path

from PIL import Image

import numpy as np

from silver_halide.pipeline import Recipe, Develop
from silver_halide.proof.spectra import metrics
from silver_halide.stages.encoder import was_processed

ROOT = Path(__file__).parent
SOURCE = Path("/home/eder/.codex/generated_images/01a03972-9584-70f0-935f-ee41ade98aaf/exec-fcef8f8a-a229-469b-af4e-3702485ab928.png")
OUT = Path("/tmp/opencode/silver_test/test-out.jpg")


def test_develop_adds_sensor_texture_and_natural_spectrum():
    Develop(Recipe()).run(str(SOURCE), str(OUT))
    before = metrics(str(SOURCE))
    after = metrics(str(OUT))
    assert after["mean"] > before["mean"] * 2.5
    assert after["slope"] > before["slope"]
    assert -3.2 < after["slope"] < -2.0


def test_develop_is_deterministic_per_seed():
    a = Path("/tmp/opencode/silver_test/seed-a.jpg")
    b = Path("/tmp/opencode/silver_test/seed-b.jpg")
    Develop(Recipe(seed=3)).run(str(SOURCE), str(a))
    Develop(Recipe(seed=3)).run(str(SOURCE), str(b))
    assert a.read_bytes() == b.read_bytes()


def test_marker_makes_rerun_exit_2():
    code = subprocess.run(
        [sys.executable, "-m", "silver_halide.cli", str(OUT), "/tmp/opencode/silver_test/again.jpg"],
        capture_output=True,
    ).returncode
    assert code == 2
    assert was_processed(str(OUT))


def test_missing_input_exit_1():
    code = subprocess.run(
        [sys.executable, "-m", "silver_halide.cli", "/nope.png", "/tmp/x.jpg"],
        capture_output=True,
    ).returncode
    assert code == 1
