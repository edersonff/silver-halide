import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from flat_noise import flat_noise_stats

from silver_halide.pipeline import Recipe, Develop
from silver_halide.stages.encoder import was_processed

ROOT = Path(__file__).parent
SOURCE = Path("/home/eder/.codex/generated_images/01a03972-9584-70f0-935f-ee41ade98aaf/exec-fcef8f8a-a229-469b-af4e-3702485ab928.png")
OUT = Path("/tmp/opencode/silver_test/test-out.jpg")


def test_develop_matches_real_phone_flat_noise_band():
    Develop(Recipe()).run(str(SOURCE), str(OUT))
    flat = flat_noise_stats(str(OUT))
    assert flat is not None and flat["flat_patches"] > 50
    assert 0.002 <= flat["noise_std_mean"] <= 0.006, flat
    assert flat["lf_hf_log10"] > 0.0, "grain must be fine (HF-weighted), not a lowpass veil"


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
