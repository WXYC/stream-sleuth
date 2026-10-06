"""The Olaf adapter against a real Olaf build: store two references, find a cut of one.

Needs ``STREAM_SLEUTH_OLAF_BIN`` (or ``olaf`` on PATH) and ``ffmpeg``, which Olaf
decodes with; the ``olaf`` CI job provides both. Seeded pink noise stands in for
music: its spectral peaks are dense and repeatable, which is all Olaf's landmarks need.
"""

from __future__ import annotations

import importlib
import os
import subprocess

import pytest

pytestmark = pytest.mark.olaf

# Read before fresh_recognizer clears every STREAM_SLEUTH_* variable.
OLAF_BIN = os.environ.get("STREAM_SLEUTH_OLAF_BIN") or "olaf"


def render(path, seed):
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"anoisesrc=d=60:c=pink:seed={seed}:a=0.5",
         "-ac", "1", "-ar", "16000", str(path)],
        check=True,
    )  # fmt: skip
    return path


@pytest.fixture
def olaf(fresh_recognizer):
    fresh_recognizer(WXDU_SHAZAM_SECRET="not-a-real-secret")
    return importlib.import_module("stream_sleuth.recognizers.olaf")


def test_a_128_kbps_cut_of_a_stored_reference_is_found_at_its_offset(olaf, tmp_path):
    molina = render(tmp_path / "molina.wav", seed=11)
    pratt = render(tmp_path / "pratt.wav", seed=22)
    recognizer = olaf.OlafRecognizer(tmp_path / "snapshot", olaf_bin=OLAF_BIN)
    recognizer.store(
        [(str(molina), "juana-molina-la-paradoja"), (str(pratt), "jessica-pratt-back-baby")]
    )

    clip = tmp_path / "clip.mp3"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-ss", "20", "-t", "12", "-i", str(molina),
         "-c:a", "libmp3lame", "-b:a", "128k", str(clip)],
        check=True,
    )  # fmt: skip
    result = recognizer.recognize(str(clip))

    assert result is not None
    assert result["ref_key"] == "juana-molina-la-paradoja"
    # The clip starts 20 s into the reference: reference_start - query_start recovers it.
    assert result["ref_start_s"] - result["query_offset_s"] == pytest.approx(20.0, abs=1.0)
    assert result["confidence"] >= olaf.DEFAULT_MIN_MATCH_COUNT
    assert (tmp_path / "snapshot" / ".olaf" / "db" / "data.mdb").exists()


def test_index_build_cli_fills_a_snapshot_a_query_can_read(fresh_recognizer, tmp_path):
    fresh_recognizer(WXDU_SHAZAM_SECRET="not-a-real-secret")
    cli = importlib.import_module("stream_sleuth.cli")
    olaf = importlib.import_module("stream_sleuth.recognizers.olaf")
    ref = render(tmp_path / "gutierrez.wav", seed=33)
    home = tmp_path / "snapshot"
    argv = [
        "index",
        "build",
        "--home",
        str(home),
        "--olaf-bin",
        OLAF_BIN,
        str(ref),
        "hermanos-gutierrez",
    ]
    assert cli.main(argv) == 0
    result = olaf.OlafRecognizer(home, olaf_bin=OLAF_BIN).recognize(str(ref))
    assert result["ref_key"] == "hermanos-gutierrez"
