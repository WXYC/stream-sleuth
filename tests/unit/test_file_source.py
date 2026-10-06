"""``FileSource``: a local audio file read at the position an injected clock gives.

The recording ``ffmpeg`` stub pins the arguments, so these run without an encoder;
``tests/integration/test_file_source_audio.py`` checks the real output.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass

import pytest

from tests.ffmpeg_stub import FfmpegStub


@dataclass
class FixedClock:
    offset: float

    def now(self) -> float:
        return self.offset


@pytest.fixture
def sources(fresh_recognizer):
    fresh_recognizer(WXDU_SHAZAM_SECRET="not-a-real-secret")
    return importlib.import_module("stream_sleuth.sources")


@pytest.fixture
def ffmpeg(tmp_path, monkeypatch):
    root = tmp_path / "stub"
    root.mkdir()
    stub = FfmpegStub(root).install()
    monkeypatch.setenv("PATH", f"{stub.bin_dir}{os.pathsep}{os.environ['PATH']}")
    return stub


@pytest.fixture
def hour(tmp_path):
    path = tmp_path / "Juana Molina - DOGA.mp3"
    path.write_bytes(b"not decoded by the stub")
    return path


def test_file_source_is_a_source(sources, hour):
    assert isinstance(sources.FileSource(hour, FixedClock(0.0)), sources.Source)


@pytest.mark.parametrize(("offset", "seconds"), [(0.0, 6), (75.5, 12), (3585.0, 12)])
def test_captures_from_the_clocks_offset(sources, ffmpeg, hour, tmp_path, offset, seconds):
    out = tmp_path / "clip.wav"

    sources.FileSource(hour, FixedClock(offset)).capture(str(out), seconds)

    [argv] = ffmpeg.calls()
    assert argv[argv.index("-ss") + 1] == f"{offset:.3f}"
    assert argv[argv.index("-t") + 1] == str(seconds)
    assert argv.index("-ss") < argv.index("-i"), "seek on the input, not by decoding up to it"
    assert argv[argv.index("-i") + 1] == f"file:{hour}"
    assert argv[argv.index("-ac") + 1] == "1"
    assert argv[argv.index("-ar") + 1] == "16000"
    assert argv[-3:] == ["-f", "wav", str(out)]


def test_reads_the_clock_on_every_capture(sources, ffmpeg, hour, tmp_path):
    clock = FixedClock(0.0)
    source = sources.FileSource(hour, clock)

    source.capture(str(tmp_path / "a.wav"), 6)
    clock.offset = 30.0
    source.capture(str(tmp_path / "b.wav"), 6)

    assert [argv[argv.index("-ss") + 1] for argv in ffmpeg.calls()] == ["0.000", "30.000"]


@pytest.mark.parametrize("url", ["https://audio-mp3.ibiblio.org/wxyc.mp3", "s3://bucket/key.mp3"])
def test_refuses_anything_but_a_local_file(sources, url):
    with pytest.raises(ValueError, match="local file"):
        sources.FileSource(url, FixedClock(0.0))


def test_refuses_a_missing_file(sources, tmp_path):
    with pytest.raises(FileNotFoundError):
        sources.FileSource(tmp_path / "absent.mp3", FixedClock(0.0))


def test_refuses_a_negative_offset(sources, ffmpeg, hour, tmp_path):
    with pytest.raises(ValueError, match="offset"):
        sources.FileSource(hour, FixedClock(-1.0)).capture(str(tmp_path / "clip.wav"), 6)
    assert ffmpeg.calls() == []
