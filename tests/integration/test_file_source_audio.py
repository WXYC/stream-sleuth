"""``FileSource`` against a real ``ffmpeg``: the clip has the live source's shape.

End-of-file behavior is pinned here: near the end the clip is short, and at or
past the end it is a valid WAV with no frames. The source never raises for it;
the replay knows each hour's length and stops its clock there.
"""

from __future__ import annotations

import importlib
import wave
from dataclasses import dataclass

import pytest

from tests.audio import render

pytestmark = pytest.mark.ffmpeg

TONE_SECONDS = 10


@dataclass
class FixedClock:
    offset: float

    def now(self) -> float:
        return self.offset


@pytest.fixture
def sources(fresh_recognizer):
    fresh_recognizer(WXDU_SHAZAM_SECRET="not-a-real-secret")
    return importlib.import_module("stream_sleuth.sources")


@pytest.fixture(scope="module")
def tone(tmp_path_factory):
    path = tmp_path_factory.mktemp("audio") / "Jessica Pratt - Back, Baby.mp3"
    return render(path, TONE_SECONDS, args=["-ac", "2", "-ar", "44100"])


@pytest.mark.parametrize(
    ("offset", "seconds", "expected"),
    [
        (0.0, 3, 3.0),
        (4.5, 3, 3.0),
        (8.5, 3, 1.5),  # a short clip near the end
        (TONE_SECONDS, 3, 0.0),  # an empty clip at the end
        (TONE_SECONDS + 5, 3, 0.0),  # and past it
    ],
)
def test_clip_is_mono_16khz_and_as_long_as_the_file_allows(
    sources, tone, tmp_path, offset, seconds, expected
):
    out = tmp_path / "clip.wav"

    sources.FileSource(tone, FixedClock(offset)).capture(str(out), seconds)

    with wave.open(str(out)) as wav:
        assert wav.getnchannels() == 1
        assert wav.getframerate() == 16000
        assert wav.getnframes() / wav.getframerate() == pytest.approx(expected, abs=0.1)
