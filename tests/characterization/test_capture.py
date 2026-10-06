"""Pins ``recognizer.capture()`` against a real ``ffmpeg``.

``capture()`` passes ``-i <stream URL>`` with no ``-f lavfi``, so the input has
to be a real file or URL: the test renders a tone to a file first and points
``WXDU_STREAM_URL`` at it.
"""

from __future__ import annotations

import subprocess
import wave

import pytest

pytestmark = pytest.mark.ffmpeg


@pytest.mark.parametrize("seconds", [2, 3])
def test_writes_mono_16khz_wav_of_the_requested_length(fresh_recognizer, tmp_path, seconds):
    source = tmp_path / "tone.mp3"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=10",
         "-ac", "2", "-ar", "44100", str(source)],
        check=True,
    )  # fmt: skip
    recognizer = fresh_recognizer(WXDU_SHAZAM_SECRET="x", WXDU_STREAM_URL=str(source))
    output = tmp_path / "capture.wav"

    recognizer.capture(str(output), seconds)

    with wave.open(str(output)) as wav:
        assert wav.getnchannels() == 1
        assert wav.getframerate() == 16000
        assert wav.getnframes() / wav.getframerate() == pytest.approx(seconds, abs=0.05)
