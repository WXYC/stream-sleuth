"""Where audio comes from: the ``Source`` protocol and the live-stream source."""

import subprocess
from typing import Protocol, runtime_checkable

from .config import STREAM_URL


@runtime_checkable
class Source(Protocol):
    def capture(self, path: str, seconds: int) -> None:
        """Write ``seconds`` of audio to ``path`` as a mono 16 kHz wav."""


class IcecastSource(Source):
    """The station's live stream (``WXDU_STREAM_URL``), captured with ``ffmpeg``."""

    def capture(self, path: str, seconds: int) -> None:
        capture(path, seconds)


def capture(path, seconds):
    """Grab `seconds` of the stream into a small mono 16kHz wav via ffmpeg."""
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            STREAM_URL,
            "-t",
            str(seconds),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "wav",
            path,
        ],
        check=True,
        timeout=seconds + 25,
        stdin=subprocess.DEVNULL,
    )
