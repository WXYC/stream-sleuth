"""Where audio comes from: the ``Source`` protocol, the live-stream source, and a local-file source."""

import os
import subprocess
from pathlib import Path
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


@runtime_checkable
class Clock(Protocol):
    def now(self) -> float:
        """Seconds from the start of the audio the source reads."""


class FileSource(Source):
    """A local audio file, captured at the offset an injected ``Clock`` gives.

    For replaying recorded audio through the loop. It reads only a local path:
    ``ffmpeg`` gets the input as ``file:<path>``, so no URL can reach the network.
    Near the end of the file the clip is short, and at or past the end it is a
    valid WAV with no frames; the caller knows the file's length and stops there.
    """

    def __init__(self, path: str | os.PathLike[str], clock: Clock) -> None:
        if "://" in os.fspath(path):
            raise ValueError(f"FileSource reads a local file, not a URL: {os.fspath(path)}")
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.clock = clock

    def capture(self, path: str, seconds: int) -> None:
        offset = self.clock.now()
        if offset < 0:
            raise ValueError(f"clock offset must not be negative: {offset}")
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{offset:.3f}", "-i", f"file:{self.path}",
             "-t", str(seconds), "-ac", "1", "-ar", "16000", "-f", "wav", path],
            check=True,
            timeout=seconds + 25,
            stdin=subprocess.DEVNULL,
        )  # fmt: skip


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
