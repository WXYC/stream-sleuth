"""Clip addresses on the 15 s grid, and ephemeral clips cut from hour files with codec emulation.

Station-neutral: reads only hour files, never a flowsheet or archive client. A
clip is addressed by ``(hour key, grid offset, capture length, codec profile)``;
recognizer results are stored under that address, so a clip is cut when a
recognizer needs it and deleted as soon as the caller is done with it. Each cut
re-encodes to the profile's constant-bitrate MP3 (the live mount is 128 kbps),
and can decode that back to the mono 16 kHz WAV the live capture produces.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

GRID_S = 15
CAPTURE_LENGTHS_S = (6, 12, 20)
PROFILES = ("128k", "320k")
HOUR_S = 3600

_KEY = re.compile(r"^(?P<hour>.+)#(?P<offset>\d+)\+(?P<length>\d+)@(?P<profile>[^#+@]+)$")


class ClipError(RuntimeError):
    """ffmpeg failed to cut or decode a clip."""


@dataclass(frozen=True)
class ClipAddress:
    """Where a clip comes from and how it was encoded; ``str()`` is its stable store key."""

    hour_key: str
    offset_s: int
    length_s: int
    profile: str = "128k"

    def __post_init__(self) -> None:
        if self.offset_s < 0 or self.offset_s % GRID_S:
            raise ValueError(f"offset {self.offset_s} s is not on the {GRID_S} s grid")
        if self.length_s not in CAPTURE_LENGTHS_S:
            raise ValueError(f"capture length {self.length_s} s is not one of {CAPTURE_LENGTHS_S}")
        if self.profile not in PROFILES:
            raise ValueError(f"codec profile {self.profile!r} is not one of {PROFILES}")

    def __str__(self) -> str:
        return f"{self.hour_key}#{self.offset_s}+{self.length_s}@{self.profile}"

    @classmethod
    def parse(cls, key: str) -> ClipAddress:
        """Inverse of ``str()``."""
        match = _KEY.match(key)
        if match is None:
            raise ValueError(f"{key!r} is not a clip address")
        return cls(match["hour"], int(match["offset"]), int(match["length"]), match["profile"])


def grid(
    hour_key: str, length_s: int, profile: str = "128k", hour_s: int = HOUR_S
) -> list[ClipAddress]:
    """Every grid address in an hour whose clip ends inside it (240 for 6 s and 12 s, 239 for 20 s)."""
    return [
        ClipAddress(hour_key, o, length_s, profile) for o in range(0, hour_s - length_s + 1, GRID_S)
    ]


def _ffmpeg(*args: str) -> None:
    try:
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", *args], check=True, capture_output=True, timeout=120
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        detail = (exc.stderr or b"").decode(errors="replace").strip() or str(exc)
        raise ClipError(f"ffmpeg {' '.join(args)}: {detail}") from exc


@contextmanager
def cut(
    address: ClipAddress, hour_path: Path, work_dir: Path, *, wav: bool = False
) -> Iterator[Path]:
    """Cut ``address`` from ``hour_path`` under ``work_dir`` and yield its path; it is deleted on exit.

    ``work_dir`` belongs under the data directory, never the checkout. Each cut
    gets its own temporary directory, so concurrent cuts never collide.
    """
    with tempfile.TemporaryDirectory(dir=work_dir, prefix="clip-") as tmp:
        mp3 = Path(tmp) / "clip.mp3"
        _ffmpeg(
            "-ss",
            str(address.offset_s),
            "-t",
            str(address.length_s),
            "-i",
            str(hour_path),
            "-c:a",
            "libmp3lame",
            "-b:a",
            address.profile,
            str(mp3),
        )
        if not wav:
            yield mp3
            return
        decoded = Path(tmp) / "clip.wav"
        _ffmpeg("-i", str(mp3), "-ac", "1", "-ar", "16000", str(decoded))
        yield decoded
