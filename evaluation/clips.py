"""Clip addresses on the 15 s grid, and ephemeral clips cut from hour files with codec emulation.

Station-neutral: reads only hour files, never a flowsheet or archive client. A
clip is addressed by ``(hour key, grid offset, capture length, codec profile)``;
recognizer results are stored under that address, so a clip is cut when a
recognizer needs it and deleted as soon as the caller is done with it. Each cut
re-encodes to the profile's constant-bitrate MP3 (the live mount is 128 kbps),
and can decode that back to the mono 16 kHz WAV the live capture produces.

A clip never runs past the end of its hour: :func:`grid` holds only the offsets
whose clip ends by the hour's end, and :func:`cut` refuses any other address.
"""

from __future__ import annotations

import functools
import logging
import re
import subprocess
import tempfile
from collections import Counter
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from stream_sleuth.paths import require_outside_checkout

log = logging.getLogger(__name__)

GRID_S = 15
CAPTURE_LENGTHS_S = (6, 12, 20)
PROFILES = ("128k", "320k")
HOUR_S = 3600

# The key's separators, which an hour key may therefore not contain.
_SEPARATORS = "#+@"
_KEY = re.compile(r"(?P<hour>.+)#(?P<offset>[0-9]+)\+(?P<length>[0-9]+)@(?P<profile>.+)", re.ASCII)


class ClipError(RuntimeError):
    """A clip cannot be cut: its hour is unreadable or too short, or ffmpeg failed."""


@dataclass(frozen=True)
class ClipAddress:
    """Where a clip comes from and how it was encoded; :attr:`key` is its store key.

    The key is ``<hour key>#<offset>+<length>@<profile>`` with the offset and
    length as plain decimal integers, so every address has exactly one key and
    :meth:`parse` accepts that key and no other spelling.
    """

    hour_key: str
    offset_s: int
    length_s: int
    profile: str = "128k"

    def __post_init__(self) -> None:
        if type(self.hour_key) is not str:
            raise TypeError(f"hour key {self.hour_key!r} is not a str")
        if not self.hour_key or not self.hour_key.isprintable():
            raise ValueError(f"hour key {self.hour_key!r} is empty or holds a control character")
        if any(sep in self.hour_key for sep in _SEPARATORS):
            raise ValueError(f"hour key {self.hour_key!r} holds a key separator ({_SEPARATORS})")
        for name in ("offset_s", "length_s"):
            if type(getattr(self, name)) is not int:
                raise TypeError(f"{name} {getattr(self, name)!r} is not an int")
        if self.offset_s < 0 or self.offset_s % GRID_S:
            raise ValueError(f"offset {self.offset_s} s is not on the {GRID_S} s grid")
        if self.length_s not in CAPTURE_LENGTHS_S:
            raise ValueError(f"capture length {self.length_s} s is not one of {CAPTURE_LENGTHS_S}")
        if self.profile not in PROFILES:
            raise ValueError(f"codec profile {self.profile!r} is not one of {PROFILES}")

    @property
    def key(self) -> str:
        """The stable store key, ``<hour key>#<offset>+<length>@<profile>``."""
        return f"{self.hour_key}#{self.offset_s}+{self.length_s}@{self.profile}"

    def __str__(self) -> str:
        return self.key

    @classmethod
    def parse(cls, key: str) -> ClipAddress:
        """Inverse of :attr:`key`; raises ``ValueError`` for any string that is not a canonical key."""
        match = _KEY.fullmatch(key)
        if match is None:
            raise ValueError(f"{key!r} is not a clip address")
        address = cls(match["hour"], int(match["offset"]), int(match["length"]), match["profile"])
        if address.key != key:
            raise ValueError(f"{key!r} is not the canonical spelling {address.key!r}")
        return address


def grid(
    hour_key: str, length_s: int, profile: str = "128k", hour_s: float = HOUR_S
) -> list[ClipAddress]:
    """Every 15 s offset from 0 whose clip ends by ``hour_s``.

    A full 3,600 s hour holds 240 clips of 6 s or 12 s and 239 of 20 s (a 20 s
    clip at 3,585 s would end at 3,605 s). For an hour file that may be short,
    pass ``hour_s=hour_duration(path)`` so the grid holds only clips :func:`cut` accepts.
    """
    return [
        ClipAddress(hour_key, o, length_s, profile)
        for o in range(0, int(hour_s) + 1, GRID_S)
        if o + length_s <= hour_s
    ]


def _ffmpeg(*args: str) -> bytes:
    try:
        done = subprocess.run(
            ["ffmpeg", "-v", "error", "-y", *args],
            check=True,
            capture_output=True,
            stdin=subprocess.DEVNULL,
            timeout=120,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        detail = (exc.stderr or b"").decode(errors="replace").strip() or str(exc)
        raise ClipError(f"ffmpeg {' '.join(args)}: {detail}") from exc
    return done.stdout


def hour_duration(hour_path: Path) -> float:
    """The decoded length of ``hour_path`` in seconds, measured once per version of the file.

    Decodes the whole file (about 2 s for an hour of MP3), as a recognizer would
    hear it; the container's duration can count encoder padding or be missing.
    """
    try:
        stat = hour_path.stat()
    except OSError as exc:
        raise ClipError(f"cannot read hour {hour_path}: {exc}") from exc
    return _decoded_seconds(str(hour_path.resolve()), stat.st_size, stat.st_mtime_ns)


@functools.lru_cache(maxsize=256)
def _decoded_seconds(path: str, size: int, mtime_ns: int) -> float:
    progress = _ffmpeg(
        "-i", f"file:{path}", "-map", "0:a:0", "-f", "null", "-progress", "pipe:1", "-"
    ).decode(errors="replace")
    times = [
        line.partition("=")[2] for line in progress.splitlines() if line.startswith("out_time_us=")
    ]
    if not times or not times[-1].isdigit():
        raise ClipError(f"ffmpeg decoded no audio from {path}")
    return int(times[-1]) / 1_000_000


def hour_addresses(
    keys: Iterable[str], archive_dir: Path, length_s: int, profile: str
) -> list[ClipAddress]:
    """Every clip address that fits each hour's decoded length, in ``keys`` order.

    An hour that is missing or unreadable is logged and contributes nothing, so
    one bad hour never stops the run; a short hour yields only the clips that fit.
    A repeated key is used once, at its first position, with one warning naming it.
    """
    addresses: list[ClipAddress] = []
    counts = Counter(keys)
    for key, n in counts.items():
        if n > 1:
            log.warning("hour %s is listed %d times; using it once", key, n)
        try:
            hour_s = hour_duration(archive_dir / key)
        except ClipError as exc:
            log.warning("skipped hour %s: %s", key, exc)
            continue
        addresses += grid(key, length_s, profile, hour_s=hour_s)
    return addresses


@contextmanager
def cut(
    address: ClipAddress, hour_path: Path, work_dir: Path, *, wav: bool = False
) -> Iterator[Path]:
    """Cut ``address`` from ``hour_path`` under ``work_dir`` and yield its path; it is deleted on exit.

    ``work_dir`` must be absolute and outside the checkout (``DataPathError``
    otherwise, before anything is created). Raises :class:`ClipError` when the
    clip would run past the hour's decoded end, rather than yield a short or
    empty clip. Each cut gets its own temporary directory, so concurrent cuts
    never collide.
    """
    require_outside_checkout(work_dir)
    end_s = hour_duration(hour_path)
    if address.offset_s + address.length_s > end_s:
        raise ClipError(f"{address.key} runs past the end of {hour_path} ({end_s:.3f} s)")
    with tempfile.TemporaryDirectory(dir=work_dir, prefix="clip-") as tmp:
        mp3 = Path(tmp) / "clip.mp3"
        _ffmpeg(
            "-ss",
            str(address.offset_s),
            "-t",
            str(address.length_s),
            "-i",
            f"file:{hour_path}",
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
        _ffmpeg("-i", f"file:{mp3}", "-ac", "1", "-ar", "16000", str(decoded))
        yield decoded
