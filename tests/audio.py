"""Synthetic audio for tests: the one place that builds an ffmpeg ``lavfi`` command line.

``render(path, 10)`` writes ten seconds of a 440 Hz sine. Each segment may instead be a
``(source, seconds)`` pair, where ``source`` is a lavfi source such as ``anoisesrc=seed=11``
(the helper adds the duration); two or more segments are concatenated, so they must share a
sample rate and channel layout, which each source string sets itself. Output options (codec,
channels, rate, a cut) go in ``args``, and the output format follows ``path``'s extension.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Sequence
from pathlib import Path

# ffmpeg's synthetic-source input format; the tests that check the command line read it here.
FORMAT = "lavfi"
SINE = "sine=frequency=440"

Segment = float | tuple[str, float]

# A source that already names its duration (``d=60``, ``duration=60``) keeps it.
_HAS_DURATION = re.compile(r"[=:](?:d|duration)=")


def render(path: Path, *segments: Segment, args: Sequence[str] = ()) -> Path:
    """Render ``segments`` (default: one 440 Hz sine) to ``path`` with ffmpeg and return ``path``.

    A bare number is that many seconds of ``SINE``. Raises ``CalledProcessError`` if ffmpeg fails.
    """
    cmd = ["ffmpeg", "-nostdin", "-y", "-loglevel", "error"]
    for segment in segments:
        source, seconds = (SINE, segment) if isinstance(segment, (int, float)) else segment
        if not _HAS_DURATION.search(source):
            source = f"{source}:duration={seconds}"
        cmd += ["-f", FORMAT, "-i", source]
    if len(segments) > 1:
        inputs = "".join(f"[{i}]" for i in range(len(segments)))
        cmd += ["-filter_complex", f"{inputs}concat=n={len(segments)}:v=0:a=1"]
    cmd += [*args, str(path)]
    subprocess.run(cmd, check=True, stdin=subprocess.DEVNULL)
    return path
