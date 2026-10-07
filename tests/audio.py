"""Synthetic audio for tests: the one place that builds an ffmpeg ``lavfi`` command line.

``render(path, 10)`` writes ten seconds of a 440 Hz sine. Each segment may instead be a
``(source, seconds)`` pair, where ``source`` is a lavfi source such as ``anoisesrc=seed=11``
(the helper adds the duration, and refuses a source that sets its own); two or more segments
are concatenated, so they must share a sample rate and channel layout, which each source string
sets itself. Output options (codec, channels, rate, a cut) go in ``args``, and the output format
follows ``path``'s extension.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Sequence
from pathlib import Path

SINE = "sine=frequency=440"

Segment = float | tuple[str, float]

# A source that names its own duration (``d=60``, ``duration=60``); a segment has one, its seconds.
_HAS_DURATION = re.compile(r"[=:](?:d|duration)=")


def render(path: Path, *segments: Segment, args: Sequence[str] = ()) -> Path:
    """Render ``segments`` to ``path`` with ffmpeg and return ``path``.

    A bare number is that many seconds of ``SINE``. Raises ``ValueError`` for no segments or a
    source that sets its own duration, and ``CalledProcessError`` if ffmpeg fails.
    """
    if not segments:
        raise ValueError("render() needs at least one segment")
    cmd = ["ffmpeg", "-nostdin", "-y", "-loglevel", "error"]
    for segment in segments:
        source, seconds = (SINE, segment) if isinstance(segment, (int, float)) else segment
        if _HAS_DURATION.search(source):
            raise ValueError(f"{source!r} sets its own duration; pass the seconds in the segment")
        source += f"{':' if '=' in source else '='}duration={seconds}"
        cmd += ["-f", "lavfi", "-i", source]
    if len(segments) > 1:
        inputs = "".join(f"[{i}]" for i in range(len(segments)))
        cmd += ["-filter_complex", f"{inputs}concat=n={len(segments)}:v=0:a=1"]
    cmd += [*args, str(path)]
    subprocess.run(cmd, check=True, stdin=subprocess.DEVNULL)
    return path
