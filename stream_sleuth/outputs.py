"""Where identifications go: the ``Output`` protocol, the HTTP POST output, and JSONL."""

import json
import urllib.request
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, runtime_checkable

from .config import API_SECRET, API_URL


@runtime_checkable
class Output(Protocol):
    def emit(self, track: Mapping[str, object]) -> object:
        """Deliver one identification; the return value is logged.

        ``track`` is a ``Mapping`` so an ``Identification`` (a TypedDict, which is
        not a ``dict`` to mypy) and a plain ``dict`` from ``parse()`` both fit.
        """


class HttpPostOutput(Output):
    """POSTs each identification to the ingest API (``WXDU_SHAZAM_API``)."""

    def emit(self, track: Mapping[str, object]) -> object:
        return post(track)


class JsonlOutput(Output):
    """Appends each identification to ``path`` as one JSON object per line.

    The record is the track as given plus ``emitted_at`` (ISO 8601, UTC by default),
    a name no wire or evaluation key uses. Non-ASCII is written as UTF-8, not
    escaped, so the file reads as it sounds. Each line is flushed before ``emit``
    returns, so a crash loses at most the record being written, and existing
    content is never truncated.
    """

    def __init__(self, path: str | Path, clock: Callable[[], datetime] | None = None):
        self.path = Path(path)
        self.clock = clock

    def emit(self, track: Mapping[str, object]) -> object:
        now = self.clock() if self.clock else datetime.now(timezone.utc)
        line = json.dumps({**track, "emitted_at": now.isoformat()}, ensure_ascii=False)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        return self.path


def post(track):
    """POST a recognized track to the wxdu API ingest endpoint."""
    body = json.dumps(track).encode("utf-8")
    req = urllib.request.Request(
        API_URL,
        data=body,
        headers={
            "Content-Type": "application/json",
            "X-Ingest-Secret": API_SECRET,
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return resp.status
