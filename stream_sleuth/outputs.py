"""Where identifications go: the ``Output`` protocol, the HTTP POST output, and JSONL."""

import json
import sys
import urllib.request
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import NoReturn, Protocol, runtime_checkable

from .config import API_SECRET, API_URL, OUTPUT, OUTPUT_PATH
from .paths import CHECKOUT, inside_checkout


@runtime_checkable
class Output(Protocol):
    def emit(self, track: Mapping[str, object]) -> object:
        """Deliver one identification; the return value is logged.

        ``track`` is a ``Mapping`` so an ``Identification`` (a TypedDict, which is
        not a ``dict`` to mypy) and a plain ``dict`` from ``parse()`` both fit.
        """


class HttpPostOutput(Output):
    """POSTs each identification to the ingest API (``STREAM_SLEUTH_SHAZAM_API``)."""

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


def select_output() -> tuple[Output, str]:
    """The output ``STREAM_SLEUTH_OUTPUT`` names, and where it delivers to, for the banner.

    Refuses to run (stderr, exit 1) when the chosen output is incomplete: the HTTP
    output needs the shared secret, with WXDU's original message. JSONL needs an
    absolute path outside the checkout, so a run started from the checkout
    (``run.sh`` ``cd``s there) or under launchd (working directory ``/``) never
    writes a result store into the repo; the file is then opened for append once,
    creating it if absent, so an unwritable path fails at startup rather than on
    every emission.
    """
    if OUTPUT == "http":
        if not API_SECRET:
            _refuse("WXDU_SHAZAM_SECRET is not set")
        return HttpPostOutput(), API_URL
    if OUTPUT == "jsonl":
        if not OUTPUT_PATH:
            _refuse("STREAM_SLEUTH_OUTPUT_PATH is not set")
        path = Path(OUTPUT_PATH)
        if not path.is_absolute():
            _refuse(f"STREAM_SLEUTH_OUTPUT_PATH must be an absolute path, not {OUTPUT_PATH!r}")
        if inside_checkout(path):
            _refuse(f"STREAM_SLEUTH_OUTPUT_PATH {path} is inside the checkout {CHECKOUT}")
        try:
            path.open("a", encoding="utf-8").close()
        except OSError as e:
            _refuse(f"cannot append to STREAM_SLEUTH_OUTPUT_PATH {path} ({e.strerror})")
        return JsonlOutput(OUTPUT_PATH), OUTPUT_PATH
    _refuse(f"STREAM_SLEUTH_OUTPUT must be http or jsonl, not {OUTPUT!r}")


def _refuse(reason: str) -> NoReturn:
    print(f"{reason}; refusing to run.", file=sys.stderr)
    sys.exit(1)


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
