"""Where identifications go: the ``Output`` protocol and the HTTP POST output."""

import json
import urllib.request
from collections.abc import Mapping
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
