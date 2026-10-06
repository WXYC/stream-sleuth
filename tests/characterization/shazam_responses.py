"""Shazam response bodies in the shape ``shazamio`` returns from ``Shazam.recognize``.

Trimmed to the fields ``recognizer.parse()`` reads, plus enough of the
surrounding structure (``matches``, extra sections and metadata rows) that a
parser which looked anywhere else would be caught.
"""

from __future__ import annotations

from typing import Any


def match(
    artist: str,
    title: str,
    *,
    album: str | None = None,
    label: str | None = None,
    released: str = "2017",
) -> dict[str, Any]:
    """A matched response with an optional Album and Label metadata row."""
    metadata: list[dict[str, Any]] = []
    if album is not None:
        metadata.append({"title": "Album", "text": album})
    if label is not None:
        metadata.append({"title": "Label", "text": label})
    metadata.append({"title": "Released", "text": released})
    return {
        "matches": [{"id": "1", "offset": 41.2, "timeskew": 0.0, "frequencyskew": 0.0}],
        "track": {
            "key": "1",
            "title": title,
            "subtitle": artist,
            "sections": [
                {"type": "SONG", "metadata": metadata},
                {"type": "VIDEO", "youtubeurl": "https://example.test/video"},
            ],
        },
    }


NO_MATCH: dict[str, Any] = {"matches": [], "tagid": "0", "timestamp": 0}

JUANA_MOLINA = match("Juana Molina", "la paradoja", album="DOGA", label="Sonamos")
JESSICA_PRATT = match(
    "Jessica Pratt", "Back, Baby", album="On Your Own Love Again", label="Drag City"
)
