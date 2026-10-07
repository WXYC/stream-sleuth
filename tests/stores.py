"""Result-store fixtures built through the writers, so a test can only hold what a writer writes.

``shazam_record`` and ``olaf_record`` file one record by way of ``ShazamOutcome`` /
``OlafOutcome`` and ``ResultStore.append``, the same path ``shazam_eval.run`` and
``run_olaf`` take. A test that pins a literal stored line as a format check keeps its literal.
"""

from __future__ import annotations

from typing import Any

from evaluation.results import ResultStore, recognizer_identity
from evaluation.run import OlafOutcome
from evaluation.shazam_eval import ShazamOutcome
from stream_sleuth.recognizers.olaf import recognizer_identity as olaf_identity

Track = tuple[str, str, str]  # artist, song, album
TRACK: Track = ("Juana Molina", "la paradoja", "DOGA")
# The status ``outcome_from`` gives each kind: 429 is load, any other non-2xx is a server error.
STATUS = {"matched": 200, "no_match": 200, "decode_error": 200, "rate_limited": 429}
SERVER_ERROR = 503
OLAF_KINDS = ("matched", "no_match")  # run_olaf files nothing else: an OlafError aborts the run


def shazam_record(
    store: ResultStore,
    address: str,
    kind: str = "matched",
    track: Track = TRACK,
    *,
    identity: str | None = None,
    **extra: Any,
) -> None:
    """File one Shazam outcome; ``track`` names only a ``matched`` one, as ``outcome_from`` does.

    ``extra`` sets any other ``ShazamOutcome`` field (``status``, ``label``, ``offset_s``).
    """
    artist, song, album = track if kind == "matched" else ("", "", "")
    fields = {
        "status": STATUS.get(kind, SERVER_ERROR),
        "artist": artist,
        "song": song,
        "album": album,
    }
    outcome = ShazamOutcome(kind=kind, **{**fields, **extra})
    store.append(address, identity or recognizer_identity(12), outcome)


def olaf_record(
    store: ResultStore,
    address: str,
    kind: str = "matched",
    track: Track = TRACK,
    *,
    identity: str | None = None,
    **extra: Any,
) -> None:
    """File one Olaf outcome: status 0 and a null ``offset_s``, as ``run_olaf`` writes them.

    A ``matched`` one carries a confidence, a reference key, and both offsets; ``extra`` sets
    any of them (``confidence``, ``ref_key``, ``query_offset_s``, ``ref_start_s``, ``label``).
    """
    if kind not in OLAF_KINDS:
        raise ValueError(f"run_olaf files only {OLAF_KINDS}, not {kind!r}")
    if kind == "no_match":
        outcome = OlafOutcome(0, kind, **extra)
    else:
        artist, song, album = track
        fields = {
            "confidence": 40.0,
            "ref_key": "ab" * 20,
            "query_offset_s": 0.0,
            "ref_start_s": 0.0,
        }
        outcome = OlafOutcome(0, kind, artist, song, album, **{**fields, **extra})
    store.append(address, identity or olaf_identity("rotation"), outcome)
