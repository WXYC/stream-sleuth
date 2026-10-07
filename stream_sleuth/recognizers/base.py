"""The ``Recognizer`` protocol and the identification types it returns."""

from typing import Literal, Protocol, TypedDict, runtime_checkable

IdentificationSource = Literal["shazam", "local"]
"""Which recognizer produced an emission: ``"shazam"`` (the Shazam client) or ``"local"`` (the Olaf index)."""


class Identification(TypedDict, total=True):
    """Exactly the four keys the ingest API receives."""

    artist: str
    song: str
    album: str
    label: str


class EvalIdentification(Identification, total=False):
    """An identification plus what the evaluation harness records about it.

    Every field is optional. They are for the evaluation harness, but nothing strips them: an
    ``Output`` serializes whatever mapping it is handed, so an ``EvalIdentification`` passed to
    ``HttpPostOutput`` sends them to the ingest API too. A live deploy that should send only the four
    wire keys needs a recognizer that returns a plain ``Identification``.
    """

    at: float
    """Seconds from the start of the hour file at which the clip (grid run) or capture (replay)
    begins; the origin and unit of ``plays.jsonl``'s ``t_offset_s`` and ``window_*_s``."""

    source: IdentificationSource
    """Which recognizer produced this emission."""

    confidence: float
    """Recognizer-specific and not comparable across sources: Olaf's ``match_count``; absent for Shazam."""

    query_offset_s: float
    """Seconds into the query clip at which the match begins."""

    ref_start_s: float
    """Seconds into the reference recording at which the match begins."""

    ref_key: str
    """The reference's pool stage id (``evaluation.pool.stage_id`` of its object key, a sha1), not the
    object key itself; joins to ``pool.db``'s ``files.stage_id``. Absent for Shazam."""


@runtime_checkable
class Recognizer(Protocol):
    def recognize(self, wav_path: str) -> Identification | None:
        """Identify the audio in ``wav_path``, or return ``None`` on no match."""
