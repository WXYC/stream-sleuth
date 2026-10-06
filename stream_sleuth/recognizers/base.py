"""The ``Recognizer`` protocol and the identification types it returns."""

from typing import Protocol, TypedDict, runtime_checkable


class Identification(TypedDict, total=True):
    """Exactly the four keys the ingest API receives."""

    artist: str
    song: str
    album: str
    label: str


class EvalIdentification(Identification, total=False):
    """An identification plus what the evaluation harness records about it."""

    at: float
    source: str
    confidence: float
    query_offset_s: float
    ref_start_s: float
    ref_key: str


@runtime_checkable
class Recognizer(Protocol):
    def recognize(self, wav_path: str) -> Identification | None:
        """Identify the audio in ``wav_path``, or return ``None`` on no match."""
