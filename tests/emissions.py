"""In-memory emissions and leg scores for the score and queue tests.

``emission`` builds what ``results.to_identification`` returns for a record the study's writers
filed (``tests.stores`` is the one way to write such a record); ``leg_scores`` wraps emissions in
the ``LegScore`` mapping the queue functions take. ``tests/unit/test_emissions.py`` pins the
first against the store round trip.
"""

from __future__ import annotations

from typing import Any, cast

from evaluation import score
from evaluation.clips import ClipAddress
from evaluation.results import Emission, recognizer_identity
from stream_sleuth.recognizers.base import EvalIdentification
from stream_sleuth.recognizers.olaf import recognizer_identity as olaf_identity
from tests.plays import HOUR, STAGE, Track

SHAZAM = recognizer_identity(12)
OLAF = olaf_identity("rotation")


def emission(
    track: Track,
    at: float,
    *,
    hour: str = HOUR,
    source: str = "shazam",
    length_s: int = 12,
    ref_key: str | None = None,
    query_offset_s: float | None = None,
    ref_start_s: float | None = None,
) -> Emission:
    """An emission of ``track`` at grid offset ``at`` of ``hour``, as ``read_results`` returns it.

    ``source`` is ``"shazam"`` or ``"local"`` (Olaf). An Olaf one carries ``confidence``, a
    ``ref_key`` (``STAGE`` unless given), and both offsets (0.0 unless given); a Shazam one carries
    both offsets or neither, and a ``ref_key`` only when given. The two offsets are set together.
    """
    offsets = {"query_offset_s": query_offset_s, "ref_start_s": ref_start_s}
    given = {k: v for k, v in offsets.items() if v is not None}
    if len(given) == 1:
        raise ValueError("set both offsets or neither: query_offset_s and ref_start_s")
    artist, song, album = track
    found: dict[str, Any] = {"artist": artist, "song": song, "album": album, "label": ""}
    found |= {"at": at, "source": source}
    if source == "local":
        found |= {"confidence": 40.0, "ref_key": ref_key or STAGE} | {
            "query_offset_s": 0.0,
            "ref_start_s": 0.0,
        }
    elif ref_key is not None:
        found["ref_key"] = ref_key
    found |= given
    address = ClipAddress(hour, int(at), length_s)
    identity = OLAF if source == "local" else recognizer_identity(length_s)
    return Emission((address.key, identity), address, cast(EvalIdentification, found))


def leg_scores(
    plays: list[score.Play],
    emissions: dict[str, list[Emission]],
    references: dict[str, tuple[str, ...]] | None = None,
) -> dict[str, score.LegScore]:
    """One ``LegScore`` per tag (``"12s/shazam"``), holding that tag's verdicts against ``plays``.

    Its coverage and per-play scores are empty: the queue functions read only the verdicts.
    """
    return {
        tag: score.LegScore(cast(Any, None), score.attribute(plays, found, references), [])
        for tag, found in emissions.items()
    }
