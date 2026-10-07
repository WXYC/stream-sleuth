"""The result store and its read side, and the table of legs, shared by the leg runner and the scorer.

Station-neutral. :class:`ResultStore` is the append-only JSONL file both recognizers' results
are filed in, keyed by clip address and recognizer identity; :func:`read_results` reads stores
back as :class:`Emission` records for the scorer. :data:`LEGS` is the study's definition of
which recognizer runs which capture length over which hours, and :func:`select_legs` and
:func:`require_snapshot` are the CLI checks that read it. This module imports neither the
leg runner nor ``shazamio``, so a reader of stores does not load the runner or Shazam's client.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections import Counter
from collections.abc import Iterable
from collections.abc import Set as AbstractSet
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

from evaluation.clips import CAPTURE_LENGTHS_S, ClipAddress
from stream_sleuth.paths import require_outside_checkout
from stream_sleuth.recognizers.base import EvalIdentification, IdentificationSource
from stream_sleuth.recognizers.olaf import DEFAULT_MIN_MATCH_COUNT
from stream_sleuth.recognizers.olaf import recognizer_identity as olaf_identity

if TYPE_CHECKING:
    from evaluation.shazam_eval import ShazamOutcome

log = logging.getLogger(__name__)

Key = tuple[str, str]  # a clip address and the recognizer identity it was queried with
SCORING_KINDS = frozenset({"matched", "no_match"})
# Retries after an address's first non-scoring outcome; past them it is reported, not queried.
MAX_RETRIES = 3
# Statuses that stop the day, with the reason: 429 is load, 403 is access policy. Neither is
# the address's fault, so neither counts toward its retries.
STOP_REASONS = {429: "rate_limited", 403: "forbidden"}
# The recognizer identity's prefix -> the emission's source.
SOURCES: dict[str, IdentificationSource] = {"shazam": "shazam", "olaf": "local"}


# The stored recognizer identity names this version, never the installed one: a new version is a
# new identity, so the whole corpus would be queried again. Change it with the owner's go-ahead;
# it must equal uv.lock's (a test checks).
SHAZAMIO_VERSION = "0.8.1"


def recognizer_identity(segment_s: int) -> str:
    """The store's recognizer key: the pinned shazamio version and the fingerprinted length."""
    return f"shazam@{SHAZAMIO_VERSION}, segment={segment_s}"


class ResultStore:
    """Append-only JSONL of Shazam outcomes keyed by clip address and recognizer identity."""

    def __init__(self, path: Path) -> None:
        self.path = require_outside_checkout(path)

    def records(self) -> list[dict[str, Any]]:
        """Every whole record; a line torn by a crash is logged and skipped, never rewritten."""
        if not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        # Decoded per line, so a crash inside a multibyte character tears only that line.
        for line in filter(None, self.path.read_bytes().split(b"\n")):
            try:
                records.append(json.loads(line.decode("utf-8")))
            except ValueError:  # includes UnicodeDecodeError
                log.warning("skipped a torn line in %s: %.60r", self.path, line)
        return records

    def _ends_mid_line(self) -> bool:
        try:
            with open(self.path, "rb") as f:
                f.seek(-1, os.SEEK_END)
                return f.read(1) != b"\n"
        except OSError:  # missing or empty
            return False

    def history(self) -> tuple[set[Key], dict[Key, int], set[Key]]:
        """The scored keys, each tried key's latest status (ordered by its latest record), and
        the exhausted keys: the one tally that ``run`` and the report share.

        A key is exhausted after a first failure and ``MAX_RETRIES`` failed retries, where a
        failure is a non-scoring outcome that is not a day-stopping status.
        """
        scored, last, failures = set[Key](), dict[Key, int](), Counter[Key]()
        for r in self.records():
            key = (r["address"], r["recognizer"])
            last.pop(key, None)  # re-inserted, so ``last`` is ordered by each key's latest record
            last[key] = r["status"]
            if r["kind"] in SCORING_KINDS:
                scored.add(key)
            elif r["status"] not in STOP_REASONS:
                failures[key] += 1
        return scored, last, {key for key, n in failures.items() if n > MAX_RETRIES}

    def append(self, address: str, recognizer: str, outcome: ShazamOutcome) -> None:
        record = {
            "address": address,
            "recognizer": recognizer,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            **asdict(outcome),
        }
        lead = "\n" if self._ends_mid_line() else ""  # never glue onto a torn line
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(lead + json.dumps(record, ensure_ascii=False) + "\n")


class Emission(NamedTuple):
    """One matched record: its store key (address and recognizer identity, so legs at one
    offset stay distinct), the parsed address (its ``hour_key`` is plays.jsonl's), and the answer."""

    key: Key
    address: ClipAddress
    found: EvalIdentification


class Results(NamedTuple):
    """What :func:`read_results` found: the matched answers, the keys with a scoring record
    (``matched`` or ``no_match``), and the tried-but-unscored keys with their latest kind."""

    emissions: list[Emission]
    scored: set[Key]
    uncovered: dict[Key, str]


class _Snapshot(ResultStore):
    """A store's records read once, so ``history()`` and the emissions share one snapshot."""

    def __init__(self, store: ResultStore, records: list[dict[str, Any]]) -> None:
        self.path, self._records = store.path, records

    def records(self) -> list[dict[str, Any]]:
        return self._records


@dataclass(frozen=True)
class Leg:
    """One capture length and codec profile over the ``all`` or ``subset`` hours, per recognizer."""

    name: str
    length_s: int
    profile: str
    hours: str
    recognizers: tuple[str, ...]


# The study's query budget. Shazam runs 12 s on the full corpus, then 6 s, 20 s, and 12 s @320k
# on the four-hour subset, in that order. Olaf is free, so it runs every length on the full
# corpus and 12 s @320k on the subset (plan 5.2). The two 12 s Shazam legs file results under
# one recognizer identity; only the address's @profile tells them apart, so a reader keys on it.
LEGS = (
    Leg("12s", 12, "128k", "all", ("shazam", "olaf")),
    Leg("6s", 6, "128k", "all", ("olaf",)),
    Leg("20s", 20, "128k", "all", ("olaf",)),
    Leg("6s-subset", 6, "128k", "subset", ("shazam",)),
    Leg("20s-subset", 20, "128k", "subset", ("shazam",)),
    Leg("12s-320k-subset", 12, "320k", "subset", ("shazam", "olaf")),
)


def to_identification(record: dict[str, Any]) -> Emission | None:
    """A matched record as an :class:`Emission`; None for every other kind.

    ``at`` is the address's grid offset. A Shazam answer sets ``query_offset_s = 0`` and
    ``ref_start_s`` from its stored offset, or neither when it stored none (the two are set
    together, and that answer is left out of lag estimation); an Olaf answer carries its own.
    """
    if record["kind"] != "matched":
        return None
    address = ClipAddress.parse(record["address"])
    source = SOURCES[record["recognizer"].partition("@")[0]]
    found: EvalIdentification = {
        "artist": record["artist"],
        "song": record["song"],
        "album": record["album"],
        "label": record["label"],
        "at": float(address.offset_s),
        "source": source,
    }
    if source == "local":
        found["confidence"] = record["confidence"]
        found["query_offset_s"] = record["query_offset_s"]
        found["ref_start_s"] = record["ref_start_s"]
        found["ref_key"] = record["ref_key"]
    elif record["offset_s"] is not None:
        found["query_offset_s"] = 0.0
        found["ref_start_s"] = record["offset_s"]
    return Emission((record["address"], record["recognizer"]), address, found)


def study_identities(
    snapshot: str | None, min_match_count: int = DEFAULT_MIN_MATCH_COUNT
) -> set[str]:
    """Every recognizer identity one study scores: each Shazam segment length and the Olaf snapshot.

    With no ``snapshot`` there is no Olaf identity, for a run that scores Shazam alone. The Shazam
    identities name the pinned shazamio version (``SHAZAMIO_VERSION``), so they match the stored
    records; :func:`read_results` still warns about records under any other identity.
    """
    identities = {recognizer_identity(n) for n in CAPTURE_LENGTHS_S}
    return identities | {olaf_identity(snapshot, min_match_count)} if snapshot else identities


def _line_of(path: Path, address: str) -> int:
    """The first line of ``path`` whose record has this ``address`` (0 if none), for an error."""
    for n, line in enumerate(path.read_bytes().split(b"\n"), 1):
        try:
            if json.loads(line.decode("utf-8")).get("address") == address:
                return n
        except (ValueError, AttributeError):
            continue
    return 0


def read_results(stores: Iterable[ResultStore], identities: AbstractSet[str]) -> Results:
    """The matched answers, scored keys, and uncovered keys in the union of ``stores``.

    Only records under ``identities`` are read. Per key the first ``matched`` or ``no_match``
    record is the answer, in store order; a later scoring record for it is logged and ignored.
    Scored keys include ``no_match`` ones, so a miss and an address never tried differ. A key is
    uncovered when it has records but none scoring (a 429, a 5xx, a decode error), and its value
    is its latest kind: missing data, never a miss. Which addresses were never tried needs the
    expected grid, which is the scorer's. Each store is read once, so a live run appending
    meanwhile cannot make its tallies and its emissions disagree.

    A record filed under another identity, such as another match floor or shazamio version, is
    not read, and a warning counts those per identity, so it never reads as zero coverage. A
    record whose address does not parse raises ``ValueError`` naming the store and line.

    ``identities`` is a set, never a bare ``str``: ``in`` on a string is a substring test, and
    Olaf identities nest (``min=1`` is inside ``min=12``), so a string raises ``TypeError``.
    """
    if isinstance(identities, str):
        raise TypeError(f"identities must be a set of identities, not the string {identities!r}")
    emissions: list[Emission] = []
    scored: set[Key] = set()
    tried: dict[Key, str] = {}  # key -> latest kind
    answered: set[Key] = set()
    outside: Counter[str] = Counter()
    for store in stores:
        records = store.records()
        store_scored, _, _ = _Snapshot(store, records).history()
        scored |= {k for k in store_scored if k[1] in identities}
        for r in records:
            key = (r["address"], r["recognizer"])
            if key[1] not in identities:
                outside[key[1]] += 1
                continue
            tried[key] = r["kind"]
            try:
                ClipAddress.parse(key[0])
            except ValueError as e:
                raise ValueError(f"{store.path}:{_line_of(store.path, key[0])}: {e}") from e
            if r["kind"] not in SCORING_KINDS:
                continue
            if key in answered:
                log.warning("ignored a later scoring record for %s under %s", *key)
            else:
                answered.add(key)
                emission = to_identification(r)
                if emission:
                    emissions.append(emission)
    uncovered = {k: kind for k, kind in tried.items() if k not in scored}
    if outside:
        log.warning(
            "%d record(s) under identities outside the requested set were not read: %s",
            sum(outside.values()),
            dict(outside),
        )
    return Results(emissions, scored, uncovered)


def select_legs(
    parser: argparse.ArgumentParser,
    only: str | None,
    names: Iterable[str] | None,
) -> tuple[list[Leg], bool, bool]:
    """The legs a run selects, and whether it uses Shazam and Olaf; the parser refuses (exit 2)
    when none are selected.

    ``only`` is one of :data:`SOURCES` or None for both; ``names`` are leg names or None for all.
    """
    use = {only} if only else set(SOURCES)
    legs = [leg for leg in LEGS if (not names or leg.name in names) and use & set(leg.recognizers)]
    if not legs:
        parser.error("no leg is selected: --only and --legs name no leg in common")
    use_shazam = "shazam" in use and any("shazam" in leg.recognizers for leg in legs)
    use_olaf = "olaf" in use and any("olaf" in leg.recognizers for leg in legs)
    return legs, use_shazam, use_olaf


def require_snapshot(parser: argparse.ArgumentParser, use_olaf: bool, snapshot: str | None) -> None:
    """The parser refuses (exit 2) an Olaf leg without a snapshot to query or score."""
    if use_olaf and not snapshot:
        parser.error("--snapshot is required for Olaf legs; pass --only shazam to skip them")
