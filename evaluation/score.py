"""Score recognizer emissions against ``plays.jsonl``: attribution, coverage, recall, precision, lag.

Station-neutral. It reads ``plays.jsonl``, the result stores through
:func:`evaluation.run.read_results`, and replay emissions files in the format ``JsonlOutput``
writes; it never reads a flowsheet, a database, or station config, and never imports
:mod:`evaluation.corpus` or :mod:`evaluation.archive` (the station import scan enforces it).

Both inputs become :class:`evaluation.run.Emission` (store key, parsed address, identification)
before attribution, so attribution and every figure run on one code path. An emission is
**correct** when :func:`evaluation.names.title_tier` is not None for it and a candidate play, a play whose window holds the emission's ``at``, by the in-pool join's rules: the
play stands where the join's play does and the emission where its pool file does. Among correct
candidates it goes to the play whose logged interval holds ``at``, else the earliest; a correct
emission outside that interval is also a **neighbor** match. A play's logged interval runs from
its ``t_offset_s`` to the next play's in its hour, the last to 3,600 s.

Audio without a scoring record is **uncovered**, never a miss: a play counts in recall only when
every grid address of the leg that starts in its window has a ``matched`` or ``no_match``
record, and a carryover play is a candidate only, never scored.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

from evaluation import names
from evaluation.clips import ClipAddress, grid
from evaluation.run import Emission, Leg, Results
from stream_sleuth.recognizers.base import EvalIdentification

HOUR_S = 3600.0
MIN_LAG_SAMPLES = 20
MAX_PAD_S = 600.0
UNTRIED = "untried"  # a grid address with no record at all


class Play(NamedTuple):
    """A ``plays.jsonl`` record's fields the scorer reads, plus its logged interval's end."""

    hour_key: str
    play_id: int
    t_offset_s: float
    logged_end_s: float
    window_start_s: float
    window_end_s: float
    artist: str | None
    album: str | None
    title: str | None
    era: str
    carryover: bool


class Verdict(NamedTuple):
    """An emission and the play it is attributed to (None: no candidate joins, a wrong emission)."""

    emission: Emission
    play: Play | None
    neighbor: bool


class PlayScore(NamedTuple):
    """One scored play. ``ttfi_s`` (covered plays only) is its first correct emission's ``at``
    less the song's start, ``lag_s`` its logged offset less the song's start; both are None when
    none of its emissions carries ``query_offset_s`` and ``ref_start_s``. The start is the median
    of those emissions' ``at + query_offset_s - ref_start_s``."""

    play: Play
    covered: bool
    identified: bool
    ttfi_s: float | None
    lag_s: float | None


@dataclass(frozen=True)
class Coverage:
    """One leg's coverage. ``uncovered_addresses`` counts the grid addresses with no scoring
    record by their latest kind (``untried`` for none); ``uncovered_plays`` is per hour."""

    grid_addresses: int
    scored_addresses: int
    uncovered_addresses: dict[str, int]
    hours_without_records: list[str]
    uncovered_plays: dict[str, int]


@dataclass(frozen=True)
class LegScore:
    coverage: Coverage
    verdicts: list[Verdict]
    plays: list[PlayScore]


_RECORD_FIELDS = [f for f in Play._fields if f != "logged_end_s"]


def plays_from(records: Iterable[dict[str, Any]]) -> list[Play]:
    """``plays.jsonl`` records as :class:`Play`, sorted by hour and offset, each with its interval."""
    by_hour: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_hour[r["hour_key"]].append(r)
    plays = []
    for _, rows in sorted(by_hour.items()):
        rows.sort(key=lambda r: r["t_offset_s"])
        ends = [r["t_offset_s"] for r in rows[1:]] + [HOUR_S]
        for r, end in zip(rows, ends, strict=True):
            plays.append(Play(**{f: r[f] for f in _RECORD_FIELDS}, logged_end_s=end))
    return plays


def read_plays(path: Path) -> list[Play]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return plays_from(json.loads(line) for line in lines if line.strip())


def load_emissions(path: Path, recognizer: str) -> list[Emission]:
    """A replay's emissions file (UTF-8 JSONL, as ``JsonlOutput`` writes it), keyed under ``recognizer``.

    Every record must carry ``address`` (a canonical clip address, which names the hour), ``at``
    (that address's grid offset), and ``source``, or ``ValueError`` names the line: a defaulted
    ``at`` would attribute the emission to the hour's first play. ``emitted_at`` is a wall-clock
    stamp, not an hour offset, and is dropped with ``address``, so ``found`` holds what a stored
    answer's would.
    """
    emissions = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        found = json.loads(line)
        try:
            if missing := [f for f in ("address", "at", "source") if f not in found]:
                raise ValueError(f"the record has no {' or '.join(missing)}")
            address = ClipAddress.parse(found.pop("address"))
            if found["at"] != address.offset_s:
                raise ValueError(f"at {found['at']} is not its address's offset")
        except ValueError as e:
            raise ValueError(f"{path}:{n}: {e}") from e
        found.pop("emitted_at", None)
        emissions.append(Emission((address.key, recognizer), address, found))
    return emissions


def _logged(play: Play, at: float) -> bool:
    return play.t_offset_s <= at < play.logged_end_s


def attribute(plays: Sequence[Play], emissions: Iterable[Emission]) -> list[Verdict]:
    """Each emission's verdict, in ``(hour, at)`` order."""
    by_hour: defaultdict[str, list[Play]] = defaultdict(list)
    for p in plays:
        by_hour[p.hour_key].append(p)
    verdicts = []
    for e in sorted(emissions, key=lambda x: (x.address.hour_key, x.found["at"])):
        f, at = e.found, e.found["at"]
        joined = [
            p
            for p in by_hour[e.address.hour_key]
            if p.window_start_s <= at <= p.window_end_s
            and names.title_tier(p.artist, p.album, p.title, (f["artist"],), f["album"], f["song"])
            is not None
        ]
        play = min(joined, key=lambda p: (not _logged(p, at), p.t_offset_s), default=None)
        verdicts.append(Verdict(e, play, play is not None and not _logged(play, at)))
    return verdicts


def precision(verdicts: Sequence[Verdict]) -> float | None:
    """Flowsheet precision: the share of emissions attributed to a play."""
    return sum(v.play is not None for v in verdicts) / len(verdicts) if verdicts else None


def distinct_precision(verdicts: Sequence[Verdict]) -> float | None:
    """Precision over runs of consecutive identical emissions (same hour, artist, and song), as
    the loop, which emits only on change, would show them; a run is correct when any of it is."""
    runs: list[list[Verdict]] = []
    for v in verdicts:
        if runs and _song(runs[-1][-1]) == _song(v):
            runs[-1].append(v)
        else:
            runs.append([v])
    return sum(any(v.play for v in run) for run in runs) / len(runs) if runs else None


def _song(v: Verdict) -> tuple[str, str, str]:
    return v.emission.address.hour_key, v.emission.found["artist"], v.emission.found["song"]


def score_plays(
    plays: Sequence[Play], verdicts: Iterable[Verdict], covered: AbstractSet[Play]
) -> list[PlayScore]:
    """A :class:`PlayScore` for each play that is not a carryover."""
    found: defaultdict[Play, list[EvalIdentification]] = defaultdict(list)
    for v in verdicts:
        if v.play:
            found[v.play].append(v.emission.found)
    rows = []
    for p in plays:
        if p.carryover:
            continue
        hits = found[p]
        starts = [
            h["at"] + h["query_offset_s"] - h["ref_start_s"]
            for h in hits
            if "query_offset_s" in h and "ref_start_s" in h
        ]
        start = statistics.median(starts) if starts else None
        ttfi = None
        if start is not None and p in covered:
            ttfi = max(0.0, min(h["at"] for h in hits) - start)
        lag = None if start is None else p.t_offset_s - start
        rows.append(PlayScore(p, p in covered, bool(hits), ttfi, lag))
    return rows


def recall(rows: Sequence[PlayScore]) -> float | None:
    """Per-play recall over covered plays; None when no play is covered."""
    covered = [r for r in rows if r.covered]
    return sum(r.identified for r in covered) / len(covered) if covered else None


def pads(rows: Iterable[PlayScore]) -> dict[str, float | None]:
    """Per era, the pad that covers the nearest-rank 95th-percentile absolute lag, capped at
    600 s; None (insufficient data) below 20 lag samples."""
    lags: defaultdict[str, list[float]] = defaultdict(list)
    for r in rows:
        lags[r.play.era] += [] if r.lag_s is None else [abs(r.lag_s)]
    return {
        era: min(MAX_PAD_S, sorted(v)[math.ceil(0.95 * len(v)) - 1])
        if len(v) >= MIN_LAG_SAMPLES
        else None
        for era, v in lags.items()
    }


def score_leg(
    plays: Sequence[Play],
    results: Results,
    recognizer: str,
    leg: Leg,
    hours: Sequence[str],
    addresses: Sequence[ClipAddress],
) -> LegScore:
    """Score one leg (``recognizer`` at ``leg``'s capture length and profile) over ``hours``.

    ``addresses`` is the leg's grid as ``clips.hour_addresses`` returns it: each hour's clips that
    fit its decoded length, nothing for a skipped hour. A play is uncovered when a grid address
    of a full hour that starts in its window is missing from ``addresses`` (its hour was skipped,
    or the window runs past the hour's last address) or has no scoring record.
    """

    def in_leg(key: tuple[str, str]) -> ClipAddress | None:
        a = ClipAddress.parse(key[0])
        same = (key[1], a.length_s, a.profile) == (recognizer, leg.length_s, leg.profile)
        return a if same and a.hour_key in leg_hours else None

    leg_hours = set(hours)
    keys = {(a.key, recognizer) for a in addresses}
    scored = keys & results.scored
    recorded = {a.hour_key for k in (*results.scored, *results.uncovered) if (a := in_leg(k))}
    plays = [p for p in plays if p.hour_key in leg_hours]
    covered = set()
    for p in plays:
        inside = [
            (a.key, recognizer)
            for a in grid(p.hour_key, leg.length_s, leg.profile)
            if p.window_start_s <= a.offset_s <= p.window_end_s
        ]
        if all(k in scored for k in inside):
            covered.add(p)
    verdicts = attribute(plays, [e for e in results.emissions if in_leg(e.key)])
    coverage = Coverage(
        grid_addresses=len(keys),
        scored_addresses=len(scored),
        uncovered_addresses=dict(Counter(results.uncovered.get(k, UNTRIED) for k in keys - scored)),
        hours_without_records=[h for h in hours if h not in recorded],
        uncovered_plays=dict(
            Counter(p.hour_key for p in plays if p not in covered and not p.carryover)
        ),
    )
    return LegScore(coverage, verdicts, score_plays(plays, verdicts, covered))
