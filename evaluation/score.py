"""Score recognizer emissions against ``plays.jsonl``: attribution, coverage, recall, precision, lag.

Station-neutral. It reads ``plays.jsonl``, the result stores through
:func:`evaluation.run.read_results`, and replay emissions files in the format ``JsonlOutput``
writes; it never reads a flowsheet, a database, or station config, and never imports
:mod:`evaluation.corpus` or :mod:`evaluation.archive` (the station import scan enforces it).

Both inputs become :class:`evaluation.run.Emission` (store key, parsed address, identification)
before :func:`attribute`, so attribution, precision, and per-play scores run on one code path;
only coverage differs (:func:`score_leg` computes a grid leg's, and the replay reports its own).
An emission is **correct** when :func:`evaluation.names.title_tier` is not None for it and a
candidate play, a play whose window holds the emission's ``at``, under the in-pool join's rules:
the play stands where the join's play does and the emission where its pool file does. Among correct
candidates it goes to the play whose logged interval holds ``at``, else the earliest; a correct
emission outside that interval is also a **neighbor** match. A local (Olaf) emission names every
artist tag its reference file carries when :func:`attribute` is given ``references``
(:func:`evaluation.pool.reference_artists`), as the join does, so a match on a co-credited file is
correct by either tag; a Shazam emission names its one artist. A play's logged interval runs from
its ``t_offset_s`` to the next play's in its hour, the last to 3,600 s.

Audio without a scoring record is **uncovered**, never a miss: a play counts in recall only when
every grid address of the leg that starts in its window has a ``matched`` or ``no_match``
record, and a carryover play is a candidate only, never scored.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple, cast

from evaluation import names
from evaluation.clips import ClipAddress, grid
from evaluation.run import Emission, Leg, Results
from stream_sleuth.recognizers.base import EvalIdentification

HOUR_S = 3600.0
MIN_LAG_SAMPLES = 20
DEFAULT_PAD_S = 180.0  # the plan's default pad, which a lag sample may raise and never lowers
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
    pad_s: float
    carryover: bool
    in_pool: bool | None
    pool_match_tier: str | None
    group: str | None
    band: str
    subset: bool


class Verdict(NamedTuple):
    """An emission and the play it is attributed to (None: no candidate joins, a wrong emission)."""

    emission: Emission
    play: Play | None
    neighbor: bool


class PlayScore(NamedTuple):
    """One scored play. ``first_s`` is its first correct emission's ``at``. For a covered play
    only (else both are None), ``ttfi_s`` is ``first_s`` less the song's start, never below 0,
    and ``lag_s`` the logged offset less the song's start. The start is ``at + query_offset_s - ref_start_s`` of the
    play's earliest correct emission that carries both offsets, as Phase 1 measured the lag
    that set the pads; a Shazam answer with a null offset is skipped, and with none left both
    are None (``first_s`` and ``t_offset_s`` still give an approximate time)."""

    play: Play
    covered: bool
    identified: bool
    first_s: float | None
    ttfi_s: float | None
    lag_s: float | None


class Pad(NamedTuple):
    """An era's recommended pad: the plan's 180 s default unless the nearest-rank
    95th-percentile absolute lag exceeds it, then that, capped at 600 s. ``p95_s`` is None
    (insufficient data, 180 s stands) below 20 lag samples. ``window_pads_s`` are the distinct
    ``pad_s`` its plays' windows were written with, for comparison: the lags were measured inside
    those windows, so a lag past one is censored, not counted."""

    recommended_s: float
    samples: int
    p95_s: float | None
    window_pads_s: tuple[float, ...]


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

    @property
    def recall(self) -> float | None:
        return recall(self.plays)

    @property
    def in_pool_recall(self) -> float | None:
        """Recall over covered plays with ``in_pool: true``; a ``null`` one is never in it."""
        return recall([r for r in self.plays if r.play.in_pool is True])

    @property
    def unjoinable_plays(self) -> int:
        """Covered plays with no title key (no artist or title left after normalization), which
        no emission can join: recall counts them as misses, as the plan defines it."""
        return sum(
            r.covered and not names.title_keys(r.play.artist, r.play.title) for r in self.plays
        )


_RECORD_FIELDS = [f for f in Play._fields if f != "logged_end_s"]
_REQUIRED = ("address", "at", "source", "artist", "song", "album", "label")


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

    Every record must be a JSON object carrying ``address`` (a canonical clip address, which names
    the hour), ``at`` (that address's grid offset), ``source``, and the wire keys, or
    ``ValueError`` names the line: a defaulted ``at`` would attribute the emission to the hour's
    first play. ``emitted_at`` is a wall-clock
    stamp, not an hour offset, and is dropped with ``address``, so ``found`` holds what a stored
    answer's would.
    """
    emissions = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            found = json.loads(line)
        except ValueError as e:
            raise ValueError(f"{path}:{n}: not JSON: {e}") from e
        try:
            if not isinstance(found, dict):
                raise ValueError("the line is not a JSON object")
            if missing := [f for f in _REQUIRED if f not in found]:
                raise ValueError(f"the record has no {' or '.join(missing)}")
            if not isinstance(found["address"], str):
                raise ValueError(f"address {found['address']!r} is not a string")
            if type(found["at"]) not in (int, float):
                raise ValueError(f"at {found['at']!r} is not a number")
            address = ClipAddress.parse(found.pop("address"))
            if found["at"] != address.offset_s:
                raise ValueError(f"at {found['at']} is not its address's offset")
        except ValueError as e:
            raise ValueError(f"{path}:{n}: {e}") from e
        found.pop("emitted_at", None)
        emissions.append(
            Emission((address.key, recognizer), address, cast(EvalIdentification, found))
        )
    return emissions


def _logged(play: Play, at: float) -> bool:
    return play.t_offset_s <= at < play.logged_end_s


def _artists(
    f: EvalIdentification, references: Mapping[str, tuple[str, ...]] | None
) -> tuple[str, ...]:
    """The artist names an emission names: its own, plus for a local match every name its reference carries.

    The emission's own name stays, so a mapping built from another ``pool.db`` than the one that
    tagged the emission can never score a match worse than without it."""
    if f["source"] == "local" and references and f.get("ref_key", "") in references:
        return (f["artist"], *references[f["ref_key"]])
    return (f["artist"],)


def attribute(
    plays: Sequence[Play],
    emissions: Iterable[Emission],
    references: Mapping[str, tuple[str, ...]] | None = None,
) -> list[Verdict]:
    """Each emission's verdict, in ``(hour, at)`` order.

    ``references`` is :func:`evaluation.pool.reference_artists`'s stage id to artist names. A
    local emission whose ``ref_key`` is in it names all of them, so a co-credited file is correct
    when the play joins it by either tag; any other emission names its one artist.
    """
    by_hour: defaultdict[str, list[Play]] = defaultdict(list)
    for p in plays:
        by_hour[p.hour_key].append(p)
    verdicts = []
    for e in sorted(emissions, key=lambda x: (x.address.hour_key, x.found["at"])):
        f, at = e.found, e.found["at"]
        artists = _artists(f, references)
        joined = [
            p
            for p in by_hour[e.address.hour_key]
            if p.window_start_s <= at <= p.window_end_s
            and names.title_tier(p.artist, p.album, p.title, artists, f["album"], f["song"])
            is not None
        ]
        play = min(joined, key=lambda p: (not _logged(p, at), p.t_offset_s), default=None)
        verdicts.append(Verdict(e, play, play is not None and not _logged(play, at)))
    return verdicts


def precision(verdicts: Sequence[Verdict]) -> float | None:
    """Flowsheet precision: the share of emissions attributed to a play."""
    return sum(v.play is not None for v in verdicts) / len(verdicts) if verdicts else None


def distinct_precision(verdicts: Sequence[Verdict]) -> float | None:
    """Precision over runs of consecutive identical emissions, as the loop, which emits only on
    change, would show them; a run is correct when any of it is. Identical is ``loop.step``'s
    key, ``(artist.lower(), song.lower())``, within one hour: the replay drives each hour
    with a fresh loop, so a song across the top of the hour starts a second run."""
    runs: list[list[Verdict]] = []
    for v in verdicts:
        if runs and _song(runs[-1][-1]) == _song(v):
            runs[-1].append(v)
        else:
            runs.append([v])
    return sum(any(v.play for v in run) for run in runs) / len(runs) if runs else None


def _song(v: Verdict) -> tuple[str, str, str]:
    f = v.emission.found
    return v.emission.address.hour_key, f["artist"].lower(), f["song"].lower()


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
        hits = sorted(found[p], key=lambda h: h["at"])
        first = hits[0]["at"] if hits else None
        timed = [h for h in hits if "query_offset_s" in h and "ref_start_s" in h]
        start = (
            timed[0]["at"] + timed[0]["query_offset_s"] - timed[0]["ref_start_s"] if timed else None
        )
        ttfi = lag = None
        if first is not None and start is not None and p in covered:
            ttfi, lag = max(0.0, first - start), p.t_offset_s - start
        rows.append(PlayScore(p, p in covered, bool(hits), first, ttfi, lag))
    return rows


def recall(rows: Sequence[PlayScore]) -> float | None:
    """Per-play recall over covered plays; None when no play is covered."""
    covered = [r for r in rows if r.covered]
    return sum(r.identified for r in covered) / len(covered) if covered else None


def pads(rows: Iterable[PlayScore]) -> dict[str, Pad]:
    """Each era's :class:`Pad` from its plays' lags."""
    lags: defaultdict[str, list[float]] = defaultdict(list)
    windows: defaultdict[str, set[float]] = defaultdict(set)
    for r in rows:
        windows[r.play.era].add(r.play.pad_s)
        lags[r.play.era] += [] if r.lag_s is None else [abs(r.lag_s)]
    result = {}
    for era, v in lags.items():
        p95 = sorted(v)[math.ceil(0.95 * len(v)) - 1] if len(v) >= MIN_LAG_SAMPLES else None
        pad = DEFAULT_PAD_S if p95 is None else max(DEFAULT_PAD_S, min(MAX_PAD_S, p95))
        result[era] = Pad(pad, len(v), p95, tuple(sorted(windows[era])))
    return result


def score_leg(
    plays: Sequence[Play],
    results: Results,
    recognizer: str,
    leg: Leg,
    hours: Sequence[str],
    addresses: Sequence[ClipAddress],
    references: Mapping[str, tuple[str, ...]] | None = None,
) -> LegScore:
    """Score one leg (``recognizer`` at ``leg``'s capture length and profile) over ``hours``.

    This is the grid path, over ``read_results``'s stores; a replay file mixes capture lengths
    (the 6 s / 12 s cadence), so it is scored with :func:`attribute`, the precisions, and
    :func:`score_plays` over the replay's own covered plays, never here.

    ``addresses`` is the leg's grid as ``clips.hour_addresses`` returns it: each hour's clips that
    fit its decoded length, nothing for a skipped hour. A play is uncovered when a grid address
    of a full hour that starts in its window is missing from ``addresses`` (its hour was skipped,
    or the window runs past the hour's last address) or has no scoring record. ``references``
    is passed to :func:`attribute`.
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
    verdicts = attribute(plays, [e for e in results.emissions if in_leg(e.key)], references)
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
