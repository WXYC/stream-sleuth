"""Score recognizer emissions against ``plays.jsonl``: attribution, coverage, recall, precision, lag.

Station-neutral. It reads ``plays.jsonl``, the result stores through
:func:`evaluation.results.read_results`, and replay emissions files in the format ``JsonlOutput``
writes; it never reads a flowsheet or station config, and never imports
:mod:`evaluation.corpus` or :mod:`evaluation.archive` (the station import scan enforces it). The
scoring functions touch no database: they take the Olaf references' artist names as an argument,
and only the CLI's :func:`main` opens the snapshot's own ``pool.db``, read-only, to build them.

Both inputs become :class:`evaluation.results.Emission` (store key, parsed address, identification)
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

``python -m evaluation.score`` scores every selected leg and writes the score file and the near-miss queue
(:func:`main`); wrong emissions a person should judge are queued, never absorbed.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from contextlib import closing
from dataclasses import asdict, dataclass
from itertools import groupby
from pathlib import Path
from typing import Any, NamedTuple, cast

from evaluation import names
from evaluation.clips import ClipAddress, grid, hour_addresses
from evaluation.olaf_snapshot import RESULTS, SnapshotError, checked_snapshot_dir
from evaluation.pool import open_read_only, reference_artists
from evaluation.results import (
    LEGS,
    SOURCES,
    Emission,
    Leg,
    Results,
    ResultStore,
    read_results,
    recognizer_identity,
    require_snapshot,
    select_legs,
    study_identities,
)
from evaluation.selection import read_hours
from stream_sleuth.paths import DataPathError, data_dir, require_outside_checkout
from stream_sleuth.recognizers.base import EvalIdentification
from stream_sleuth.recognizers.olaf import DEFAULT_MIN_MATCH_COUNT
from stream_sleuth.recognizers.olaf import recognizer_identity as olaf_identity

log = logging.getLogger(__name__)

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
    pool_format: str | None
    rotation: bool
    reorder_flag: bool | None
    play_order_status: str
    talk_rows: int
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


NEAR_MISS_SIMILARITY = 0.8
RUN_COLUMNS = (
    "leg",
    "recognizer",
    "address",
    "last_address",
    "emissions",
    "hour_key",
    "start",
    "end",
    "source",
    "artist",
    "song",
    "album",
    "reference_artists",
)
NEAR_COLUMNS = (
    *RUN_COLUMNS,
    "play_id",
    "play_carryover",
    "play_artist",
    "play_title",
    "play_album",
    "matched_on",
    "verdict",
)
# What the score file says of a play: the fields report.py breaks results down by. A carryover
# repeats the previous hour's ``play_id``, so a play is keyed by (hour_key, play_id, carryover).
PLAY_FIELDS = (
    "hour_key",
    "play_id",
    "carryover",
    "era",
    "in_pool",
    "pool_match_tier",
    "pool_format",
    "rotation",
    "group",
    "band",
    "subset",
    "reorder_flag",
    "play_order_status",
    "talk_rows",
)


def wrong_runs(verdicts: Sequence[Verdict]) -> list[list[Verdict]]:
    """Runs of consecutive wrong emissions of one song (``distinct_precision``'s song key).

    A run also ends at a correct emission, which ``distinct_precision``'s runs do not: a song run
    it counts once can contribute more than one wrong run here.
    """
    groups = groupby(verdicts, lambda v: (v.play is None, _song(v)))
    return [list(run) for (wrong, _), run in groups if wrong]


def _clock(seconds: float) -> str:
    return f"{int(seconds) // 3600}:{int(seconds) // 60 % 60:02d}:{int(seconds) % 60:02d}"


def run_row(tag: str, run: Sequence[Verdict]) -> dict[str, Any]:
    """A queue row for a run: leg, the identity it was scored under, first and last clip address,
    count, and hour-file position, with the first emission's metadata."""
    first, last = run[0].emission, run[-1].emission
    f = first.found
    return {
        "leg": tag,
        "recognizer": first.key[1],
        "address": first.address.key,
        "last_address": last.address.key,
        "emissions": len(run),
        "hour_key": first.address.hour_key,
        "start": _clock(f["at"]),
        "end": _clock(last.found["at"] + last.address.length_s),
        "source": f["source"],
        "artist": f["artist"],
        "song": f["song"],
        "album": f["album"],
    }


def _same(a: str | None, b: str | None) -> bool:
    key = names.fuzzy(a)
    return bool(key) and key == names.fuzzy(b)


def _tokens(artist: str | None, title: str | None) -> set[str]:
    return set(names.fuzzy(f"{artist or ''} {title or ''}").split())


def _nearness(p: Play, f: EvalIdentification, artists: Sequence[str]) -> tuple[list[str], float]:
    """The fields of ``p`` equal to the emission's under the fuzzy key, and the best Jaccard."""
    fields = [
        field
        for field, hit in (
            ("artist", any(_same(p.artist, a) for a in artists)),
            ("title", _same(p.title, f["song"])),
        )
        if hit
    ]
    theirs = _tokens(p.artist, p.title)
    jaccards = [
        len(theirs & mine) / len(theirs | mine) for a in artists if (mine := _tokens(a, f["song"]))
    ]
    return fields, max(jaccards, default=0.0)


def near_miss(
    run: Sequence[Verdict],
    by_hour: Mapping[str, Sequence[Play]],
    references: Mapping[str, tuple[str, ...]] | None,
) -> tuple[Play, str, Emission] | None:
    """The play nearest a run of wrong emissions, why, and the emission that found it.

    Every emission of the run is judged.

    A candidate is a play whose window holds an emission's ``at`` and that shares ``artist`` or
    ``title`` with it (both can hold, as for a version mismatch) under the fuzzy key, else
    ``similarity``: a Jaccard of the token sets of at least 0.8. The best candidate has the most
    shared fields, then the highest similarity, and the earliest emission wins a tie.
    """
    near = []
    for v in run:
        f = v.emission.found
        artists = _artists(f, references)
        for p in by_hour[v.emission.address.hour_key]:
            if p.window_start_s <= f["at"] <= p.window_end_s:
                fields, similarity = _nearness(p, f, artists)
                if fields or similarity >= NEAR_MISS_SIMILARITY:
                    label = "+".join(fields) or "similarity"
                    near.append(((len(fields), similarity), p, label, v.emission))
    return max(near, key=lambda n: n[0])[1:] if near else None


def near_miss_rows(
    plays: Sequence[Play],
    legs: Mapping[str, LegScore],
    references: Mapping[str, tuple[str, ...]] | None,
) -> list[dict[str, Any]]:
    """The near-miss queue: each run of wrong emissions that is a near miss, with its nearest
    play and an empty ``verdict`` (``correct`` or ``wrong``) for a person to fill."""
    by_hour: defaultdict[str, list[Play]] = defaultdict(list)
    for p in plays:
        by_hour[p.hour_key].append(p)
    rows = []
    for tag, ls in legs.items():
        for run in wrong_runs(ls.verdicts):
            if found := near_miss(run, by_hour, references):
                play, matched_on, emission = found
                tags = references.get(emission.found.get("ref_key", ""), ()) if references else ()
                rows.append(
                    {
                        **run_row(tag, run),
                        "reference_artists": " | ".join(tags),
                        "play_id": play.play_id,
                        "play_carryover": play.carryover,
                        "play_artist": play.artist,
                        "play_title": play.title,
                        "play_album": play.album,
                        "matched_on": matched_on,
                        "verdict": "",
                    }
                )
    return rows


def _refusal(path: Path, columns: Sequence[str]) -> str | None:
    """Why the file at ``path`` must not be overwritten by a queue with ``columns``, or None.

    Only an unfilled queue of this command's own is overwritten: its header (matched without
    regard to case, and past a byte-order mark a spreadsheet adds) is ``columns`` and no row has
    a ``verdict``. Anything else, such as a JSONL store, another CSV, an empty file, or a file that
    cannot be read, is not ours to replace.
    """
    try:
        with path.open(encoding="utf-8-sig", newline="") as f:
            rows = csv.reader(f)
            header = next(rows, None)
            if header is None or [h.strip().lower() for h in header] != list(columns):
                return "is not an unfilled queue of this command's"
            verdict = list(columns).index("verdict")
            if any(len(row) > verdict and row[verdict].strip() for row in rows):
                return "has verdicts filled in"
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError, csv.Error):
        return "cannot be read"
    return None


def check_queue(path: Path, columns: Sequence[str] = NEAR_COLUMNS) -> None:
    """``SystemExit``, one line naming ``path``, when :func:`_refusal` says not to overwrite it."""
    if reason := _refusal(path, columns):
        raise SystemExit(f"{path}: {reason}; not overwriting (move it, or pass another path)")


def write_queue(path: Path, columns: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    """Write a queue as UTF-8 CSV with a byte-order mark, so a spreadsheet shows diacritics, and
    its header always, so an empty queue still says what it holds.

    A person's work is never overwritten (:func:`check_queue`): the caller writes nothing else.
    """
    check_queue(path, columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, columns)
        writer.writeheader()
        writer.writerows(rows)


def _play_row(p: Play) -> dict[str, Any]:
    return {f: getattr(p, f) for f in PLAY_FIELDS}


def leg_json(leg: Leg, identity: str, ls: LegScore) -> dict[str, Any]:
    """One leg's entry in the score file, which ``report.py`` reads: figures, coverage, and rows.

    ``plays`` are the scored plays; ``carryover_plays`` the carryover plays some emission was
    attributed to, which are candidates only and never scored. An emission row names its play by
    ``(hour_key, play_id, carryover)``, which keys one of those two lists (``play_id`` and
    ``carryover`` are null for a wrong emission).
    """
    attributed = {v.play for v in ls.verdicts if v.play and v.play.carryover}
    return {
        "leg": leg.name,
        "recognizer": identity,
        "length_s": leg.length_s,
        "profile": leg.profile,
        "coverage": asdict(ls.coverage),
        "recall": ls.recall,
        "in_pool_recall": ls.in_pool_recall,
        "unjoinable_plays": ls.unjoinable_plays,
        "precision": precision(ls.verdicts),
        "distinct_precision": distinct_precision(ls.verdicts),
        "adjudicated_precision": None,
        "adjudicated_distinct_precision": None,
        "pads": {era: p._asdict() for era, p in pads(ls.plays).items()},
        "plays": [
            {
                **_play_row(r.play),
                "covered": r.covered,
                "identified": r.identified,
                "first_s": r.first_s,
                "ttfi_s": r.ttfi_s,
                "lag_s": r.lag_s,
            }
            for r in ls.plays
        ],
        "carryover_plays": [
            _play_row(p) for p in sorted(attributed, key=lambda p: (p.hour_key, p.t_offset_s))
        ],
        "emissions": [
            {
                "address": v.emission.address.key,
                "hour_key": v.emission.address.hour_key,
                "source": v.emission.found["source"],
                "artist": v.emission.found["artist"],
                "song": v.emission.found["song"],
                "album": v.emission.found["album"],
                "play_id": v.play.play_id if v.play else None,
                "carryover": v.play.carryover if v.play else None,
                "neighbor": v.neighbor,
            }
            for v in ls.verdicts
        ],
    }


def check_score_file(path: Path) -> None:
    """``SystemExit`` naming ``path`` unless it is absent or a score file this command wrote.

    A score file is a JSON object of exactly ``version`` and ``legs``; anything else there, such
    as a result store or a settings file, is not ours to replace.
    """
    try:
        found = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return
    except (OSError, ValueError):
        found = None
    if not isinstance(found, dict) or set(found) != {"version", "legs"}:
        raise SystemExit(f"{path}: exists and is not a score file; not overwriting it")


def _same_file(a: Path, b: Path) -> bool:
    """Whether ``a`` and ``b`` are one file, by resolved path or, when both exist, by identity."""
    return a.resolve() == b.resolve() or (a.exists() and b.exists() and os.path.samefile(a, b))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--plays", type=Path, required=True, help="the plays.jsonl to score against"
    )
    parser.add_argument("--selection", type=Path, help="default: <data>/selection.json")
    parser.add_argument("--archive-dir", type=Path, help="default: <data>/archive")
    parser.add_argument("--store", type=Path, help="default: <data>/shazam/results.jsonl")
    parser.add_argument("--snapshot", help="the Olaf snapshot whose results and pool.db to score")
    parser.add_argument("--min-match-count", type=int, default=DEFAULT_MIN_MATCH_COUNT)
    parser.add_argument("--only", choices=sorted(SOURCES), help="score one recognizer's legs")
    parser.add_argument(
        "--legs", nargs="+", choices=[leg.name for leg in LEGS], help="default: all"
    )
    parser.add_argument("--out", type=Path, help="the score file; default: <data>/score/score.json")
    parser.add_argument("--near-misses", type=Path, help="default: near_misses.csv beside --out")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    legs, use_shazam, use_olaf = select_legs(parser, args.only, args.legs)
    require_snapshot(parser, use_olaf, args.snapshot)
    snapshot = args.snapshot if use_olaf else None  # as run.py: ignored when no Olaf leg runs
    selected = [
        (leg, who)
        for leg in legs
        for who in leg.recognizers
        if (who == "shazam" and use_shazam) or (who == "olaf" and use_olaf)
    ]
    try:
        data = data_dir()
        plays_path = require_outside_checkout(args.plays)
        selection = require_outside_checkout(args.selection or data / "selection.json")
        archive_dir = require_outside_checkout(args.archive_dir or data / "archive")
        store = ResultStore(
            require_outside_checkout(args.store or data / "shazam" / "results.jsonl")
        )
        out = require_outside_checkout(args.out or data / "score" / "score.json")
        near_path = require_outside_checkout(args.near_misses or out.parent / "near_misses.csv")
        home = checked_snapshot_dir(snapshot) if snapshot else None
    except (DataPathError, SnapshotError) as refusal:
        raise SystemExit(str(refusal)) from None
    read = [
        plays_path,
        selection,
        store.path,
        *([home / "pool.db", home / RESULTS] if home else []),
    ]
    if _same_file(out, near_path):
        raise SystemExit(f"{out}: --out and --near-misses are the same file")
    for output in (out, near_path):
        if clash := next((i for i in read if _same_file(output, i)), None):
            raise SystemExit(f"{output}: is also an input ({clash}); not overwriting it")
    check_score_file(out)
    check_queue(near_path)
    if home and not (home / "pool.db").is_file():
        raise SystemExit(f"{home}: no snapshot here; build it first")
    try:
        plays = read_plays(plays_path)
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise SystemExit(f"{plays_path}: not a readable plays.jsonl ({error!r})") from None
    hours = read_hours(selection)
    stores = [store, ResultStore(home / RESULTS)] if home else [store]
    results = read_results(stores, study_identities(snapshot, args.min_match_count))
    references: dict[str, tuple[str, ...]] = {}
    if home:
        with closing(open_read_only(home / "pool.db")) as db:
            references = dict(reference_artists(db))
    scored: dict[str, LegScore] = {}
    entries: dict[str, dict[str, Any]] = {}
    for leg, who in selected:
        identity = (
            recognizer_identity(leg.length_s)
            if who == "shazam"
            else olaf_identity(args.snapshot, args.min_match_count)
        )
        addresses = hour_addresses(hours[leg.hours], archive_dir, leg.length_s, leg.profile)
        ls = score_leg(plays, results, identity, leg, hours[leg.hours], addresses, references)
        scored[f"{leg.name}/{who}"] = ls
        entries[f"{leg.name}/{who}"] = leg_json(leg, identity, ls)
    write_queue(near_path, NEAR_COLUMNS, near_miss_rows(plays, scored, references))
    out.parent.mkdir(parents=True, exist_ok=True)
    document = json.dumps({"version": 1, "legs": entries}, ensure_ascii=False, indent=1)
    temp = out.with_name(out.name + ".tmp")  # a complete file, then a rename: never half-written
    temp.write_text(document + "\n", encoding="utf-8")
    os.replace(temp, out)
    log.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
