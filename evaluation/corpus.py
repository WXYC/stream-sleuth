"""WXYC's flowsheet export to ``plays.jsonl``: eras, pads, carryover, in-pool join.

WXYC-specific. This module and :mod:`evaluation.archive` are the station side of
the ``plays.jsonl`` boundary: everything downstream reads only ``plays.jsonl``
and hour files. It reads the two CSVs that ``sql/flowsheet-export.sql`` writes
and never holds a database credential.

WXYC's history shapes the rules (plan §3, §4). Before ``ETL_STOP`` the flowsheet
was filled by an ETL whose ``add_time`` is not a logging moment (``etl`` era);
after it, rows are written when the DJ logs them (``canonical``). A
``canonical`` show touched by the legacy mirror (any ``legacy_entry_id``) has two
writers, so its ``play_order`` is unreliable; a single-writer show whose
``play_order`` disagrees with ``(add_time, id)`` order is reorder-flagged. Pads
were measured in the Phase 1 smoke test.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sqlite3
import statistics
import sys
from bisect import bisect_left
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from evaluation.archive import EASTERN, hour_key, hour_start
from evaluation.names import album_key, fold, fuzzy, named_qualifiers, qualifiers
from stream_sleuth.paths import data_dir, require_outside_checkout

log = logging.getLogger(__name__)

# Never earlier than the commit that stopped the ETL, 2026-08-08 21:30 PDT.
ETL_STOP_FLOOR = datetime(2026, 8, 9, 4, 30, tzinfo=timezone.utc)
# Eastern days whose rows came from a gap import, never selected (plan §10.11).
EXCLUDED_DAYS = {date(2026, 8, 9), date(2026, 8, 10), date(2026, 8, 11)}
PADS = {"canonical": 180.0, "etl": 220.0}
MIN_TRACKS = 8
MIN_MEDIAN_GAP_S = 90.0  # batch-logged hours log tracks seconds apart
# Time-of-day bands by the hour's America/New_York start, and the plan §5.2 corpus:
# 12 high-share and 4 low-share canonical DJ hours per band quota, and 4 contrast
# hours from 2022-2024, at most one per show and per recurring slot. No talk-hour group:
# WXYC plays music every hour, and its talkset rows are DJ mic breaks inside music hours.
BANDS = {"overnight": range(0, 6), "daytime": range(6, 18), "evening": range(18, 24)}
HIGH_QUOTAS = {"daytime": 5, "evening": 4, "overnight": 3}
LOW_QUOTAS = {"daytime": 2, "evening": 1, "overnight": 1}
LOW_SHARE_MAX = 0.10
CONTRAST_YEARS = (2022, 2023, 2024)
CONTRAST_HOURS = 4
TALK_TYPES = {"talkset", "message"}

_TIME = re.compile(r"(.{19})(?:\.(\d+))?([+-]\d{2})(?::?(\d{2}))?")


def parse_add_time(text: str) -> datetime:
    """Parse a Postgres ``timestamptz`` (``2026-08-12 20:31:05.1+00``) on Python 3.10."""
    match = _TIME.fullmatch(text.strip())
    if not match:
        raise ValueError(f"not a timestamp: {text!r}")
    base, frac, hours, minutes = match.groups()
    offset = timedelta(hours=int(hours), minutes=int(minutes or 0) * (-1 if hours[0] == "-" else 1))
    t = datetime.strptime(base, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone(offset))
    return (t + timedelta(microseconds=int((frac or "0")[:6].ljust(6, "0")))).astimezone(
        timezone.utc
    )


def etl_stop(export_dir: Path) -> datetime:
    """``flowsheet-etl``'s last run plus one half-hourly run, never before the floor."""
    with open(export_dir / "cronjob_runs.csv", newline="", encoding="utf-8") as f:
        runs = [
            parse_add_time(r["last_run"])
            for r in csv.DictReader(f)
            if r["job_name"] == "flowsheet-etl"
        ]
    if not runs:
        raise ValueError(
            f"{export_dir / 'cronjob_runs.csv'} has no flowsheet-etl row; re-run the export"
        )
    return max(max(runs) + timedelta(minutes=30), ETL_STOP_FLOOR)


Key = tuple[str, str]
Slot = tuple[int, int, int]  # (Eastern year, weekday with Monday 0, hour) of a show's first track
FileRef = tuple[str, str]  # (files.key, files.format)
Entry = tuple[FileRef, frozenset[str]]  # a file and the qualifiers named in its album and title


@dataclass
class PoolIndex:
    """Join keys over ``pool.db``'s indexed files, by tier (plan §5.1).

    Each key maps to its indexed files in ``files.key`` order, each with the qualifiers
    its album and title name (:func:`named_qualifiers`). A key with an empty part (a
    missing tag, or a name that normalizes to nothing) never joins.
    """

    album: dict[Key, list[Entry]] = field(default_factory=dict)
    album_fuzzy: dict[Key, list[Entry]] = field(default_factory=dict)
    title: dict[Key, list[Entry]] = field(default_factory=dict)
    title_fuzzy: dict[Key, list[Entry]] = field(default_factory=dict)

    @classmethod
    def load(cls, pool_db: Path) -> PoolIndex:
        """Read ``pool_db`` read-only; a missing file raises instead of being created."""
        index = cls()
        db = sqlite3.connect(f"{Path(pool_db).resolve().as_uri()}?mode=ro", uri=True)
        try:
            rows = db.execute(
                "SELECT key, format, artist, album_artist, album, title FROM files"
                " WHERE status = 'indexed' ORDER BY key"
            ).fetchall()
        finally:
            db.close()
        for key, fmt, artist, album_artist, album, title in rows:
            entry = ((key, fmt), named_qualifiers(album, title))
            for a in {artist, album_artist} - {None, ""}:
                for _, keys, k in index._keys(a, album, title):
                    keys.setdefault(k, []).append(entry)
        return index

    def _keys(
        self, artist: str, album: str | None, title: str | None
    ) -> list[tuple[str, dict[Key, list[Entry]], Key]]:
        """``(tier, map, key)`` in tier order, leaving out every key with an empty part."""
        candidates = [
            ("exact", self.album, (fold(artist), album_key(album))),
            ("fuzzy", self.album_fuzzy, (fuzzy(artist), fuzzy(album))),
            ("title", self.title, (fold(artist), album_key(title))),
            ("title", self.title_fuzzy, (fuzzy(artist), fuzzy(title))),
        ]
        return [(tier, keys, k) for tier, keys, k in candidates if all(k)]

    def match(self, artist: str, album: str, title: str) -> tuple[str, str] | None:
        """``(tier, format)`` of the first file by key at the best matching tier, else None.

        Tiers: ``exact`` or ``fuzzy`` on (artist, album), else ``title`` on (artist, title).
        On every tier a file joins only when its album or title names each qualifier the
        play's album or title names in a version clause (:func:`named_qualifiers`: in its
        album anywhere, in its title bracketed or after a dash), so a live play never joins
        the studio file.
        """
        need = qualifiers(album, title)
        hits = [
            (tier, ref)
            for tier, keys, k in self._keys(artist, album, title)
            for ref, has in keys.get(k, [])
            if need <= has
        ]
        if not hits:
            return None
        best = hits[0][0]
        return best, min(ref for tier, ref in hits if tier == best)[1]

    def tier(self, artist: str, album: str, title: str) -> str | None:
        """The best matching tier, or None."""
        found = self.match(artist, album, title)
        return found[0] if found else None


@dataclass(frozen=True)
class Row:
    """One flowsheet export row; ``show_id`` is ``""`` for a row with no show."""

    id: int
    show_id: str
    play_order: int
    legacy: bool
    entry_type: str
    add_time: datetime
    artist: str
    title: str
    album: str
    rotation: bool


@dataclass
class Flowsheet:
    """The export's rows in ``(add_time, id)`` order, with ``ETL_STOP`` and show labels."""

    rows: list[Row]
    stop: datetime
    by_hour: dict[str, list[Row]] = field(default_factory=lambda: defaultdict(list))
    order_status: dict[str, tuple[str, bool | None]] = field(default_factory=dict)
    show_slot: dict[str, Slot] = field(default_factory=dict)

    @classmethod
    def load(cls, export_dir: Path) -> Flowsheet:
        """Read ``flowsheet.csv`` and ``cronjob_runs.csv`` from ``export_dir``.

        ``by_hour`` holds each hour key's rows; rows in the fall-back hour, which has
        no key, appear only in ``rows``. A show with a legacy row of any entry type
        has two writers (``unreliable``); rows with no show get no label. A show's
        ``show_slot`` is the Eastern year, weekday and hour of its first track row, a
        proxy for a recurring program (a weekly show is one ``show_id`` per broadcast);
        a show with no track row has none.
        """
        with open(export_dir / "flowsheet.csv", newline="", encoding="utf-8") as f:
            rows = [
                Row(int(r["id"]), r["show_id"], int(r["play_order"]), bool(r["legacy_entry_id"]),
                    r["entry_type"], parse_add_time(r["add_time"]), r["artist_name"],
                    r["track_title"], r["album_title"], bool(r["rotation_id"]))
                for r in csv.DictReader(f)
            ]  # fmt: skip
        sheet = cls(sorted(rows, key=lambda r: (r.add_time, r.id)), etl_stop(export_dir))
        shows: dict[str, list[Row]] = defaultdict(list)
        for r in sheet.rows:
            if (key := hour_key(r.add_time)) is not None:
                sheet.by_hour[key].append(r)
            if r.show_id:
                shows[r.show_id].append(r)
        for show, show_rows in shows.items():
            if any(r.legacy for r in show_rows):
                sheet.order_status[show] = ("unreliable", None)
            else:
                tracks = [r for r in show_rows if r.entry_type == "track"]
                by_order = sorted(tracks, key=lambda r: r.play_order)
                sheet.order_status[show] = ("single_writer", by_order != tracks)
            if first := next((r for r in show_rows if r.entry_type == "track"), None):
                local = first.add_time.astimezone(EASTERN)
                sheet.show_slot[show] = (local.year, local.weekday(), local.hour)
        log.info("loaded %d rows, ETL_STOP %s", len(sheet.rows), sheet.stop.isoformat())
        return sheet

    def era(self, t: datetime) -> str:
        """``canonical`` at or after ``ETL_STOP``, else ``etl``."""
        return "canonical" if t >= self.stop else "etl"


@dataclass(frozen=True)
class HourStats:
    """One hour's selection inputs; ``shows`` and ``slots`` cover every track row in it."""

    key: str
    era: str
    track_rows: int
    in_pool: int
    median_gap_s: float
    reorder_flagged: bool
    shows: frozenset[str] = frozenset()
    slots: frozenset[Slot] = frozenset()


def hour_stats(sheet: Flowsheet, pool: PoolIndex) -> dict[str, HourStats]:
    """Per-hour counts and median inter-track gap; the era is the hour start's."""
    stats = {}
    for key, rows in sheet.by_hour.items():
        tracks = [r for r in rows if r.entry_type == "track"]
        times = [r.add_time.timestamp() for r in tracks]
        gaps = [b - a for a, b in zip(times, times[1:], strict=False)]
        stats[key] = HourStats(
            key, sheet.era(hour_start(key)), len(tracks),
            sum(pool.tier(r.artist, r.album, r.title) is not None for r in tracks),
            statistics.median(gaps) if gaps else 0.0,
            any(sheet.order_status.get(r.show_id, ("unreliable", None))[1]
                for r in tracks if sheet.era(r.add_time) == "canonical"),
            frozenset(r.show_id for r in tracks if r.show_id),
            frozenset(sheet.show_slot[r.show_id] for r in tracks if r.show_id),
        )  # fmt: skip
    return stats


def _eligible(h: HourStats) -> bool:
    return hour_start(h.key).astimezone(EASTERN).date() not in EXCLUDED_DAYS


def select_hours(stats: dict[str, HourStats], *, era: str, target: int) -> list[str]:
    """DJ hours in descending in-pool order until their in-pool plays reach ``target``.

    Hours holding a reorder-flagged show rank after every unflagged hour (plan §4).
    To restrict the date window, filter ``stats`` before calling.
    """
    candidates = sorted(
        (h for h in stats.values() if h.era == era and _eligible(h) and h.in_pool
         and h.track_rows >= MIN_TRACKS and h.median_gap_s >= MIN_MEDIAN_GAP_S),
        key=lambda h: (h.reorder_flagged, -h.in_pool, h.key),
    )  # fmt: skip
    chosen, total = [], 0
    for h in candidates:
        if total >= target:
            break
        chosen.append(h.key)
        total += h.in_pool
    log.info("%s: %d hours, %d expected in-pool plays (target %d)", era, len(chosen), total, target)
    return chosen


def band(key: str) -> str:
    """The time-of-day band of an hour key's Eastern start."""
    hour = hour_start(key).astimezone(EASTERN).hour
    return next(name for name, hours in BANDS.items() if hour in hours)


@dataclass
class Selection:
    """Chosen hour keys -> ``(group, band)`` in selection order, and every shortfall.

    Groups are ``canonical-high`` (12), ``canonical-low`` (4) and ``contrast`` (4):
    20 hours when nothing is short. Contrast hours come from distinct shows
    (``show_id``, one per broadcast) and distinct recurring slots, where a slot is the
    Eastern year, weekday and hour of a show's first track row. An hour belongs to the
    show and slot of every track row in it, so an hour spanning two blocks both. The
    slot is a proxy for distinct programs, not a guarantee: a program whose first
    track lands in a neighbouring hour gets a different slot, a program holding the
    same slot in different years can supply one hour per year, and no DJ is identified.

    A ``<group>/<band>`` shortfall was relaxed: filled from the group's other bands.
    A ``contrast/<year>`` shortfall was relaxed too: that year had no eligible hour
    from an unused show and slot, so its place went to the next best hour from the
    other contrast years, if one was left. Only ``<group>/unfilled`` counts hours that
    could not be found at all; the one-per-show and per-slot caps are never relaxed
    to fill one. A ``subset/<group>`` shortfall (added by :func:`choose_subset`) is
    neither: it counts four-hour-subset slots of that group left empty, never filled
    from another group, and says nothing about the corpus itself.
    """

    hours: dict[str, tuple[str, str]] = field(default_factory=dict)
    shortfalls: dict[str, int] = field(default_factory=dict)

    def take(self, hours: Iterable[HourStats], group: str) -> int:
        taken = 0
        for h in hours:
            if h.key not in self.hours:
                self.hours[h.key] = (group, band(h.key))
                taken += 1
        return taken

    def short(self, name: str, missing: int) -> None:
        if missing > 0:
            self.shortfalls[name] = missing


def _dj(h: HourStats) -> bool:
    return _eligible(h) and h.track_rows >= MIN_TRACKS and h.median_gap_s >= MIN_MEDIAN_GAP_S


def _share(h: HourStats) -> float:
    return h.in_pool / h.track_rows


def _by_share(h: HourStats) -> tuple[bool, float, int, str]:
    return (h.reorder_flagged, -_share(h), -h.in_pool, h.key)


def _by_tracks(h: HourStats) -> tuple[bool, int, str]:
    return (h.reorder_flagged, -h.track_rows, h.key)


def _fill(sel: Selection, candidates: list[HourStats], quotas: dict[str, int], group: str) -> None:
    """Fill each band's quota in rank order, then relax shortfalls from the other bands."""
    for name, quota in quotas.items():
        got = sel.take(
            [h for h in candidates if band(h.key) == name and h.key not in sel.hours][:quota], group
        )
        sel.short(f"{group}/{name}", quota - got)
    missing = sum(quotas.values()) - sum(g == group for g, _ in sel.hours.values())
    sel.short(
        f"{group}/unfilled",
        missing - sel.take([h for h in candidates if h.key not in sel.hours][:missing], group),
    )


def _contrast(sel: Selection, old: list[HourStats]) -> None:
    """Up to ``CONTRAST_HOURS`` hours, the best share per year first, one per show and slot."""
    used: set[str] = set()
    used_slots: set[Slot] = set()

    def pick(ranked: Iterable[HourStats]) -> int:
        h = next(
            (
                h
                for h in ranked
                if h.key not in sel.hours and not h.shows & used and not h.slots & used_slots
            ),
            None,
        )
        if h is None:
            return 0
        used.update(h.shows)
        used_slots.update(h.slots)
        return sel.take([h], "contrast")

    chosen = 0
    for year in CONTRAST_YEARS:
        got = pick(h for h in old if h.key.startswith(str(year)))
        sel.short(f"contrast/{year}", 1 - got)
        chosen += got
    while chosen < CONTRAST_HOURS and pick(old):
        chosen += 1
    sel.short("contrast/unfilled", CONTRAST_HOURS - chosen)


def select_corpus(stats: dict[str, HourStats]) -> Selection:
    """The plan §5.2 corpus: stratified canonical DJ hours and contrast hours.

    Canonical hours never come from the ``etl`` era, however thin a band is.
    """
    sel = Selection()
    canonical = [h for h in stats.values() if h.era == "canonical" and _dj(h)]
    high = sorted((h for h in canonical if _share(h) > LOW_SHARE_MAX), key=_by_share)
    low = sorted((h for h in canonical if _share(h) <= LOW_SHARE_MAX), key=_by_tracks)
    _fill(sel, high, HIGH_QUOTAS, "canonical-high")
    _fill(sel, low, LOW_QUOTAS, "canonical-low")
    old = sorted((h for h in stats.values() if h.era == "etl" and _dj(h)
                  and hour_start(h.key).astimezone(EASTERN).year in CONTRAST_YEARS), key=_by_share)  # fmt: skip
    _contrast(sel, old)
    for name, missing in sel.shortfalls.items():
        log.warning("selection shortfall %s: %d", name, missing)
    log.info("selected %d hours", len(sel.hours))
    return sel


def choose_subset(sel: Selection, stats: dict[str, HourStats]) -> list[str]:
    """The plan §5.2 four-hour subset, drawn only from ``sel``: high, high, low, contrast.

    The two high hours are the top-ranked ``canonical-high`` hour and the top-ranked one
    from a different band, in the corpus's own ranking (reorder-flagged hours last); the
    low hour is the top-ranked ``canonical-low`` hour and the contrast hour the top-ranked
    ``contrast`` hour. A slot with no candidate is a ``subset/<group>`` shortfall in
    ``sel`` and is never filled from another group.
    """

    def chosen(group: str, rank: Any, *, skip_band: str = "") -> list[HourStats]:
        pool = [stats[k] for k, (g, b) in sel.hours.items() if g == group and b != skip_band]
        return sorted(pool, key=rank)[:1]

    high = chosen("canonical-high", _by_share)
    high += chosen("canonical-high", _by_share, skip_band=sel.hours[high[0].key][1]) if high else []
    low = chosen("canonical-low", _by_tracks)
    contrast = chosen("contrast", _by_share)
    sel.short("subset/canonical-high", 2 - len(high))
    sel.short("subset/canonical-low", 1 - len(low))
    sel.short("subset/contrast", 1 - len(contrast))
    return [h.key for h in (*high, *low, *contrast)]


def write_selection(
    out_dir: Path, sel: Selection, subset: list[str], export: str, pool_db: Path
) -> None:
    """Write ``hours.txt``, ``subset.txt`` and ``selection.json`` into ``out_dir``; never overwrite.

    ``out_dir`` is checked with :func:`stream_sleuth.paths.require_outside_checkout`
    before anything is created. A file that already exists raises ``FileExistsError``
    and removes the files this call created, so a frozen selection is never half replaced.
    ``selection.json`` is a record for people and for the report's shortfall line, not a
    boundary artifact: ``hours`` maps each key to its ``group``, ``band`` and ``subset``.
    """
    out_dir = require_outside_checkout(out_dir)
    record = {
        "export": export,
        "pool_db": str(pool_db),
        "hours": {
            k: {"group": g, "band": b, "subset": k in subset} for k, (g, b) in sel.hours.items()
        },
        "shortfalls": dict(sorted(sel.shortfalls.items())),
    }
    files = {
        "hours.txt": "".join(f"{k}\n" for k in sel.hours),
        "subset.txt": "".join(f"{k}\n" for k in subset),
        "selection.json": json.dumps(record, ensure_ascii=False, indent=2) + "\n",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    try:
        for name, text in files.items():
            with open(out_dir / name, "x", encoding="utf-8") as f:
                created.append(out_dir / name)
                f.write(text)
    except BaseException:
        for path in created:
            path.unlink()  # only what this call created
        raise


def write_plays(out: Path, hours: Iterable[str], sheet: Flowsheet, pool: PoolIndex) -> None:
    """Write one ``plays.jsonl`` record per track row in each hour; never overwrites ``out``.

    Every key is validated before ``out`` is created, so a bad key leaves no partial
    file, and a failed write removes the file it created. An hour with no flowsheet
    rows (outside the export, or a gap) writes no records and logs a warning.

    Each hour also gets the last track row of the 60 minutes before it as an
    attribution-only play (``carryover: true``, negative ``t_offset_s``): a song
    started before the top of the hour is still playing in it. Plays with no show
    are ``unreliable``: there is no show order to check them against.

    Raises :class:`~stream_sleuth.paths.DataPathError` before building anything when
    ``out`` is relative or inside the checkout: ``plays.jsonl`` holds real flowsheet rows.
    """
    require_outside_checkout(out)
    lines = []
    for key in dict.fromkeys(hours):
        start = hour_start(key)
        rows = sheet.by_hour.get(key, [])
        if not rows:
            log.warning("%s: no flowsheet rows; is the hour inside the export?", key)
        tracks = [r for r in rows if r.entry_type == "track"]
        talk_rows = sum(r.entry_type in TALK_TYPES for r in rows)
        # By instant, not hour key, so the hour after fall-back still gets its carryover.
        lo = bisect_left(sheet.rows, start - timedelta(hours=1), key=lambda r: r.add_time)
        hi = bisect_left(sheet.rows, start, lo=lo, key=lambda r: r.add_time)
        before = [r for r in sheet.rows[lo:hi] if r.entry_type == "track"]
        plays = [(r, True) for r in before[-1:]] + [(r, False) for r in tracks]
        for i, (r, carryover) in enumerate(plays):
            t = (r.add_time - start).total_seconds()
            t_next = (
                (plays[i + 1][0].add_time - start).total_seconds() if i + 1 < len(plays) else 3600.0
            )
            era = sheet.era(r.add_time)
            tier, fmt = pool.match(r.artist, r.album, r.title) or (None, None)
            status, flag = (
                sheet.order_status.get(r.show_id, ("unreliable", None))
                if era == "canonical"
                else ("etl", None)
            )
            record = {
                "hour_key": key, "play_id": r.id, "t_offset_s": t,
                "window_start_s": max(0.0, t - PADS[era]), "window_end_s": min(3600.0, t_next + PADS[era]),
                "artist": r.artist, "title": r.title, "album": r.album, "era": era, "pad_s": PADS[era],
                "in_pool": tier is not None if fold(r.artist) else None, "pool_match_tier": tier,
                "pool_format": fmt, "rotation": r.rotation, "reorder_flag": flag, "play_order_status": status,
                "carryover": carryover, "track_rows": len(tracks), "talk_rows": talk_rows,
            }  # fmt: skip
            lines.append(json.dumps(record, ensure_ascii=False) + "\n")
        log.info("%s: %d plays (%d carryover)", key, len(plays), len(plays) - len(tracks))
    # Built in full first, so a bad key leaves no partial file. An existing ``out``
    # raises before the guard; the guard covers close, whose final flush can fail too.
    f = open(out, "x", encoding="utf-8")
    try:
        with f:
            f.writelines(lines)
    except BaseException:
        out.unlink()  # the file this call created, never an earlier one
        raise


def _select(argv: list[str]) -> int:
    """``select``: freeze the corpus selection and its four-hour subset; see the README."""
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.corpus select",
        description="Write hours.txt, subset.txt and selection.json; never overwrites them.",
    )
    parser.add_argument("--export", type=Path, required=True, help="dated export directory")
    parser.add_argument("--pool-db", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, help="default: the data directory")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out_dir = args.out_dir if args.out_dir else data_dir()
    stats = hour_stats(Flowsheet.load(args.export), PoolIndex.load(args.pool_db))
    sel = select_corpus(stats)
    subset = choose_subset(sel, stats)
    write_selection(out_dir, sel, subset, args.export.name, args.pool_db)
    return 0


def main(argv: list[str] | None = None) -> int:
    """The CLI: ``select ...`` freezes a selection; otherwise ``--hours`` writes plays."""
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["select"]:
        return _select(argv[1:])
    parser = argparse.ArgumentParser(
        description="Write plays.jsonl from a flowsheet export. Run `select` first to freeze the hours."
    )
    parser.add_argument("--export", type=Path, required=True, help="dated export directory")
    parser.add_argument("--pool-db", type=Path, required=True)
    parser.add_argument("--hours", type=Path, required=True, help="file of hour keys, one per line")
    parser.add_argument("--out", type=Path, required=True, help="plays.jsonl to create")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    hours = args.hours.read_text().split()
    write_plays(args.out, hours, Flowsheet.load(args.export), PoolIndex.load(args.pool_db))
    return 0


if __name__ == "__main__":
    sys.exit(main())
