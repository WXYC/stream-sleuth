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
import unicodedata
from bisect import bisect_left
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from evaluation.archive import EASTERN, hour_key, hour_start

log = logging.getLogger(__name__)

# Never earlier than the commit that stopped the ETL, 2026-08-08 21:30 PDT.
ETL_STOP_FLOOR = datetime(2026, 8, 9, 4, 30, tzinfo=timezone.utc)
# Eastern days whose rows came from a gap import, never selected (plan §10.11).
EXCLUDED_DAYS = {date(2026, 8, 9), date(2026, 8, 10), date(2026, 8, 11)}
PADS = {"canonical": 180.0, "etl": 220.0}
MIN_TRACKS = 8
MIN_MEDIAN_GAP_S = 90.0  # batch-logged hours log tracks seconds apart
MAX_TALK_HOUR_TRACKS = 3
# Time-of-day bands by the hour's America/New_York start, and the plan §5.2 corpus:
# 12 high-share and 4 low-share canonical DJ hours per band quota, 4 contrast hours
# from 2022-2024, 2 talk-heavy hours.
BANDS = {"overnight": range(0, 6), "daytime": range(6, 18), "evening": range(18, 24)}
HIGH_QUOTAS = {"daytime": 5, "evening": 4, "overnight": 3}
LOW_QUOTAS = {"daytime": 2, "evening": 1, "overnight": 1}
LOW_SHARE_MAX = 0.10
CONTRAST_YEARS = (2022, 2023, 2024)
CONTRAST_HOURS = 4
TALK_HOURS = 2
TALK_TYPES = {"talkset", "message"}

_TIME = re.compile(r"(.{19})(?:\.(\d+))?([+-]\d{2})(?::?(\d{2}))?")
_CRUFT = re.compile(
    r"\s*[\(\[](?:feat|ft|featuring|deluxe|remaster(?:ed)?|expanded|anniversary|bonus)\b[^\)\]]*[\)\]]",
    re.IGNORECASE,
)
_BRACKETED = re.compile(r"\([^()]*\)|\[[^\[\]]*\]|\{[^{}]*\}")


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


def fold(s: str | None) -> str:
    """NFKD, strip combining marks, casefold; punctuation kept (the exact-tier key)."""
    decomposed = unicodedata.normalize("NFKD", s or "")
    return "".join(c for c in decomposed if not unicodedata.combining(c)).casefold().strip()


def album_key(s: str | None) -> str:
    """Lowercase, drop featuring and edition cruft, collapse whitespace."""
    return " ".join(_CRUFT.sub("", (s or "").lower()).split())


def fuzzy(s: str | None) -> str:
    """The fuzzy-tier key: ``fold(album_key(s))`` less bracketed clauses, non-word runs as a space.

    ``(...)``, ``[...]`` and ``{...}`` clauses are dropped after NFKD, which folds
    full-width brackets to ASCII, so "The Worm" joins a tag "The Worm（ザ・ワーム）".
    Letters and digits of every script outside brackets survive (a Japanese or
    Cyrillic name keeps a real key); a name that is all brackets keys to "" and never
    joins on this tier. Underscores and punctuation separate.
    """
    return " ".join(re.sub(r"[\W_]+", " ", _BRACKETED.sub(" ", fold(album_key(s)))).split())


Key = tuple[str, str]
FileRef = tuple[str, str]  # (files.key, files.format)


@dataclass
class PoolIndex:
    """Join keys over ``pool.db``'s indexed files, by tier (plan §5.1).

    Each key maps to the first indexed file by ``files.key`` that has it. A key with
    an empty part (a missing tag, or a name that normalizes to nothing) never joins.
    """

    album: dict[Key, FileRef] = field(default_factory=dict)
    album_fuzzy: dict[Key, FileRef] = field(default_factory=dict)
    title: dict[Key, FileRef] = field(default_factory=dict)
    title_fuzzy: dict[Key, FileRef] = field(default_factory=dict)

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
            for a in {artist, album_artist} - {None, ""}:
                for _, keys, k in index._keys(a, album, title):
                    keys.setdefault(k, (key, fmt))
        return index

    def _keys(
        self, artist: str, album: str | None, title: str | None
    ) -> list[tuple[str, dict[Key, FileRef], Key]]:
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
        """
        hits = [(tier, keys[k]) for tier, keys, k in self._keys(artist, album, title) if k in keys]
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

    @classmethod
    def load(cls, export_dir: Path) -> Flowsheet:
        """Read ``flowsheet.csv`` and ``cronjob_runs.csv`` from ``export_dir``.

        ``by_hour`` holds each hour key's rows; rows in the fall-back hour, which has
        no key, appear only in ``rows``. A show with a legacy row of any entry type
        has two writers (``unreliable``); rows with no show get no label.
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
        log.info("loaded %d rows, ETL_STOP %s", len(sheet.rows), sheet.stop.isoformat())
        return sheet

    def era(self, t: datetime) -> str:
        """``canonical`` at or after ``ETL_STOP``, else ``etl``."""
        return "canonical" if t >= self.stop else "etl"


@dataclass(frozen=True)
class HourStats:
    key: str
    era: str
    track_rows: int
    talk_rows: int
    in_pool: int
    median_gap_s: float
    reorder_flagged: bool


def hour_stats(sheet: Flowsheet, pool: PoolIndex) -> dict[str, HourStats]:
    """Per-hour counts and median inter-track gap; the era is the hour start's."""
    stats = {}
    for key, rows in sheet.by_hour.items():
        tracks = [r for r in rows if r.entry_type == "track"]
        times = [r.add_time.timestamp() for r in tracks]
        gaps = [b - a for a, b in zip(times, times[1:], strict=False)]
        stats[key] = HourStats(
            key, sheet.era(hour_start(key)), len(tracks), sum(r.entry_type in TALK_TYPES for r in rows),
            sum(pool.tier(r.artist, r.album, r.title) is not None for r in tracks),
            statistics.median(gaps) if gaps else 0.0,
            any(sheet.order_status.get(r.show_id, ("unreliable", None))[1]
                for r in tracks if sheet.era(r.add_time) == "canonical"),
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


def select_talk_hours(stats: dict[str, HourStats], *, count: int) -> list[str]:
    """Hours with the most talkset/message rows and at most three track rows."""
    talk = [
        h
        for h in stats.values()
        if _eligible(h) and h.talk_rows and h.track_rows <= MAX_TALK_HOUR_TRACKS
    ]
    return [h.key for h in sorted(talk, key=lambda h: (-h.talk_rows, h.key))[:count]]


def band(key: str) -> str:
    """The time-of-day band of an hour key's Eastern start."""
    hour = hour_start(key).astimezone(EASTERN).hour
    return next(name for name, hours in BANDS.items() if hour in hours)


@dataclass
class Selection:
    """Chosen hour keys -> ``(group, band)`` in selection order, and every shortfall.

    A ``<group>/<band>`` shortfall was relaxed: filled from the group's other bands.
    A ``contrast/<year>`` shortfall was relaxed too: that year had no eligible hour,
    so its slot went to the next best hour from the other contrast years, if one was
    left. Only ``<group>/unfilled`` counts hours that could not be found at all.
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


def select_corpus(stats: dict[str, HourStats]) -> Selection:
    """The plan §5.2 corpus: stratified canonical DJ hours, contrast hours, talk hours.

    Canonical hours never come from the ``etl`` era, however thin a band is.
    """
    sel = Selection()
    canonical = [h for h in stats.values() if h.era == "canonical" and _dj(h)]
    high = sorted((h for h in canonical if _share(h) > LOW_SHARE_MAX), key=_by_share)
    low = sorted((h for h in canonical if _share(h) <= LOW_SHARE_MAX),
                 key=lambda h: (h.reorder_flagged, -h.track_rows, h.key))  # fmt: skip
    _fill(sel, high, HIGH_QUOTAS, "canonical-high")
    _fill(sel, low, LOW_QUOTAS, "canonical-low")
    old = sorted((h for h in stats.values() if h.era == "etl" and _dj(h)
                  and hour_start(h.key).astimezone(EASTERN).year in CONTRAST_YEARS), key=_by_share)  # fmt: skip
    for year in CONTRAST_YEARS:
        sel.short(
            f"contrast/{year}",
            1 - sel.take([h for h in old if h.key.startswith(str(year))][:1], "contrast"),
        )
    chosen = sum(g == "contrast" for g, _ in sel.hours.values())
    chosen += sel.take(
        [h for h in old if h.key not in sel.hours][: CONTRAST_HOURS - chosen], "contrast"
    )
    sel.short("contrast/unfilled", CONTRAST_HOURS - chosen)
    talk = select_talk_hours(stats, count=TALK_HOURS)
    sel.short("talk/unfilled", TALK_HOURS - sel.take([stats[k] for k in talk], "talk"))
    for name, missing in sel.shortfalls.items():
        log.warning("selection shortfall %s: %d", name, missing)
    log.info("selected %d hours", len(sel.hours))
    return sel


def write_plays(out: Path, hours: Iterable[str], sheet: Flowsheet, pool: PoolIndex) -> None:
    """Write one ``plays.jsonl`` record per track row in each hour; never overwrites ``out``.

    Every key is validated before ``out`` is created, so a bad key leaves no partial
    file, and a failed write removes the file it created. An hour with no flowsheet
    rows (outside the export, or a gap) writes no records and logs a warning.

    Each hour also gets the last track row of the 60 minutes before it as an
    attribution-only play (``carryover: true``, negative ``t_offset_s``): a song
    started before the top of the hour is still playing in it. Plays with no show
    are ``unreliable``: there is no show order to check them against.
    """
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


def main(argv: list[str] | None = None) -> int:
    """The CLI: ``--export``, ``--pool-db``, ``--hours`` and ``--out``; see the README."""
    parser = argparse.ArgumentParser(description="Write plays.jsonl from a flowsheet export.")
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
