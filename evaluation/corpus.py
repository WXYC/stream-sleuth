"""WXYC's flowsheet export to ``plays.jsonl``: eras, pads, hour selection, in-pool join.

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
TALK_TYPES = {"talkset", "message"}

_TIME = re.compile(r"(.{19})(?:\.(\d+))?([+-]\d{2})(?::?(\d{2}))?")
_CRUFT = re.compile(
    r"\s*[\(\[](?:feat\.?|ft\.?|featuring|deluxe|remaster(?:ed)?|expanded|anniversary|bonus)[^\)\]]*[\)\]]",
    re.IGNORECASE,
)


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
    with open(export_dir / "cronjob_runs.csv", newline="") as f:
        last = max(
            parse_add_time(r["last_run"])
            for r in csv.DictReader(f)
            if r["job_name"] == "flowsheet-etl"
        )
    return max(last + timedelta(minutes=30), ETL_STOP_FLOOR)


def fold(s: str | None) -> str:
    """NFKD, strip combining marks, casefold; punctuation kept (the exact-tier key)."""
    decomposed = unicodedata.normalize("NFKD", s or "")
    return "".join(c for c in decomposed if not unicodedata.combining(c)).casefold().strip()


def album_key(s: str | None) -> str:
    """Lowercase, drop featuring and edition cruft, collapse whitespace."""
    return " ".join(_CRUFT.sub("", (s or "").lower()).split())


def fuzzy(s: str | None) -> str:
    """The fuzzy-tier key: ``fold(album_key(s))`` with non-alphanumeric runs as one space."""
    return " ".join(re.sub(r"[^0-9a-z]+", " ", fold(album_key(s))).split())


@dataclass
class PoolIndex:
    """Join keys over ``pool.db``'s indexed files, by tier (plan §5.1)."""

    keys: dict[str, set[tuple[str, str]]] = field(default_factory=lambda: defaultdict(set))

    @classmethod
    def load(cls, pool_db: Path) -> PoolIndex:
        index = cls()
        db = sqlite3.connect(pool_db)
        rows = db.execute(
            "SELECT artist, album_artist, album, title FROM files WHERE status = 'indexed'"
        )
        for artist, album_artist, album, title in rows:
            for a in {artist, album_artist} - {None, ""}:
                index.keys["album"].add((fold(a), album_key(album)))
                index.keys["album_fuzzy"].add((fuzzy(a), fuzzy(album)))
                index.keys["title"].add((fold(a), album_key(title)))
                index.keys["title_fuzzy"].add((fuzzy(a), fuzzy(title)))
        db.close()
        return index

    def tier(self, artist: str, album: str, title: str) -> str | None:
        """``exact`` or ``fuzzy`` on (artist, album), else ``title`` on (artist, title), else None."""
        if not artist:
            return None
        if (fold(artist), album_key(album)) in self.keys["album"]:
            return "exact"
        if (fuzzy(artist), fuzzy(album)) in self.keys["album_fuzzy"]:
            return "fuzzy"
        if (fold(artist), album_key(title)) in self.keys["title"] or (
            fuzzy(artist),
            fuzzy(title),
        ) in self.keys["title_fuzzy"]:
            return "title"
        return None


@dataclass(frozen=True)
class Row:
    id: int
    show_id: str
    play_order: int | None
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
        with open(export_dir / "flowsheet.csv", newline="") as f:
            rows = [
                Row(int(r["id"]), r["show_id"], int(r["play_order"]) if r["play_order"] else None,
                    bool(r["legacy_entry_id"]), r["entry_type"], parse_add_time(r["add_time"]),
                    r["artist_name"], r["track_title"], r["album_title"], bool(r["rotation_id"]))
                for r in csv.DictReader(f)
            ]  # fmt: skip
        sheet = cls(sorted(rows, key=lambda r: (r.add_time, r.id)), etl_stop(export_dir))
        shows: dict[str, list[Row]] = defaultdict(list)
        for r in sheet.rows:
            if (key := hour_key(r.add_time)) is not None:
                sheet.by_hour[key].append(r)
            if r.entry_type == "track":
                shows[r.show_id].append(r)
        for show, tracks in shows.items():
            if any(r.legacy for r in tracks):
                sheet.order_status[show] = ("unreliable", None)
            else:
                by_order = sorted(tracks, key=lambda r: (r.play_order is None, r.play_order or 0))
                sheet.order_status[show] = ("single_writer", by_order != tracks)
        log.info("loaded %d rows, ETL_STOP %s", len(sheet.rows), sheet.stop.isoformat())
        return sheet

    def era(self, t: datetime) -> str:
        return "canonical" if t >= self.stop else "etl"


@dataclass(frozen=True)
class HourStats:
    key: str
    era: str
    track_rows: int
    talk_rows: int
    in_pool: int
    median_gap_s: float


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
        )  # fmt: skip
    return stats


def _eligible(h: HourStats) -> bool:
    return hour_start(h.key).astimezone(EASTERN).date() not in EXCLUDED_DAYS


def select_hours(stats: dict[str, HourStats], *, era: str, target: int) -> list[str]:
    """DJ hours in descending in-pool order until their in-pool plays reach ``target``."""
    candidates = sorted(
        (h for h in stats.values() if h.era == era and _eligible(h) and h.in_pool
         and h.track_rows >= MIN_TRACKS and h.median_gap_s >= MIN_MEDIAN_GAP_S),
        key=lambda h: (-h.in_pool, h.key),
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


def write_plays(out: Path, hours: Iterable[str], sheet: Flowsheet, pool: PoolIndex) -> None:
    """Write one ``plays.jsonl`` record per track row in each hour; never overwrites ``out``.

    Each hour also gets the previous hour's last track row as an attribution-only
    play (``carryover: true``, negative ``t_offset_s``): a song started before the
    top of the hour is still playing in it.
    """
    with open(out, "x") as f:
        for key in hours:
            start = hour_start(key)
            rows = sheet.by_hour.get(key, [])
            tracks = [r for r in rows if r.entry_type == "track"]
            before = [r for r in sheet.by_hour.get(hour_key(start - timedelta(hours=1)) or "", [])
                      if r.entry_type == "track"]  # fmt: skip
            plays = [(r, True) for r in before[-1:]] + [(r, False) for r in tracks]
            for i, (r, carryover) in enumerate(plays):
                t = (r.add_time - start).total_seconds()
                t_next = (
                    (plays[i + 1][0].add_time - start).total_seconds()
                    if i + 1 < len(plays)
                    else 3600.0
                )
                era = sheet.era(r.add_time)
                tier = pool.tier(r.artist, r.album, r.title)
                status, flag = (
                    sheet.order_status[r.show_id] if era == "canonical" else ("etl", None)
                )
                record = {
                    "hour_key": key, "play_id": r.id, "t_offset_s": t,
                    "window_start_s": max(0.0, t - PADS[era]), "window_end_s": min(3600.0, t_next + PADS[era]),
                    "artist": r.artist, "title": r.title, "album": r.album, "era": era, "pad_s": PADS[era],
                    "in_pool": tier is not None if r.artist else None, "pool_match_tier": tier,
                    "rotation": r.rotation, "reorder_flag": flag, "play_order_status": status,
                    "carryover": carryover, "track_rows": len(tracks),
                    "talk_rows": sum(r.entry_type in TALK_TYPES for r in rows),
                }  # fmt: skip
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            log.info("%s: %d plays (%d carryover)", key, len(plays), len(plays) - len(tracks))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
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
