"""Tests for evaluation.corpus: the flowsheet export to plays.jsonl.

Every row is synthetic. Times are written in UTC; in August, Eastern local time
is UTC-4, so 20:00 UTC is the archive hour ``…1600.mp3``.
"""

from __future__ import annotations

import csv
import io
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from evaluation import corpus
from evaluation.pool import SCHEMA

UTC = timezone.utc
COLUMNS = [
    "id",
    "show_id",
    "play_order",
    "legacy_entry_id",
    "entry_type",
    "add_time",
    "artist_name",
    "track_title",
    "album_title",
    "rotation_id",
    "album_id",
]
POOL: list[tuple[str, str | None, str | None, str]] = [
    ("Juana Molina", "DOGA", "la paradoja", "mp3"),
    ("Jessica Pratt", "On Your Own Love Again", "Back, Baby", "flac"),
    ("Chuquimamani-Condori", "Edits", "Call Your Name", "mp4"),
    # Untagged album and title: joinable on neither.
    ("Hermanos Gutiérrez", None, None, "wav"),
    # Japanese script: its fuzzy keys are real, never empty.
    ("ジェシカ・プラット", "ザ・ワーム", "ザ・ワーム", "aac"),
    ("Хуана Молина", "Дога", "Парадоха", "ogg"),
    # A Latin name tagged with a full-width native-script clause, as Phase 1's pool has.
    ("Stereolab（ステレオラブ）", "Dots and Loops（ドッツ・アンド・ループス）", "Brakhage", "m4a"),
]


def ts(hour: int, minute: int = 0, second: float = 0, day: int = 12) -> str:
    """A Postgres-style ``add_time`` for 2026-08-<day> <hour>:<minute> UTC."""
    t = datetime(2026, 8, day, hour, minute, tzinfo=UTC) + timedelta(seconds=second)
    return t.strftime("%Y-%m-%d %H:%M:%S.") + f"{t.microsecond // 1000:03d}+00"


def row(id_: int, add_time: str, entry_type: str = "track", **kw: object) -> dict[str, object]:
    return {
        "id": id_,
        "show_id": kw.get("show_id", 1),
        "play_order": kw.get("play_order", id_),
        "legacy_entry_id": kw.get("legacy_entry_id", ""),
        "entry_type": entry_type,
        "add_time": add_time,
        "artist_name": kw.get("artist", "Hermanos Gutiérrez" if entry_type == "track" else ""),
        "track_title": kw.get("title", "Hijo del Sol" if entry_type == "track" else ""),
        "album_title": kw.get("album", "Hijo del Sol" if entry_type == "track" else ""),
        "rotation_id": kw.get("rotation_id", ""),
        "album_id": "",
    }


def write_export(tmp_path: Path, rows: list[dict[str, object]], last_run: str) -> Path:
    export = tmp_path / "export"
    export.mkdir()
    with open(export / "flowsheet.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    (export / "cronjob_runs.csv").write_text(f"job_name,last_run\nflowsheet-etl,{last_run}\n")
    return export


@pytest.fixture
def pool_db(tmp_path: Path) -> Path:
    path = tmp_path / "pool.db"
    db = sqlite3.connect(path)
    db.execute(SCHEMA)
    for i, (artist, album, title, fmt) in enumerate(POOL):
        db.execute(
            "INSERT INTO files (key, stage_id, prefix, format, size, artist, album, title, status)"
            " VALUES (?, ?, 'rotation/', ?, 1, ?, ?, ?, 'indexed')",
            (f"rotation/{i}.{fmt}", f"{i:040x}", fmt, artist, album, title),
        )
    db.commit()
    db.close()
    return path


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2026-08-12 20:31:05.123+00", datetime(2026, 8, 12, 20, 31, 5, 123000, tzinfo=UTC)),
        ("2026-08-12 20:31:05.1+00", datetime(2026, 8, 12, 20, 31, 5, 100000, tzinfo=UTC)),
        ("2026-08-12 20:31:05+00", datetime(2026, 8, 12, 20, 31, 5, tzinfo=UTC)),
        ("2026-08-12 16:31:05-04:00", datetime(2026, 8, 12, 20, 31, 5, tzinfo=UTC)),
    ],
)
def test_parse_add_time(text: str, expected: datetime) -> None:
    assert corpus.parse_add_time(text) == expected


@pytest.mark.parametrize(
    ("last_run", "expected"),
    [
        # last_run plus one half-hourly run.
        ("2026-08-09 05:00:43.537+00", datetime(2026, 8, 9, 5, 30, 43, 537000, tzinfo=UTC)),
        # A watermark before the code change never widens the era: the floor binds.
        ("2026-08-01 00:00:00+00", corpus.ETL_STOP_FLOOR),
    ],
)
def test_etl_stop(tmp_path: Path, last_run: str, expected: datetime) -> None:
    export = write_export(tmp_path, [], last_run)
    assert corpus.etl_stop(export) == expected


@pytest.mark.parametrize(
    ("s", "folded", "album_key", "fuzzy"),
    [
        ("Hermanos Gutiérrez", "hermanos gutierrez", "hermanos gutiérrez", "hermanos gutierrez"),
        ("Nilüfer Yanya", "nilufer yanya", "nilüfer yanya", "nilufer yanya"),
        ("DOGA (Deluxe Edition)", "doga (deluxe edition)", "doga", "doga"),
        ("Edits [Remastered]", "edits [remastered]", "edits", "edits"),
        (
            "Call  Your Name (feat. Someone)",
            "call  your name (feat. someone)",
            "call your name",
            "call your name",
        ),
        ("Back, Baby", "back, baby", "back, baby", "back baby"),
        # Cruft words match whole words only: the exact key keeps "(Feathers)", and
        # only the fuzzy key, which drops every bracketed clause, loses it.
        ("Ghost (Feathers)", "ghost (feathers)", "ghost (feathers)", "ghost"),
        ("Halo (ft. Someone)", "halo (ft. someone)", "halo", "halo"),
        (None, "", "", ""),
        # The fuzzy key drops bracketed clauses after NFKD, full-width ones included.
        ("The Worm（ザ・ワーム）", "the worm(サ・ワーム)", "the worm（ザ・ワーム）", "the worm"),
        (
            "Edits {Live} [Tape] (Demo)",
            "edits {live} [tape] (demo)",
            "edits {live} [tape] (demo)",
            "edits",
        ),
        # An entirely bracketed name has no fuzzy key, so it never joins on that tier.
        ("（ザ・ワーム）", "(サ・ワーム)", "（ザ・ワーム）", ""),
        # Letters and digits of any script survive the fuzzy key; diacritics still fold.
        ("ザ・ワーム", "サ・ワーム", "ザ・ワーム", "サ ワーム"),
        ("Хуана Молина", "хуана молина", "хуана молина", "хуана молина"),
        ("Csillagrablók_2", "csillagrablok_2", "csillagrablók_2", "csillagrablok 2"),
    ],
)
def test_normalizers(s: str | None, folded: str, album_key: str, fuzzy: str) -> None:
    assert corpus.fold(s) == folded
    assert corpus.album_key(s) == album_key
    assert corpus.fuzzy(s) == fuzzy


@pytest.mark.parametrize(
    ("artist", "album", "title", "tier", "pool_format"),
    [
        ("Juana Molina", "DOGA", "anything", "exact", "mp3"),
        ("juana molina", "DOGA (Deluxe Edition)", "anything", "exact", "mp3"),
        ("Jessica Pratt", "On Your Own Love Again!", "anything", "fuzzy", "flac"),
        ("Chuquimamani-Condori", "Some Compilation", "Call Your Name", "title", "mp4"),
        ("Chuquimamani Condori", "Some Compilation", "call your name", "title", "mp4"),
        ("Juana Molina", "Halo", "Paraguaya", None, None),
        ("", "", "", None, None),
        # An empty key never joins, on the play's side or the pool file's.
        ("Hermanos Gutiérrez", "", "", None, None),
        ("Hermanos Gutiérrez", "", "Hijo del Sol", None, None),
        ("Hermanos Gutiérrez", "Hijo del Sol", "", None, None),
        # Non-Latin scripts keep real keys: an unrelated Cyrillic play never joins.
        ("Хуана Молина", "Сон", "Сон", None, None),
        ("ジェシカ・プラット", "ザ・ワーム!", "x", "fuzzy", "aac"),
        ("ジェシカ・プラット", "Other", "ザ・ワーム", "title", "aac"),
        ("Хуана Молина", "Дога!", "x", "fuzzy", "ogg"),
        # A Latin play joins a pool tag that adds a bracketed native-script clause...
        ("Stereolab", "Dots and Loops", "x", "fuzzy", "m4a"),
        ("Stereolab", "Other", "Brakhage", "title", "m4a"),
        # ...but an unrelated name in the same script still does not.
        ("Stereolab", "Dots and Dashes", "x", None, None),
    ],
)
def test_pool_index_tiers(
    pool_db: Path, artist: str, album: str, title: str, tier: str | None, pool_format: str | None
) -> None:
    index = corpus.PoolIndex.load(pool_db)
    assert index.tier(artist, album, title) == tier
    assert index.match(artist, album, title) == ((tier, pool_format) if tier else None)


def test_pool_format_is_the_first_matching_file_by_key(tmp_path: Path) -> None:
    path = tmp_path / "pool.db"
    db = sqlite3.connect(path)
    db.execute(SCHEMA)
    for i, (key, fmt) in enumerate([("rotation/b.mp3", "mp3"), ("rotation/a.wav", "wav")]):
        db.execute(
            "INSERT INTO files (key, stage_id, prefix, format, size, artist, album, title, status)"
            " VALUES (?, ?, 'rotation/', ?, 1, 'Nilüfer Yanya', 'PAINLESS', 't', 'indexed')",
            (key, f"{i:040x}", fmt),
        )
    db.commit()
    db.close()
    assert corpus.PoolIndex.load(path).match("Nilüfer Yanya", "PAINLESS", "t") == ("exact", "wav")


def test_pool_index_never_creates_a_missing_pool_db(tmp_path: Path) -> None:
    missing = tmp_path / "pool.db"  # its directory exists, so a writable open would create it
    with pytest.raises(sqlite3.OperationalError):
        corpus.PoolIndex.load(missing)
    assert not missing.exists()


def test_pool_index_matches_album_artist(tmp_path: Path) -> None:
    path = tmp_path / "pool.db"
    db = sqlite3.connect(path)
    db.execute(SCHEMA)
    db.execute(
        "INSERT INTO files (key, stage_id, prefix, format, size, artist, album_artist, album, title, status)"
        " VALUES ('k.mp3', ?, 'p/', 'mp3', 1, 'Duke Ellington', 'Duke Ellington & John Coltrane',"
        " 'Duke Ellington & John Coltrane', 'In a Sentimental Mood', 'indexed')",
        ("0" * 40,),
    )
    db.commit()
    db.close()
    index = corpus.PoolIndex.load(path)
    assert (
        index.tier("Duke Ellington & John Coltrane", "Duke Ellington & John Coltrane", "x")
        == "exact"
    )


def test_failed_pool_rows_are_not_in_pool(tmp_path: Path) -> None:
    path = tmp_path / "pool.db"
    db = sqlite3.connect(path)
    db.execute(SCHEMA)
    db.execute(
        "INSERT INTO files (key, stage_id, prefix, format, size, artist, album, title, status)"
        " VALUES ('k.mp3', ?, 'p/', 'mp3', 1, 'Juana Molina', 'DOGA', 't', 'failed')",
        ("0" * 40,),
    )
    db.commit()
    db.close()
    assert corpus.PoolIndex.load(path).tier("Juana Molina", "DOGA", "t") is None


def read_plays(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_plays_windows_pads_and_carryover(tmp_path: Path, pool_db: Path) -> None:
    rows = [
        row(
            1,
            ts(19, 55),
            artist="Jessica Pratt",
            album="On Your Own Love Again",
            title="Back, Baby",
        ),
        row(
            2, ts(20, 2), artist="Juana Molina", album="DOGA", title="la paradoja", rotation_id="7"
        ),
        row(3, ts(20, 30), "talkset"),
        row(4, ts(20, 40)),
    ]
    export = write_export(tmp_path, rows, "2026-08-09 05:00:43+00")
    out = tmp_path / "plays.jsonl"
    corpus.write_plays(
        out,
        ["2026/08/12/202608121600.mp3"],
        corpus.Flowsheet.load(export),
        corpus.PoolIndex.load(pool_db),
    )
    carry, first, last = read_plays(out)
    assert carry["carryover"] is True and carry["play_id"] == 1
    assert carry["t_offset_s"] == -300.0
    assert (carry["window_start_s"], carry["window_end_s"]) == (0.0, 120.0 + 180.0)
    assert first["carryover"] is False and first["play_id"] == 2
    assert (first["t_offset_s"], first["window_start_s"], first["window_end_s"]) == (
        120.0,
        0.0,
        2400.0 + 180.0,
    )
    assert (
        first["in_pool"],
        first["pool_match_tier"],
        first["pool_format"],
        first["rotation"],
        first["pad_s"],
    ) == (True, "exact", "mp3", True, 180.0)
    assert (last["window_start_s"], last["window_end_s"]) == (2400.0 - 180.0, 3600.0)
    assert (last["in_pool"], last["pool_match_tier"], last["pool_format"], last["rotation"]) == (
        False,
        None,
        None,
        False,
    )
    assert (first["track_rows"], first["talk_rows"], first["era"]) == (2, 1, "canonical")
    assert set(first) == {
        "hour_key", "play_id", "t_offset_s", "window_start_s", "window_end_s", "artist", "title", "album",
        "era", "pad_s", "in_pool", "pool_match_tier", "pool_format", "rotation", "reorder_flag",
        "play_order_status", "carryover", "track_rows", "talk_rows",
    }  # fmt: skip


def plays_for(
    tmp_path: Path, pool_db: Path, rows: list[dict[str, object]], hour: str
) -> list[dict[str, object]]:
    """Write ``plays.jsonl`` for one hour of a synthetic export and read it back."""
    export = write_export(tmp_path, rows, "2026-08-09 05:00:43+00")
    out = tmp_path / "plays.jsonl"
    corpus.write_plays(out, [hour], corpus.Flowsheet.load(export), corpus.PoolIndex.load(pool_db))
    return read_plays(out)


@pytest.mark.parametrize("artist", ["", "   "])
def test_a_play_without_an_artist_has_unknown_pool_status(
    tmp_path: Path, pool_db: Path, artist: str
) -> None:
    (play,) = plays_for(
        tmp_path, pool_db, [row(1, ts(20, 10), artist=artist)], "2026/08/12/202608121600.mp3"
    )
    assert (play["in_pool"], play["pool_match_tier"], play["pool_format"]) == (None, None, None)


@pytest.mark.parametrize(
    "rows",
    [
        # A legacy-mirrored talkset makes the show two-writer, though no track row is legacy.
        [
            row(1, ts(20, 10), play_order=1),
            row(2, ts(20, 20), "talkset", play_order=2, legacy_entry_id="9001"),
            row(3, ts(20, 30), play_order=3),
        ],
        # Tracks with no show have no play order to check, whatever their play_order says.
        [
            row(1, ts(20, 10), show_id="", play_order=2),
            row(2, ts(20, 30), show_id="", play_order=1),
        ],
    ],
)
def test_play_order_is_unreliable_beyond_single_writer_track_rows(
    tmp_path: Path, pool_db: Path, rows: list[dict[str, object]]
) -> None:
    plays = plays_for(tmp_path, pool_db, rows, "2026/08/12/202608121600.mp3")
    assert {(p["play_order_status"], p["reorder_flag"]) for p in plays} == {("unreliable", None)}


def test_the_hour_after_fall_back_gets_its_carryover(tmp_path: Path, pool_db: Path) -> None:
    # 06:58 UTC is 01:58 EST in the repeated 01:00 hour, which has no hour key of its own.
    rows = [row(1, "2026-11-01 06:58:00+00"), row(2, "2026-11-01 07:10:00+00")]
    carry, play = plays_for(tmp_path, pool_db, rows, "2026/11/01/202611010200.mp3")
    assert (carry["play_id"], carry["carryover"], carry["t_offset_s"]) == (1, True, -120.0)
    assert (play["play_id"], play["carryover"]) == (2, False)


def test_an_hour_with_no_rows_is_logged_as_a_warning(
    tmp_path: Path, pool_db: Path, caplog: pytest.LogCaptureFixture
) -> None:
    plays = plays_for(tmp_path, pool_db, [row(1, ts(20, 10))], "2021/06/01/202106011600.mp3")
    assert plays == []
    assert any(
        r.levelname == "WARNING" and "no flowsheet rows" in r.message for r in caplog.records
    )


class FullDisk(io.FileIO):
    """A real file whose every write lands ten bytes and then fails, as a full disk does."""

    def write(self, b: Any) -> int:
        super().write(bytes(b)[:10])
        raise OSError(28, "No space left on device")


# write_through sends each write to the disk at once, so writelines fails; without
# it the text sits in the buffer and the failure comes at the final flush in close.
@pytest.mark.parametrize("write_through", [True, False], ids=["in-writelines", "at-close"])
def test_a_failed_write_leaves_no_partial_file(
    tmp_path: Path, pool_db: Path, monkeypatch: pytest.MonkeyPatch, write_through: bool
) -> None:
    sheet = corpus.Flowsheet.load(
        write_export(tmp_path, [row(1, ts(20, 10))], "2026-08-09 05:00:43+00")
    )

    def full_disk_open(path: Path, mode: str, **kwargs: Any) -> io.TextIOWrapper:
        return io.TextIOWrapper(FullDisk(path, mode), write_through=write_through, **kwargs)

    monkeypatch.setattr(corpus, "open", full_disk_open, raising=False)
    out = tmp_path / "plays.jsonl"
    with pytest.raises(OSError, match="No space"):
        corpus.write_plays(
            out, ["2026/08/12/202608121600.mp3"], sheet, corpus.PoolIndex.load(pool_db)
        )
    assert not out.exists()


def test_cli_help_works_without_docstrings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(corpus, "__doc__", None)  # as under python -OO
    with pytest.raises(SystemExit) as exit_:
        corpus.main(["--help"])
    assert exit_.value.code == 0


@pytest.mark.parametrize(
    ("delta", "era"),
    [
        (timedelta(microseconds=-1), "etl"),
        (timedelta(0), "canonical"),
        (timedelta(seconds=1), "canonical"),
    ],
)
def test_era_boundary_is_etl_stop(tmp_path: Path, delta: timedelta, era: str) -> None:
    stop = datetime(2026, 8, 9, 5, 30, 43, tzinfo=UTC)  # last_run plus 30 minutes
    add_time = (stop + delta).strftime("%Y-%m-%d %H:%M:%S.%f+00")
    sheet = corpus.Flowsheet.load(
        write_export(tmp_path, [row(1, add_time)], "2026-08-09 05:00:43+00")
    )
    assert sheet.stop == stop
    assert sheet.era(sheet.rows[0].add_time) == era


def test_etl_era_uses_its_pad(tmp_path: Path, pool_db: Path) -> None:
    rows = [row(1, ts(20, 10, day=5)), row(2, ts(20, 40, day=5))]  # before ETL_STOP
    export = write_export(tmp_path, rows, "2026-08-09 05:00:43+00")
    out = tmp_path / "plays.jsonl"
    corpus.write_plays(
        out,
        ["2026/08/05/202608051600.mp3"],
        corpus.Flowsheet.load(export),
        corpus.PoolIndex.load(pool_db),
    )
    first, second = read_plays(out)
    assert (first["era"], first["pad_s"], first["play_order_status"], first["reorder_flag"]) == (
        "etl",
        220.0,
        "etl",
        None,
    )
    assert (first["window_end_s"], second["window_start_s"]) == (2400.0 + 220.0, 2400.0 - 220.0)


def test_rows_order_by_add_time_then_id(tmp_path: Path, pool_db: Path) -> None:
    rows = [row(9, ts(20, 10)), row(3, ts(20, 10)), row(5, ts(20, 5))]
    export = write_export(tmp_path, rows, "2026-08-09 05:00:43+00")
    assert [r.id for r in corpus.Flowsheet.load(export).rows] == [5, 3, 9]


@pytest.mark.parametrize(
    ("show_rows", "status", "flag"),
    [
        # Single writer, play_order agrees with (add_time, id): not flagged.
        ([(1, 20, 1, ""), (2, 30, 2, "")], "single_writer", False),
        # Single writer, a late-logged track moved up by the DJ: flagged.
        ([(1, 20, 2, ""), (2, 30, 1, "")], "single_writer", True),
        # Any legacy key means two writers: play_order is unreliable, never flagged.
        ([(1, 20, 2, "9001"), (2, 30, 1, "")], "unreliable", None),
    ],
)
def test_play_order_status(
    tmp_path: Path,
    pool_db: Path,
    show_rows: list[tuple[int, int, int, str]],
    status: str,
    flag: bool | None,
) -> None:
    rows = [row(i, ts(20, m), play_order=po, legacy_entry_id=leg) for i, m, po, leg in show_rows]
    export = write_export(tmp_path, rows, "2026-08-09 05:00:43+00")
    out = tmp_path / "plays.jsonl"
    corpus.write_plays(
        out,
        ["2026/08/12/202608121600.mp3"],
        corpus.Flowsheet.load(export),
        corpus.PoolIndex.load(pool_db),
    )
    assert {(p["play_order_status"], p["reorder_flag"]) for p in read_plays(out)} == {
        (status, flag)
    }


def test_write_plays_never_overwrites(tmp_path: Path, pool_db: Path) -> None:
    export = write_export(tmp_path, [row(1, ts(20, 10))], "2026-08-09 05:00:43+00")
    out = tmp_path / "plays.jsonl"
    out.write_text("kept\n")
    with pytest.raises(FileExistsError):
        corpus.write_plays(
            out,
            ["2026/08/12/202608121600.mp3"],
            corpus.Flowsheet.load(export),
            corpus.PoolIndex.load(pool_db),
        )
    assert out.read_text() == "kept\n"


def test_export_sql_is_read_only() -> None:
    sql = (Path(corpus.__file__).parent / "sql" / "flowsheet-export.sql").read_text()
    statements = [line for line in sql.splitlines() if line.startswith("\\copy")]
    assert len(statements) == 2
    assert all("SELECT" in s and "INSERT" not in s and "UPDATE" not in s for s in statements)


def test_write_plays_leaves_no_partial_file_on_a_bad_key(tmp_path: Path, pool_db: Path) -> None:
    export = write_export(tmp_path, [row(1, ts(20, 10))], "2026-08-09 05:00:43+00")
    out = tmp_path / "plays.jsonl"
    with pytest.raises(ValueError):
        corpus.write_plays(
            out,
            [
                "2026/08/12/202608121600.mp3",
                "2026/11/01/202611010100.mp3",
            ],  # the second is the fall-back hour
            corpus.Flowsheet.load(export),
            corpus.PoolIndex.load(pool_db),
        )
    assert not out.exists()


def test_etl_stop_without_a_flowsheet_etl_row_says_so(tmp_path: Path) -> None:
    export = tmp_path / "export"
    export.mkdir()
    (export / "cronjob_runs.csv").write_text(
        "job_name,last_run\nother-job,2026-08-09 05:00:43+00\n"
    )
    with pytest.raises(ValueError, match="no flowsheet-etl row"):
        corpus.etl_stop(export)
