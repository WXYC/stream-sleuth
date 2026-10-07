"""Tests for evaluation.corpus: the flowsheet export to plays.jsonl.

Every row is synthetic. Times are written in UTC; in August, Eastern local time
is UTC-4, so 20:00 UTC is the archive hour ``…1600.mp3``.
"""

from __future__ import annotations

import csv
import io
import itertools
import json
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from evaluation import corpus, names
from evaluation.pool import SCHEMA
from evaluation.run import read_hours
from stream_sleuth import paths
from stream_sleuth.paths import CHECKOUT, DataPathError

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
        # Every tier refuses a play naming another recording, in its title or its album,
        # that the matched file does not name: each of these joins at the tier noted
        # without the qualifier.
        ("Jessica Pratt", "On Your Own Love Again", "Back, Baby (Live)", None, None),  # exact
        ("Jessica Pratt", "On Your Own Love Again!", "Back, Baby (Demo)", None, None),  # fuzzy
        ("Stereolab", "Dots and Loops (Live)", "Brakhage", None, None),  # title, exact key
        ("Stereolab", "Dots & Loops [Live]", "brakhage!", None, None),  # title, fuzzy key
        ("Juana Molina", "DOGA (Remixes)", "la paradoja", None, None),  # title, exact key
        # A same-recording clause names no other recording, so it does not refuse.
        ("Jessica Pratt", "On Your Own Love Again", "Back, Baby (Radio Edit)", "exact", "flac"),
        ("Stereolab", "Dots and Loops (2011 Remastered Version)", "Brakhage", "fuzzy", "m4a"),
        # Edition and featuring clauses are cruft on every tier, so the exact tier joins.
        *(
            ("Jessica Pratt", f"On Your Own Love Again ({edition})", "Back, Baby", "exact", "flac")
            for edition in ("Deluxe Version", "Bonus Track Version", "Expanded Version")
        ),
        *(
            ("Jessica Pratt", "On Your Own Love Again", f"Back, Baby ({clause})", "exact", "flac")
            for clause in ("Remastered 2011 Version", "feat. Mix Master Mike", "ft. Live Skull")
        ),
        # So do an edition phrase that does not lead its clause, and a "with" credit.
        *(
            ("Jessica Pratt", f"On Your Own Love Again ({edition})", "Back, Baby", "exact", "flac")
            for edition in (
                "2011 Deluxe Version",
                "Special Version",
                "2011 Expanded Version",
                "25th Anniversary Version",
                "Japan Bonus Track Version",
                "with Live Skull",
            )
        ),
        (
            "Jessica Pratt",
            "On Your Own Love Again",
            "Back, Baby (with Live Skull)",
            "exact",
            "flac",
        ),
        # A full-width bracket is folded by the fuzzy keys only, so the album joins there; a
        # play's title joins on the exact album key, which never sees the clause. The fuzzy
        # key and the qualifiers read each clause after the fold, whichever bracket it has.
        *(
            (
                "Jessica Pratt",
                f"On Your Own Love Again{wide[0]}{clause}{wide[1]}",
                "Back, Baby",
                "fuzzy",
                "flac",
            )
            for wide in ("（）", "［］")
            for clause in (
                "Remastered 2011 Version",
                "Deluxe Edition Version",
                "Super Deluxe Version",
            )
        ),
        *(
            ("Jessica Pratt", "On Your Own Love Again", f"Back, Baby（{clause}）", "exact", "flac")
            for clause in (
                "Remastered 2011 Version",
                "Deluxe Edition Version",
                "Super Deluxe Version",
            )
        ),
    ],
)
def test_pool_index_tiers(
    pool_db: Path, artist: str, album: str, title: str, tier: str | None, pool_format: str | None
) -> None:
    index = corpus.PoolIndex.load(pool_db)
    assert index.tier(artist, album, title) == tier
    assert index.match(artist, album, title) == ((tier, pool_format) if tier else None)


@pytest.mark.parametrize(
    ("title", "tier"),
    [
        # A play logged as another recording never joins the studio file...
        ("Back, Baby (Live)", None),
        ("Back, Baby (Remix)", None),
        ("Back, Baby [Demo]", None),
        ("Back, Baby（Instrumental）", None),
        ("Back, Baby (Live) (ft. Someone)", None),
        # Words split as the key splits them, so an underscore separates too.
        ("Back, Baby (Live_Session)", None),
        ("Back, Baby (Live_Version)", None),
        # Plural and past forms are qualifiers.
        *(
            (f"Back, Baby ({word})", None)
            for word in "Remixes, Remixed, Demos, Peel Sessions, Versions, Edits, Mixes".split(", ")
        ),
        # A version clause opening with a cruft word is a version clause, not cruft.
        ("Back, Baby (Bonus Live Track)", None),
        ("Back, Baby (Bonus Track - Demo)", None),
        ("Back, Baby (Remastered Live Version)", None),
        # A lone "version" still names another recording.
        ("Back, Baby (Alternate Version)", None),
        ("Back, Baby (Extended Version)", None),
        # ...but a bracket that names no recording is still dropped...
        ("Back, Baby (Feathers)", "title"),
        ("Back, Baby（ザ・ワーム）", "title"),
        ("Back, Baby (Remastered 2011)", "title"),
        # ...and so is one that names the same recording.
        *(
            (f"Back, Baby ({phrase})", "title")
            for phrase in (
                "Radio Edit, Single Edit, FCC Edit, Clean_Edit, Album Version, Single Version, "
                "Original Mix, 2011 Remaster, 2011 Remastered Version, Mono Version, "
                "Stereo Version, Mono, LP Version, Clean Version, Explicit Version, "
                "Radio Version, Original Version, Edited Version, Edited, Clean, Explicit"
            ).split(", ")
        ),
        # ...and so are edition and featuring clauses, which are cruft.
        *(
            (f"Back, Baby ({clause})", "title")
            for clause in (
                "Deluxe Version, Bonus Track Version, Remastered 2011 Version, "
                "feat. Mix Master Mike, ft. Live Skull, with Live Skull, 2011 Deluxe Version, "
                "Super Deluxe Version, Special Version, Expanded Version, Anniversary Version, "
                "2011 Expanded Version, 25th Anniversary Version, Japan Bonus Track Version"
            ).split(", ")
        ),
        # A full-width clause is read after the fold: "version" beside words that no
        # same-recording phrase covers is still an edition clause.
        *(
            (f"Back, Baby{wide[0]}{clause}{wide[1]}", "title")
            for wide in ("（）", "［］")
            for clause in ("Deluxe Version", "Remastered 2011 Version", "Deluxe Edition Version")
        ),
    ],
)
def test_version_qualified_plays_do_not_join_the_studio_title(
    pool_db: Path, title: str, tier: str | None
) -> None:
    assert corpus.PoolIndex.load(pool_db).tier("Jessica Pratt", "Other", title) == tier


# (artist, album_artist, album, title): files that name a version in their album, in brackets
# or after a dash in their title, and files credited to two names.
AGREEMENT_POOL: list[tuple[str, str | None, str, str]] = [
    ("Jessica Pratt", None, "Live at KEXP", "Back, Baby"),
    ("Jessica Pratt", None, "On Your Own Love Again", "Back, Baby (Live)"),
    ("Jessica Pratt", None, "Bootleg", "Back, Baby - Demo"),
    ("Juana Molina", "Various Artists", "DOGA", "la paradoja"),
    ("Stereolab（ステレオラブ）", "Stereolab", "Dots and Loops", "Brakhage"),
    ("Hermanos Gutiérrez", None, "Session 9", "Hijo del Sol"),
    ("U.S. Girls", None, "Heavy Light", "Overtime"),
    ("A.R. Kane", "Various Artists", "69", "Baby Milk Snatcher"),
]


def test_the_title_tier_is_names_title_tier(tmp_path: Path) -> None:
    """The join's title tier and the shared predicate agree over files that name versions and
    files with an album artist, so a scorer calling the predicate stands where the join does.
    The play's album matches no pool album, so no album tier masks the title tier."""
    path = tmp_path / "pool.db"
    db = sqlite3.connect(path)
    db.execute(SCHEMA)
    for i, (artist, album_artist, album, title) in enumerate(AGREEMENT_POOL):
        db.execute(
            "INSERT INTO files (key, stage_id, prefix, format, size, artist, album_artist, album,"
            " title, status) VALUES (?, ?, 'rotation/', 'mp3', 1, ?, ?, ?, ?, 'indexed')",
            (f"rotation/{i}.mp3", f"{i:040x}", artist, album_artist, album, title),
        )
    db.commit()
    db.close()
    index = corpus.PoolIndex.load(path)
    artists = [
        "Jessica Pratt",
        "Juana Molina",
        "Various Artists",
        "Stereolab",
        "Hermanos Gutiérrez",
        "US Girls",
        "AR Kane",
        "J. Mascis",
    ]
    albums = ["Other", "Other [Live]", "Other (Demo)", "Other (Remixed)"]
    titles = ["Back, Baby", "Back, Baby (Live)", "Back, Baby - Demo", "Back, Baby - Live"]
    titles += ["BACK BABY", "la paradoja", "Brakhage", "Hijo del Sol", "Hijo del Sol (Session)"]
    titles += ["Overtime", "Baby Milk Snatcher"]

    seen = set()
    for artist, album, title in itertools.product(artists, albums, titles):
        expected = any(
            names.title_tier(artist, album, title, (a, aa), file_album, file_title)
            for a, aa, file_album, file_title in AGREEMENT_POOL
        )
        assert (index.tier(artist, album, title) == "title") == expected, (artist, album, title)
        seen.add(expected)

    assert seen == {True, False}


@pytest.mark.parametrize(
    ("pool_artist", "play_artist", "tier"),
    [
        # A dotted initialism and the same letters undotted join, on the fuzzy tier only.
        ("A.R. Kane", "AR Kane", "fuzzy"),
        ("AR Kane", "A.R. Kane", "fuzzy"),
        ("A.R. Kane", "A.R. Kane", "exact"),
        ("R.E.M.", "REM", "fuzzy"),
        ("REM", "R.E.M.", "fuzzy"),
        # A single initial is not an initialism.
        ("J. Mascis", "JM Mascis", None),
        ("J. Mascis", "J Mascis", "fuzzy"),
    ],
)
def test_a_dotted_initialism_joins_the_same_letters_on_the_fuzzy_tier(
    tmp_path: Path, pool_artist: str, play_artist: str, tier: str | None
) -> None:
    path = tmp_path / "pool.db"
    db = sqlite3.connect(path)
    db.execute(SCHEMA)
    db.execute(
        "INSERT INTO files (key, stage_id, prefix, format, size, artist, album, title, status)"
        " VALUES ('k.mp3', ?, 'p/', 'mp3', 1, ?, '69', 'Baby Milk Snatcher', 'indexed')",
        (f"{0:040x}", pool_artist),
    )
    db.commit()
    db.close()
    index = corpus.PoolIndex.load(path)
    assert index.tier(play_artist, "69", "Baby Milk Snatcher") == tier
    assert index.tier(play_artist, "Other", "Baby Milk Snatcher") == (
        None if tier is None else "title"
    )


def jessica_pratt_pool(tmp_path: Path, files: list[tuple[str, str, str]]) -> Path:
    """A pool.db of Jessica Pratt files, ``(key, album, title)``, format from the key."""
    path = tmp_path / "pool.db"
    db = sqlite3.connect(path)
    db.execute(SCHEMA)
    for i, (key, album, title) in enumerate(files):
        db.execute(
            "INSERT INTO files (key, stage_id, prefix, format, size, artist, album, title, status)"
            " VALUES (?, ?, 'p/', ?, 1, 'Jessica Pratt', ?, ?, 'indexed')",
            (key, f"{i:040x}", key.rsplit(".", 1)[1], album, title),
        )
    db.commit()
    db.close()
    return path


@pytest.mark.parametrize(
    ("pool_title", "play_title", "tier"),
    [
        # The play's exact title key differs (no comma), so only the fuzzy key can join it.
        ("Back, Baby (Live)", "Back Baby (LIVE)", "title"),
        ("Back, Baby (Live)", "back baby [live]", "title"),
        ("Back, Baby [Demo]", "Back, Baby（Demo）", "title"),
        ("Back, Baby (Live_Session)", "Back, Baby [Live Session]", "title"),
        ("Back, Baby (Live)", "Back, Baby (Demo)", None),
        ("Back, Baby (Live)", "Back, Baby", None),
        # "(Live Version)" is "(Live)", in either direction.
        ("Back, Baby (Live)", "Back, Baby (Live Version)", "title"),
        ("Back, Baby (Live Version)", "Back, Baby (Live)", "title"),
        # A qualifier joins its plural and past forms, on either side.
        ("Back, Baby (Remixed)", "Back, Baby (Remix)", "title"),
        ("Back, Baby (Remix)", "Back, Baby (Remixed)", "title"),
        # A dotted qualifier is the undotted one, whichever side carries the dots.
        ("Back, Baby (L.P. Version)", "Back, Baby (LP Version)", "title"),
        ("Back, Baby (LP Version)", "Back, Baby (L.P. Version)", "title"),
        ("Back, Baby (F.C.C. Edit)", "Back, Baby (FCC Edit)", "title"),
        ("Back, Baby (FCC Edit)", "Back, Baby (F.C.C. Edit)", "title"),
        ("Back, Baby (L.I.V.E.)", "Back, Baby (Live)", "title"),
        ("Back, Baby (Live)", "Back, Baby (L.I.V.E.)", "title"),
        ("Back, Baby (L.I.V.E.)", "Back, Baby", None),
        ("Back, Baby (Peel Sessions)", "Back, Baby (Peel Session)", "title"),
        ("Back, Baby (Peel Session)", "Back, Baby (Peel Sessions)", "title"),
        # A pool file naming its version outside brackets names it all the same.
        ("Back, Baby - Live", "Back, Baby (Live)", "title"),
        # A pool file naming the same recording still joins the bare title.
        ("Back, Baby (Radio Edit)", "Back, Baby", "title"),
        ("Back, Baby (2011 Remastered Version)", "Back, Baby", "title"),
    ],
)
def test_the_same_qualifier_on_both_sides_joins(
    tmp_path: Path, pool_title: str, play_title: str, tier: str | None
) -> None:
    path = jessica_pratt_pool(tmp_path, [("k.mp3", "Bootleg", pool_title)])
    assert corpus.PoolIndex.load(path).tier("Jessica Pratt", "Other", play_title) == tier


@pytest.mark.parametrize(
    ("play_title", "match"),
    [("Back, Baby (Live)", ("exact", "flac")), ("Back, Baby", ("exact", "mp3"))],
)
def test_an_album_tier_play_joins_the_first_file_naming_its_version(
    tmp_path: Path, play_title: str, match: tuple[str, str]
) -> None:
    files = [("a.mp3", "Bootleg", "Back, Baby"), ("b.flac", "Bootleg", "Back, Baby (Live)")]
    path = jessica_pratt_pool(tmp_path, files)
    assert corpus.PoolIndex.load(path).match("Jessica Pratt", "Bootleg", play_title) == match


@pytest.mark.parametrize(
    ("pool_album", "pool_title", "play_album", "tier"),
    [
        # The pool names the version in its album anywhere, in its title bracketed or after
        # a dash.
        ("Live at KEXP", "Back, Baby", "Live at KEXP", "exact"),
        ("Bootleg", "Back, Baby - Live", "Bootleg", "exact"),
        ("Bootleg", "Back, Baby - Live", "Other", "title"),
        # An en dash, an em dash, or a Unicode hyphen with spaces is a " - " too.
        *(
            ("Bootleg", f"Back, Baby {dash} Live", album, tier)
            for dash in ("\u2010", "\u2013", "\u2014")
            for album, tier in (("Bootleg", "exact"), ("Other", "title"))
        ),
        ("Bootleg", "Back, Baby (Live Version)", "Bootleg", "exact"),
        # A same-recording phrase names no version there either.
        ("Bootleg", "Back, Baby - Mono", "Bootleg", None),
        ("Bootleg", "Back, Baby", "Bootleg", None),
        # A title names it only after " - ": a sibling track whose name holds the word
        # ("Live Forever") never makes the album satisfy a qualified play.
        ("Bootleg", "Live Forever", "Bootleg", None),
        ("Bootleg", "Back, Baby (Live) - Mono", "Bootleg", "exact"),
    ],
)
def test_a_pool_file_names_its_version_outside_brackets(
    tmp_path: Path, pool_album: str, pool_title: str, play_album: str, tier: str | None
) -> None:
    path = jessica_pratt_pool(tmp_path, [("k.mp3", pool_album, pool_title)])
    assert (
        corpus.PoolIndex.load(path).tier("Jessica Pratt", play_album, "Back, Baby (Live)") == tier
    )


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
        "play_order_status", "carryover", "track_rows", "talk_rows", "group", "band", "subset",
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


@pytest.mark.parametrize(
    "out",
    [
        pytest.param(Path("plays.jsonl"), id="relative"),
        pytest.param(CHECKOUT, id="checkout"),
        # Under a directory that does not exist, so a regressed guard fails on open
        # rather than leaving a file in the working tree.
        pytest.param(CHECKOUT / "no-such-dir" / "plays.jsonl", id="in-checkout"),
        pytest.param(CHECKOUT / "tests" / ".." / "no-such-dir" / "p.jsonl", id="dotdot"),
    ],
)
def test_write_plays_refuses_a_path_in_the_checkout_and_creates_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pool_db: Path, out: Path
) -> None:
    export = write_export(tmp_path, [row(1, ts(20, 10))], "2026-08-09 05:00:43+00")
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    with pytest.raises(DataPathError):
        corpus.write_plays(
            out,
            ["2026/08/12/202608121600.mp3"],
            corpus.Flowsheet.load(export),
            corpus.PoolIndex.load(pool_db),
        )
    assert list(work.iterdir()) == []
    assert not (CHECKOUT / "no-such-dir").exists()


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


def hour_rows(
    start_id: int, hour: int, n: int, gap_s: float, day: int = 12, **kw: Any
) -> list[dict[str, object]]:
    return [row(start_id + i, ts(hour, second=60 + i * gap_s, day=day), **kw) for i in range(n)]


def test_hour_stats_counts_and_median_gap(tmp_path: Path, pool_db: Path) -> None:
    rows = hour_rows(1, 20, 4, 200.0, artist="Juana Molina", album="DOGA")
    rows += hour_rows(10, 20, 4, 200.0)  # out of pool, interleaved in time below
    rows[4:] = [row(10 + i, ts(20, second=160 + i * 200.0), show_id=2) for i in range(4)]
    rows.append(row(50, ts(20, 50), "talkset", show_id=3))  # a show's mic break, not its track
    export = write_export(tmp_path, rows, "2026-08-09 05:00:43+00")
    flowsheet = corpus.Flowsheet.load(export)
    stats = corpus.hour_stats(flowsheet, corpus.PoolIndex.load(pool_db))
    h = stats["2026/08/12/202608121600.mp3"]
    assert (h.era, h.track_rows, h.in_pool, h.median_gap_s, h.shows) == (
        "canonical",
        8,
        4,
        100.0,
        frozenset({"1", "2"}),
    )


def test_hour_stats_slots_are_each_shows_first_track_in_eastern_time(
    tmp_path: Path, pool_db: Path
) -> None:
    # Show 1 starts in the hour, Wednesday 16:01 EDT. Show 2 started before it: a talkset row
    # at 14:50 EDT, an Eastern hour before its first track at 15:50 EDT. Its slot is the
    # track's hour, so a slot taken from the first row of any type would read 14 instead.
    rows = hour_rows(1, 20, 8, 200.0)
    rows.append(row(20, ts(18, 50), "talkset", show_id=2))
    rows += [row(21, ts(19, 50), show_id=2), row(22, ts(20, 10), show_id=2)]
    rows.append(row(30, ts(20, 5, day=19), show_id=3))  # a week on, the same slot
    rows.append(row(40, ts(19, 30), "talkset", show_id=4))  # no track row, so no slot
    export = write_export(tmp_path, rows, "2026-08-09 05:00:43+00")
    sheet = corpus.Flowsheet.load(export)
    assert sheet.show_slot == {"1": (2026, 2, 16), "2": (2026, 2, 15), "3": (2026, 2, 16)}
    stats = corpus.hour_stats(sheet, corpus.PoolIndex.load(pool_db))
    assert stats["2026/08/12/202608121600.mp3"].slots == {(2026, 2, 15), (2026, 2, 16)}
    assert stats["2026/08/19/202608191600.mp3"].slots == {(2026, 2, 16)}


@pytest.mark.parametrize(
    ("gaps", "median", "dj_hour"),
    [
        # Batch-logged with one long break: the mean (~234 s) would pass, the median does not.
        ([10.0] * 3 + [1800.0] + [10.0] * 4, 10.0, False),
        # Logged live but a few tracks entered at once: the mean (62.5 s) would fail.
        ([100.0, 0.0, 100.0, 0.0, 100.0, 0.0, 100.0, 100.0], 100.0, True),
    ],
    ids=["mean-would-admit", "mean-would-reject"],
)
def test_hour_stats_gap_is_the_median_not_the_mean(
    tmp_path: Path, pool_db: Path, gaps: list[float], median: float, dj_hour: bool
) -> None:
    offsets = [60.0]
    for gap in gaps:
        offsets.append(offsets[-1] + gap)
    rows = [
        row(i + 1, ts(20, second=s), artist="Juana Molina", album="DOGA")
        for i, s in enumerate(offsets)
    ]
    export = write_export(tmp_path, rows, "2026-08-09 05:00:43+00")
    stats = corpus.hour_stats(corpus.Flowsheet.load(export), corpus.PoolIndex.load(pool_db))
    key = "2026/08/12/202608121600.mp3"
    assert (stats[key].track_rows, stats[key].median_gap_s) == (9, median)
    assert corpus.select_hours(stats, era="canonical", target=1) == ([key] if dj_hour else [])


def test_select_hours_ranks_by_in_pool_and_applies_rules(tmp_path: Path, pool_db: Path) -> None:
    pooled: dict[str, Any] = {"artist": "Juana Molina", "album": "DOGA"}
    rows = []
    rows += hour_rows(100, 20, 9, 200.0, **pooled)  # 9 in pool: picked first
    rows += hour_rows(200, 21, 8, 200.0, **pooled)  # 8 in pool
    rows += hour_rows(300, 22, 7, 200.0, **pooled)  # too few tracks
    rows += hour_rows(400, 23, 9, 30.0, **pooled)  # batch-logged: median gap 30 s
    rows += hour_rows(500, 20, 12, 200.0, day=10, **pooled)  # excluded day (Eastern 2026-08-10)
    rows += hour_rows(600, 14, 8, 200.0, day=13)  # nothing in pool
    export = write_export(tmp_path, rows, "2026-08-09 05:00:43+00")
    stats = corpus.hour_stats(corpus.Flowsheet.load(export), corpus.PoolIndex.load(pool_db))
    assert corpus.select_hours(stats, era="canonical", target=10) == [
        "2026/08/12/202608121600.mp3",
        "2026/08/12/202608121700.mp3",
    ]
    assert corpus.select_hours(stats, era="canonical", target=5) == ["2026/08/12/202608121600.mp3"]
    assert corpus.select_hours(stats, era="etl", target=5) == []


def test_select_hours_deprioritizes_reorder_flagged_shows(tmp_path: Path, pool_db: Path) -> None:
    pooled: dict[str, Any] = {"artist": "Juana Molina", "album": "DOGA"}
    flagged = hour_rows(100, 20, 8, 200.0, show_id=1, **pooled)
    flagged[0]["play_order"], flagged[1]["play_order"] = 101, 100  # a moved late-logged track
    clean = hour_rows(200, 21, 8, 200.0, show_id=2, **pooled)
    export = write_export(tmp_path, flagged + clean, "2026-08-09 05:00:43+00")
    stats = corpus.hour_stats(corpus.Flowsheet.load(export), corpus.PoolIndex.load(pool_db))
    assert stats["2026/08/12/202608121600.mp3"].reorder_flagged
    assert not stats["2026/08/12/202608121700.mp3"].reorder_flagged
    assert corpus.select_hours(stats, era="canonical", target=5) == ["2026/08/12/202608121700.mp3"]


def test_hour_stats_summarizes_show_less_tracks_as_unflagged(tmp_path: Path, pool_db: Path) -> None:
    # Show-less rows get no order label (they are unreliable, never reorder-flagged).
    rows = hour_rows(1, 20, 8, 200.0, show_id="")
    rows[0]["play_order"], rows[1]["play_order"] = 2, 1
    export = write_export(tmp_path, rows, "2026-08-09 05:00:43+00")
    stats = corpus.hour_stats(corpus.Flowsheet.load(export), corpus.PoolIndex.load(pool_db))
    h = stats["2026/08/12/202608121600.mp3"]
    assert (h.era, h.track_rows, h.reorder_flagged) == ("canonical", 8, False)


def stat(
    key: str, *, era: str = "canonical", tracks: int = 10, in_pool: int = 5, **kw: Any
) -> corpus.HourStats:
    """An hour's stats; unless ``shows`` is given, the hour is its own show, in no known slot."""
    return corpus.HourStats(
        key, era, tracks, in_pool, kw.get("gap", 200.0), kw.get("flagged", False),
        frozenset(kw.get("shows", {key})), frozenset(kw.get("slots", ())),
    )  # fmt: skip


def keys_in(day: int, hours: range, month: int = 9, year: int = 2026) -> list[str]:
    return [f"{year}/{month:02d}/{day:02d}/{year}{month:02d}{day:02d}{h:02d}00.mp3" for h in hours]


@pytest.mark.parametrize(
    ("key", "band"),
    [
        ("2026/08/12/202608120000.mp3", "overnight"),
        ("2026/08/12/202608120500.mp3", "overnight"),
        ("2026/08/12/202608120600.mp3", "daytime"),
        ("2026/08/12/202608121700.mp3", "daytime"),
        ("2026/08/12/202608121800.mp3", "evening"),
        ("2026/08/12/202608122300.mp3", "evening"),
        ("2026/12/02/202612020500.mp3", "overnight"),  # EST
        ("2026/12/02/202612020600.mp3", "daytime"),
        ("2026/12/02/202612021800.mp3", "evening"),
    ],
)
def test_band_uses_the_eastern_hour(key: str, band: str) -> None:
    assert corpus.band(key) == band


def plenty() -> dict[str, corpus.HourStats]:
    """Twelve high-share and three low-share canonical hours in every hour of the day."""
    stats = {
        k: stat(k, in_pool=5 + (k[-8:-6] == "05"))
        for d in range(1, 13)
        for k in keys_in(d, range(24))
    }
    stats |= {
        k: stat(k, in_pool=0, tracks=9 + d) for d in range(13, 16) for k in keys_in(d, range(24))
    }
    return stats


def test_select_corpus_fills_band_quotas_exactly() -> None:
    sel = corpus.select_corpus(plenty())
    counts: dict[tuple[str, str], int] = {}
    for group, band in sel.hours.values():
        counts[group, band] = counts.get((group, band), 0) + 1
    expected = {("canonical-high", b): q for b, q in corpus.HIGH_QUOTAS.items()}
    expected |= {("canonical-low", b): q for b, q in corpus.LOW_QUOTAS.items()}
    assert counts == expected
    assert sel.shortfalls["contrast/unfilled"] == 4
    assert {k: v for k, v in sel.shortfalls.items() if k.startswith("canonical")} == {}


def test_select_corpus_ranks_high_by_share_and_low_by_track_rows() -> None:
    sel = corpus.select_corpus(plenty())
    high = [k for k, (g, _) in sel.hours.items() if g == "canonical-high"]
    low = [k for k, (g, _) in sel.hours.items() if g == "canonical-low"]
    assert all(
        k.endswith("0500.mp3") for k in high if corpus.band(k) == "overnight"
    )  # the 6/10 share hours
    assert {k[8:10] for k in low} == {"15"}  # the day with the most track rows


def test_select_corpus_relaxes_a_thin_band_from_canonical_only() -> None:
    stats = {k: stat(k) for d in range(1, 4) for k in keys_in(d, range(6, 24))}
    stats |= {k: stat(k) for k in keys_in(4, range(0, 1))}  # one overnight hour
    stats |= {
        k: stat(k, era="etl", in_pool=10) for k in keys_in(5, range(0, 6), month=7)
    }  # etl overnight
    sel = corpus.select_corpus(stats)
    high = {k: b for k, (g, b) in sel.hours.items() if g == "canonical-high"}
    assert len(high) == 12 and list(high.values()).count("overnight") == 1
    assert sel.shortfalls["canonical-high/overnight"] == 2
    assert "canonical-high/unfilled" not in sel.shortfalls
    assert all(
        stats[k].era == "canonical" for k, (g, _) in sel.hours.items() if g.startswith("canonical")
    )


@pytest.mark.parametrize(
    ("group", "quotas", "unflagged", "flagged"),
    [
        # The best share in its band, but flagged.
        ("canonical-high", corpus.HIGH_QUOTAS, {"in_pool": 5}, {"in_pool": 10}),
        # The most track rows in its band, but flagged.
        ("canonical-low", corpus.LOW_QUOTAS, {"in_pool": 0}, {"in_pool": 0, "tracks": 20}),
    ],
    ids=["high", "low"],
)
def test_select_corpus_ranks_reorder_flagged_hours_last_in_their_band(
    group: str, quotas: dict[str, int], unflagged: dict[str, Any], flagged: dict[str, Any]
) -> None:
    # Every band has exactly its quota of unflagged hours, so nothing is relaxed.
    stats = {
        k: stat(k, **unflagged)
        for name, quota in quotas.items()
        for k in keys_in(1, corpus.BANDS[name])[:quota]
    }
    flagged_key = keys_in(2, corpus.BANDS["daytime"])[0]
    stats[flagged_key] = stat(flagged_key, flagged=True, **flagged)
    sel = corpus.select_corpus(stats)
    assert flagged_key not in sel.hours
    assert sum(g == group for g, _ in sel.hours.values()) == sum(quotas.values())


def test_select_corpus_ranks_high_share_hours_by_share_then_in_pool() -> None:
    # Ranking by in-pool count (Phase 1's rule) would put the long rotation-heavy hours first.
    long_show, tie_short, tie_long, short_show = keys_in(1, range(6, 10))
    stats = {
        long_show: stat(long_show, tracks=20, in_pool=7),  # share 0.35
        tie_short: stat(tie_short, tracks=10, in_pool=5),  # share 0.5, fewer in-pool plays
        tie_long: stat(tie_long, tracks=20, in_pool=10),  # share 0.5, more in-pool plays
        short_show: stat(short_show, tracks=8, in_pool=6),  # share 0.75
    }
    sel = corpus.select_corpus(stats)
    assert [k for k, (g, _) in sel.hours.items() if g == "canonical-high"] == [
        short_show,
        tie_long,
        tie_short,
        long_show,
    ]


@pytest.mark.parametrize(
    ("tracks", "group"),
    [
        (11, "canonical-low"),  # share 1/11, just below
        (10, "canonical-low"),  # share exactly LOW_SHARE_MAX
        (9, "canonical-high"),  # share 1/9, just above
    ],
)
def test_select_corpus_low_share_includes_the_boundary(tracks: int, group: str) -> None:
    key = keys_in(1, range(10, 11))[0]
    sel = corpus.select_corpus({key: stat(key, tracks=tracks, in_pool=1)})
    assert sel.hours[key] == (group, "daytime")


@pytest.mark.parametrize(
    ("ineligible", "chosen"),
    [
        ({"tracks": 7}, False),
        ({"gap": 89.0}, False),  # batch-logged
        ({"tracks": 8, "gap": 90.0}, True),  # both minimums met exactly
    ],
    ids=["too-few-tracks", "batch-logged", "at-the-minimums"],
)
def test_select_corpus_contrast_hours_must_be_dj_hours(
    ineligible: dict[str, Any], chosen: bool
) -> None:
    key = keys_in(5, range(20, 21), month=6, year=2023)[0]
    sel = corpus.select_corpus({key: stat(key, era="etl", in_pool=6, **ineligible)})
    assert (key in sel.hours) is chosen
    assert ("contrast/2023" in sel.shortfalls) is not chosen


# Excluded days are Eastern dates. Each evening key's UTC date is the next day's.
EXCLUDED_DAY_EDGES = [
    ("2026/08/08/202608082000.mp3", True),  # 2026-08-09 00:00 UTC
    ("2026/08/08/202608082300.mp3", True),  # 2026-08-09 03:00 UTC
    ("2026/08/09/202608090000.mp3", False),
    ("2026/08/11/202608112000.mp3", False),  # 2026-08-12 00:00 UTC
    ("2026/08/11/202608112300.mp3", False),  # 2026-08-12 03:00 UTC
    ("2026/08/12/202608120000.mp3", True),
]


def era_of(key: str) -> str:
    return "canonical" if corpus.hour_start(key) >= corpus.ETL_STOP_FLOOR else "etl"


@pytest.mark.parametrize(("key", "eligible"), EXCLUDED_DAY_EDGES)
def test_excluded_days_are_eastern_dates_for_dj_hours(key: str, eligible: bool) -> None:
    era = era_of(key)
    selected = corpus.select_hours({key: stat(key, era=era)}, era=era, target=1)
    assert selected == ([key] if eligible else [])


def test_select_corpus_skips_ineligible_hours() -> None:
    stats = {
        k: stat(k, tracks=7) if i == 0 else stat(k, gap=30.0) if i == 1 else stat(k)
        for i, k in enumerate(keys_in(1, range(6, 9)))
    }
    stats |= {k: stat(k) for k in keys_in(10, range(6, 8), month=8)}  # 2026-08-10: gap-import day
    sel = corpus.select_corpus(stats)
    assert list(sel.hours) == ["2026/09/01/202609010800.mp3"]


def test_select_corpus_contrast_one_per_year_then_best_remaining() -> None:
    stats = {k: stat(k, era="etl", in_pool=3) for k in keys_in(5, range(10, 14), year=2022)}
    stats |= {
        k: stat(k, era="etl", in_pool=8 - i)
        for i, k in enumerate(keys_in(5, range(10, 13), year=2024))
    }
    stats |= {
        k: stat(k, era="etl", in_pool=10) for k in keys_in(5, range(10, 12), year=2025)
    }  # out of range
    sel = corpus.select_corpus(stats)
    contrast = [k for k, (g, _) in sel.hours.items() if g == "contrast"]
    assert [k[:4] for k in contrast] == ["2022", "2024", "2024", "2024"]
    assert contrast[1:] == keys_in(5, range(10, 13), year=2024)
    assert sel.shortfalls["contrast/2023"] == 1


def contrast(sel: corpus.Selection) -> list[str]:
    return [k for k, (g, _) in sel.hours.items() if g == "contrast"]


def contrast_shortfalls(sel: corpus.Selection) -> dict[str, int]:
    return {k: v for k, v in sel.shortfalls.items() if k.startswith("contrast")}


def test_select_corpus_contrast_takes_at_most_one_hour_per_show() -> None:
    # Two high-share hours from one 2023 show: only the better one is chosen.
    best, same_show, other_show = keys_in(5, range(10, 13), year=2023)
    stats = {
        best: stat(best, era="etl", in_pool=9, shows={"7"}),
        same_show: stat(same_show, era="etl", in_pool=8, shows={"7"}),
        other_show: stat(other_show, era="etl", in_pool=2, shows={"8"}),
    }
    sel = corpus.select_corpus(stats)
    assert contrast(sel) == [best, other_show]
    assert contrast_shortfalls(sel) == {
        "contrast/2022": 1,
        "contrast/2024": 1,
        "contrast/unfilled": 2,
    }


def test_select_corpus_records_a_contrast_shortfall_when_shows_run_out() -> None:
    # One show across all three years: one hour, and the cap is never relaxed.
    first = keys_in(5, range(10, 11), year=2022)[0]
    stats = {first: stat(first, era="etl", in_pool=9, shows={"7"})}
    stats |= {
        k: stat(k, era="etl", in_pool=8, shows={"7"}) for k in keys_in(5, range(10, 12), year=2023)
    }
    stats |= {
        k: stat(k, era="etl", in_pool=8, shows={"7"}) for k in keys_in(5, range(10, 12), year=2024)
    }
    sel = corpus.select_corpus(stats)
    assert contrast(sel) == [first]
    assert contrast_shortfalls(sel) == {
        "contrast/2023": 1,
        "contrast/2024": 1,
        "contrast/unfilled": 3,
    }


@pytest.mark.parametrize(
    ("ranked", "chosen"),
    [
        # The two-show hour ranks first: it blocks an hour from either show.
        ([{"7", "8"}, {"7"}, {"8"}, {"9"}], [0, 3]),
        # An hour of show 7 ranks first: the two-show hour holds show 7 too.
        ([{"7"}, {"7", "8"}, {"8"}, {"9"}], [0, 2, 3]),
    ],
    ids=["spanning-first", "spanning-second"],
)
def test_select_corpus_contrast_hour_spanning_two_shows_belongs_to_both(
    ranked: list[set[str]], chosen: list[int]
) -> None:
    keys = keys_in(5, range(10, 14), year=2023)
    stats = {
        k: stat(k, era="etl", in_pool=9 - i, shows=s)
        for i, (k, s) in enumerate(zip(keys, ranked, strict=True))
    }
    sel = corpus.select_corpus(stats)
    assert contrast(sel) == [keys[i] for i in chosen]
    assert sel.shortfalls["contrast/unfilled"] == corpus.CONTRAST_HOURS - len(chosen)


def test_select_corpus_contrast_takes_one_hour_per_recurring_slot_per_year() -> None:
    # Two broadcasts of one weekly Friday 20:00 show, same year: only the better one is chosen.
    weekly = (2023, 4, 20)
    week_one, week_two, other_slot = keys_in(5, range(10, 13), year=2023)
    stats = {
        week_one: stat(week_one, era="etl", in_pool=9, slots={weekly}),
        week_two: stat(week_two, era="etl", in_pool=8, slots={weekly}),
        other_slot: stat(other_slot, era="etl", in_pool=2, slots={(2023, 1, 9)}),
    }
    sel = corpus.select_corpus(stats)
    assert contrast(sel) == [week_one, other_slot]
    assert contrast_shortfalls(sel) == {
        "contrast/2022": 1,
        "contrast/2024": 1,
        "contrast/unfilled": 2,
    }


def test_select_corpus_contrast_allows_the_same_slot_in_different_years() -> None:
    # A schedule changes each semester, so a Friday 20:00 slot in 2022 and 2023 is likely two DJs.
    in_2022, in_2023 = (keys_in(5, range(10, 11), year=y)[0] for y in (2022, 2023))
    stats = {
        in_2022: stat(in_2022, era="etl", slots={(2022, 4, 20)}),
        in_2023: stat(in_2023, era="etl", slots={(2023, 4, 20)}),
    }
    sel = corpus.select_corpus(stats)
    assert contrast(sel) == [in_2022, in_2023]


def test_select_corpus_records_a_contrast_shortfall_when_slots_run_out() -> None:
    # Distinct broadcasts, one slot per year: the cap is never relaxed to fill the remainder.
    stats = {
        k: stat(k, era="etl", in_pool=9 - i, slots={(2023, 4, 20)})
        for i, k in enumerate(keys_in(5, range(10, 14), year=2023))
    }
    sel = corpus.select_corpus(stats)
    assert contrast(sel) == [min(stats)]
    assert contrast_shortfalls(sel) == {
        "contrast/2022": 1,
        "contrast/2024": 1,
        "contrast/unfilled": 3,
    }


def test_select_corpus_contrast_hour_spanning_two_slots_belongs_to_both() -> None:
    # The spanning hour ranks first; a lower-ranked hour in each of its slots is blocked, so
    # recording only one of the two slots would let the other hour through.
    first, in_first_slot, in_second_slot, other = keys_in(5, range(10, 14), year=2023)
    stats = {
        first: stat(first, era="etl", in_pool=9, slots={(2023, 4, 20), (2023, 4, 21)}),
        in_first_slot: stat(in_first_slot, era="etl", in_pool=8, slots={(2023, 4, 20)}),
        in_second_slot: stat(in_second_slot, era="etl", in_pool=7, slots={(2023, 4, 21)}),
        other: stat(other, era="etl", in_pool=6, slots={(2023, 1, 9)}),
    }
    assert contrast(corpus.select_corpus(stats)) == [first, other]


def test_select_corpus_is_twenty_hours_with_no_shortfall() -> None:
    stats = plenty()
    stats |= {
        k: stat(k, era="etl", in_pool=3, shows={f"{year}"})
        for year in corpus.CONTRAST_YEARS
        for k in keys_in(5, range(10, 12), year=year)
    }
    stats |= {
        k: stat(k, era="etl", in_pool=2, shows={"2023b"})
        for k in keys_in(6, range(10, 11), year=2023)
    }
    sel = corpus.select_corpus(stats)
    assert len(sel.hours) == 20 and sel.shortfalls == {}
    assert len(contrast(sel)) == 4
    assert len(set().union(*(stats[k].shows for k in contrast(sel)))) == 4


def hour_in(day: int, hour: int, **kw: Any) -> corpus.HourStats:
    return stat(keys_in(day, range(hour, hour + 1))[0], **kw)


def subset_of(stats: dict[str, corpus.HourStats]) -> tuple[list[str], dict[str, int]]:
    sel = corpus.select_corpus(stats)
    subset = corpus.choose_subset(sel, stats)
    return subset, {k: v for k, v in sel.shortfalls.items() if k.startswith("subset/")}


def test_subset_is_the_top_high_hour_of_two_bands_the_top_low_and_the_top_contrast() -> None:
    stats = plenty()
    etl = {
        k: stat(k, era="etl", in_pool=n, shows={k})
        for k, n in zip(keys_in(5, range(10, 12), year=2023), (3, 7), strict=True)
    }
    stats |= etl
    subset, shortfalls = subset_of(stats)
    overnight_top = keys_in(1, range(5, 6))[0]  # share 0.6, the best of the corpus
    daytime_top = keys_in(1, range(6, 7))[0]  # the best of the other bands, ties by key
    low_top = keys_in(15, range(0, 1))[0]  # most track rows
    contrast_top = keys_in(5, range(11, 12), year=2023)[0]  # share 0.7
    assert subset == [overnight_top, daytime_top, low_top, contrast_top]
    assert shortfalls == {}


def test_subset_high_hours_come_from_two_different_bands() -> None:
    day = keys_in(1, range(8, 11))
    evening = keys_in(1, range(19, 21))
    stats = {k: stat(k, in_pool=9) for k in day} | {k: stat(k, in_pool=5) for k in evening}
    subset, _ = subset_of(stats)
    assert subset[:2] == [day[0], evening[0]]


def test_subset_ranks_a_reorder_flagged_hour_after_an_unflagged_one() -> None:
    flagged, plain, other = hour_in(1, 8, in_pool=9, flagged=True), hour_in(1, 9), hour_in(1, 20)
    subset, _ = subset_of({h.key: h for h in (flagged, plain, other)})
    assert subset[:2] == [plain.key, other.key]


@pytest.mark.parametrize(
    ("stats", "subset_size", "shortfalls"),
    [
        pytest.param({}, 0, {"subset/canonical-high": 2, "subset/canonical-low": 1,
                             "subset/contrast": 1}, id="nothing"),
        pytest.param({k: stat(k) for k in keys_in(1, range(8, 12))}, 1,
                     {"subset/canonical-high": 1, "subset/canonical-low": 1,
                      "subset/contrast": 1}, id="one-band-of-high"),
        pytest.param(plenty(), 3, {"subset/contrast": 1}, id="no-contrast"),
    ],
)  # fmt: skip
def test_a_missing_subset_slot_is_a_shortfall_never_filled_from_another_group(
    stats: dict[str, corpus.HourStats], subset_size: int, shortfalls: dict[str, int]
) -> None:
    subset, got = subset_of(stats)
    assert len(subset) == subset_size and got == shortfalls


def select_export(tmp_path: Path) -> Path:
    """Three DJ hours (two daytime, one evening), each a show of eight in-pool tracks."""
    rows = [
        row(100 * show + i, ts(hour, 5 * i), show_id=show, artist="Juana Molina", album="DOGA",
            title="la paradoja")
        for show, hour in ((1, 18), (2, 19), (3, 23))
        for i in range(8)
    ]  # fmt: skip
    return write_export(tmp_path, rows, "2026-08-09 05:00:43+00")


HOURS = [
    "2026/08/12/202608121400.mp3",
    "2026/08/12/202608121500.mp3",
    "2026/08/12/202608121900.mp3",
]


# Each select_export hour: eight track rows, all in the pool.
BASIS = {"in_pool": 8, "track_rows": 8}


def run_select(tmp_path: Path, pool_db: Path, out_dir: Path) -> int:
    export = tmp_path / "export"
    if not export.exists():
        select_export(tmp_path)
    return corpus.main(
        ["select", "--export", str(export), "--pool-db", str(pool_db), "--out-dir", str(out_dir)]
    )


def test_select_writes_the_frozen_hours_and_the_subset(tmp_path: Path, pool_db: Path) -> None:
    out = tmp_path / "frozen"
    assert run_select(tmp_path, pool_db, out) == 0
    assert (out / "hours.txt").read_text() == "".join(f"{k}\n" for k in HOURS)
    assert (out / "subset.txt").read_text() == f"{HOURS[0]}\n{HOURS[2]}\n"
    record = json.loads((out / "selection.json").read_text())
    assert record["export"] == "export" and record["pool_db"] == str(pool_db.resolve())
    assert record["hours"] == {
        HOURS[0]: {"group": "canonical-high", "band": "daytime", "subset": True} | BASIS,
        HOURS[1]: {"group": "canonical-high", "band": "daytime", "subset": False} | BASIS,
        HOURS[2]: {"group": "canonical-high", "band": "evening", "subset": True} | BASIS,
    }
    assert record["shortfalls"]["subset/canonical-low"] == 1
    assert record["shortfalls"]["canonical-high/overnight"] == 3


def test_the_leg_runner_reads_what_select_writes(tmp_path: Path, pool_db: Path) -> None:
    # selection.json is a contract between this station module and the neutral run.py,
    # which parses it on its own; this is the one test that holds both sides to it.
    out = tmp_path / "frozen"
    assert run_select(tmp_path, pool_db, out) == 0
    assert read_hours(out / "selection.json") == {
        "all": (out / "hours.txt").read_text().split(),
        "subset": (out / "subset.txt").read_text().split(),
    }


def test_select_refuses_to_overwrite_and_leaves_every_file_untouched(
    tmp_path: Path, pool_db: Path
) -> None:
    out = tmp_path / "frozen"
    run_select(tmp_path, pool_db, out)
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    with pytest.raises(FileExistsError):
        run_select(tmp_path, pool_db, out)
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before


def test_select_with_one_existing_file_creates_none_of_the_others(
    tmp_path: Path, pool_db: Path
) -> None:
    out = tmp_path / "frozen"
    out.mkdir()
    (out / "subset.txt").write_text("kept\n")
    with pytest.raises(FileExistsError):
        run_select(tmp_path, pool_db, out)
    assert [p.name for p in out.iterdir()] == ["subset.txt"]
    assert (out / "subset.txt").read_text() == "kept\n"


@pytest.mark.parametrize("where", ["checkout", "inside", "relative"])
def test_select_refuses_an_out_dir_that_is_not_outside_the_checkout(
    tmp_path: Path, pool_db: Path, where: str
) -> None:
    out = {
        "checkout": paths.CHECKOUT,
        "inside": paths.CHECKOUT / "frozen-selection",
        "relative": Path("frozen-selection"),
    }[where]
    with pytest.raises(paths.DataPathError):
        run_select(tmp_path, pool_db, out)
    assert not (paths.CHECKOUT / "frozen-selection").exists()
    assert not (paths.CHECKOUT / "hours.txt").exists()


def test_select_writes_into_the_data_dir_by_default(
    tmp_path: Path, pool_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setenv("STREAM_SLEUTH_DATA_DIR", str(data))
    export = select_export(tmp_path)
    corpus.main(["select", "--export", str(export), "--pool-db", str(pool_db)])
    assert sorted(p.name for p in data.iterdir()) == ["hours.txt", "selection.json", "subset.txt"]


def plays_main(tmp_path: Path, pool_db: Path, *selector: str) -> list[dict[str, object]]:
    out = tmp_path / "plays.jsonl"
    corpus.main(
        [
            "--export",
            str(tmp_path / "export"),
            "--pool-db",
            str(pool_db),
            *selector,
            "--out",
            str(out),
        ]
    )
    return read_plays(out)


def test_plays_from_a_selection_carry_each_hours_group_band_and_subset(
    tmp_path: Path, pool_db: Path, caplog: pytest.LogCaptureFixture
) -> None:
    out = tmp_path / "frozen"
    run_select(tmp_path, pool_db, out)
    plays = plays_main(tmp_path, pool_db, "--selection", str(out / "selection.json"))
    assert any(p["carryover"] for p in plays)
    # Unchanged inputs: the recomputed counts (carryover plays excluded) match the frozen ones.
    assert "ranking basis" not in caplog.text
    stamped = {(p["hour_key"], p["group"], p["band"], p["subset"]) for p in plays}
    assert stamped == {
        (HOURS[0], "canonical-high", "daytime", True),
        (HOURS[1], "canonical-high", "daytime", False),
        (HOURS[2], "canonical-high", "evening", True),
    }


def test_plays_from_a_bare_hours_file_have_no_group_and_the_hours_own_band(
    tmp_path: Path, pool_db: Path
) -> None:
    select_export(tmp_path)
    hours = tmp_path / "hours.txt"
    hours.write_text(f"{HOURS[2]}\n")
    plays = plays_main(tmp_path, pool_db, "--hours", str(hours))
    assert {(p["group"], p["band"], p["subset"]) for p in plays} == {(None, "evening", False)}


@pytest.mark.parametrize(
    ("given", "message"),
    [([], "is required"), (["--hours", "h.txt", "--selection", "s.json"], "not allowed with")],
    ids=["neither", "both"],
)
def test_plays_need_exactly_one_of_hours_and_selection(
    tmp_path: Path,
    pool_db: Path,
    given: list[str],
    message: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "p"
    with pytest.raises(SystemExit) as exc_info:
        corpus.main(["--export", "e", "--pool-db", str(pool_db), "--out", str(out), *given])
    # A usage error (exit 2) from the argument parser, not read_selection's "cannot read" exit.
    assert exc_info.value.code == 2
    assert message in capsys.readouterr().err
    assert not out.exists()


def test_plays_subset_only_needs_a_selection(tmp_path: Path, pool_db: Path) -> None:
    with pytest.raises(SystemExit):
        corpus.main(["--export", "e", "--pool-db", str(pool_db), "--hours", "h.txt",
                     "--subset-only", "--out", str(tmp_path / "p")])  # fmt: skip


def test_plays_subset_only_writes_just_the_subset_hours_stamped_from_the_selection(
    tmp_path: Path, pool_db: Path
) -> None:
    out = tmp_path / "frozen"
    run_select(tmp_path, pool_db, out)
    plays = plays_main(
        tmp_path, pool_db, "--selection", str(out / "selection.json"), "--subset-only"
    )
    assert {(p["hour_key"], p["group"], p["subset"]) for p in plays} == {
        (HOURS[0], "canonical-high", True),
        (HOURS[2], "canonical-high", True),
    }


def test_plays_from_a_bare_hours_file_stamp_a_carryover_play_too(
    tmp_path: Path, pool_db: Path
) -> None:
    select_export(tmp_path)
    hours = tmp_path / "hours.txt"
    hours.write_text(f"{HOURS[1]}\n")  # 15:00 Eastern: the 14:00 hour's last track leads it
    plays = plays_main(tmp_path, pool_db, "--hours", str(hours))
    assert any(p["carryover"] for p in plays)
    assert {(p["group"], p["band"], p["subset"]) for p in plays} == {(None, "daytime", False)}


CROSSING = ["2026/08/12/202608121700.mp3", "2026/08/12/202608121800.mp3"]


def crossing_export(tmp_path: Path) -> Path:
    """A daytime-hour track at 17:55 Eastern, then one at 18:10, in the evening band."""
    rows = [row(1, ts(21, 55)), row(2, ts(22, 10))]
    return write_export(tmp_path, rows, "2026-08-09 05:00:43+00")


def write_selection_json(
    path: Path, pool_db: Path, hours: object, header: dict[str, str] | None = None
) -> Path:
    record = {"export": "export", "pool_db": str(pool_db), "hours": hours} | (header or {})
    path.write_text(json.dumps(record))
    return path


def test_a_carryover_across_a_band_boundary_takes_the_band_of_the_hour_it_is_written_into(
    tmp_path: Path, pool_db: Path
) -> None:
    crossing_export(tmp_path)
    hours = tmp_path / "hours.txt"
    hours.write_text(f"{CROSSING[1]}\n")
    plays = plays_main(tmp_path, pool_db, "--hours", str(hours))
    assert [(p["play_id"], p["carryover"], p["band"]) for p in plays] == [
        (1, True, "evening"),
        (2, False, "evening"),
    ]


def test_a_carryover_takes_the_label_of_the_hour_it_is_written_into(
    tmp_path: Path, pool_db: Path
) -> None:
    crossing_export(tmp_path)
    label = {"group": "contrast", "band": "evening", "subset": True}
    sel = write_selection_json(tmp_path / "selection.json", pool_db, {CROSSING[1]: label})
    plays = plays_main(tmp_path, pool_db, "--selection", str(sel))
    assert [(p["carryover"], p["group"], p["band"], p["subset"]) for p in plays] == [
        (True, "contrast", "evening", True),
        (False, "contrast", "evening", True),
    ]


GOOD = {"group": "contrast", "band": "daytime", "subset": False}


@pytest.mark.parametrize(
    ("content", "problem"),
    [
        pytest.param([], "not an object", id="shared-structural-check"),  # see test_selection.py
        pytest.param({"hours": {HOURS[0]: GOOD | {"group": "high"}}}, "group", id="bad-group"),
        pytest.param({"hours": {HOURS[0]: GOOD | {"group": None}}}, "group", id="null-group"),
        pytest.param({"hours": {HOURS[0]: GOOD | {"band": "night"}}}, "band", id="bad-band"),
        pytest.param(
            {"hours": {HOURS[0]: {"group": "contrast", "subset": False}}}, "band", id="missing-band"
        ),
        pytest.param({"hours": {}}, "`hours` is empty", id="empty-hours"),
        pytest.param(
            {"hours": {HOURS[0]: GOOD | {"band": "evening"}}},
            f"{HOURS[0]}: band 'evening' is not the hour's own",
            id="band-not-the-hours-own",
        ),
    ],
)
def test_a_bad_selection_file_exits_with_one_clear_line_and_writes_nothing(
    tmp_path: Path, pool_db: Path, content: object, problem: str
) -> None:
    select_export(tmp_path)
    sel = tmp_path / "selection.json"
    sel.write_text(content if isinstance(content, str) else json.dumps(content))
    with pytest.raises(SystemExit) as exc:
        plays_main(tmp_path, pool_db, "--selection", str(sel))
    message = str(exc.value.code)
    assert str(sel) in message and problem in message and "\n" not in message
    assert not (tmp_path / "plays.jsonl").exists()


@pytest.mark.parametrize(
    ("header", "warned"),
    [
        pytest.param({}, False, id="same-inputs"),
        pytest.param({"export": "2020-01-01"}, True, id="other-export"),
        pytest.param({"pool_db": "/elsewhere/pool.db"}, True, id="other-pool"),
    ],
)
def test_plays_warn_when_the_selection_came_from_other_inputs(
    tmp_path: Path, pool_db: Path, caplog: pytest.LogCaptureFixture,
    header: dict[str, str], warned: bool,
) -> None:  # fmt: skip
    select_export(tmp_path)
    sel = write_selection_json(tmp_path / "selection.json", pool_db, {HOURS[0]: GOOD}, header)
    caplog.set_level(logging.WARNING, logger=corpus.log.name)
    assert plays_main(tmp_path, pool_db, "--selection", str(sel))
    assert ("selection.json was made from" in caplog.text) is warned


def test_select_logs_the_subset_and_corpus_shortfalls(
    tmp_path: Path, pool_db: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger=corpus.log.name)
    run_select(tmp_path, pool_db, tmp_path / "frozen")
    assert "selection shortfall subset/canonical-low: 1" in caplog.text
    assert "selection shortfall canonical-high/overnight: 3" in caplog.text


def test_a_relative_pool_db_given_to_select_is_recorded_absolute(
    tmp_path: Path, pool_db: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    select_export(tmp_path)
    monkeypatch.chdir(pool_db.parent)
    out = tmp_path / "frozen"
    corpus.main(["select", "--export", str(tmp_path / "export"), "--pool-db", pool_db.name,
                 "--out-dir", str(out)])  # fmt: skip
    assert json.loads((out / "selection.json").read_text())["pool_db"] == str(pool_db.resolve())
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    caplog.set_level(logging.WARNING, logger=corpus.log.name)
    assert plays_main(tmp_path, pool_db, "--selection", str(out / "selection.json"))
    assert "selection.json was made from" not in caplog.text


def test_plays_warn_once_naming_each_hour_the_pool_moved(
    tmp_path: Path, pool_db: Path, caplog: pytest.LogCaptureFixture
) -> None:
    out = tmp_path / "frozen"
    run_select(tmp_path, pool_db, out)
    db = sqlite3.connect(pool_db)
    db.execute("DELETE FROM files WHERE artist = 'Juana Molina'")
    db.commit()
    db.close()
    caplog.set_level(logging.WARNING, logger=corpus.log.name)
    plays = plays_main(tmp_path, pool_db, "--selection", str(out / "selection.json"))
    assert {p["hour_key"] for p in plays} == set(HOURS)  # the selection is unchanged
    moved = [r.getMessage() for r in caplog.records if "ranking basis" in r.getMessage()]
    assert len(moved) == 1
    assert all(f"{k} (track_rows 8 -> 8, in_pool 8 -> 0)" in moved[0] for k in HOURS)


@pytest.mark.parametrize(
    ("basis", "warning"),
    [
        pytest.param(BASIS, None, id="unchanged"),
        pytest.param(
            {"in_pool": 7, "track_rows": 8}, "track_rows 8 -> 8, in_pool 7 -> 8", id="in-pool"
        ),
        pytest.param(
            {"in_pool": 8, "track_rows": 9}, "track_rows 9 -> 8, in_pool 8 -> 8", id="tracks"
        ),
        pytest.param({}, "cannot be checked", id="no-counts"),
    ],
)
def test_plays_compare_the_recorded_basis_with_the_recomputed_one(
    tmp_path: Path, pool_db: Path, caplog: pytest.LogCaptureFixture,
    basis: dict[str, int], warning: str | None,
) -> None:  # fmt: skip
    select_export(tmp_path)
    sel = write_selection_json(
        tmp_path / "selection.json",
        pool_db,
        {HOURS[0]: GOOD | basis, HOURS[2]: GOOD | {"band": "evening"} | basis},
    )
    caplog.set_level(logging.WARNING, logger=corpus.log.name)
    assert plays_main(tmp_path, pool_db, "--selection", str(sel))
    warned = [r.getMessage() for r in caplog.records if "ranking basis" in r.getMessage()]
    assert len(warned) == (warning is not None)
    if warning:
        assert warning in warned[0]


@pytest.mark.parametrize("where", ["inside", "relative"])
@pytest.mark.parametrize("cli", ["select", "plays"])
def test_each_cli_checks_its_output_path_before_it_opens_the_export(
    tmp_path: Path, pool_db: Path, cli: str, where: str
) -> None:
    out = {"inside": paths.CHECKOUT / "unwritten", "relative": Path("unwritten")}[where]
    common = ["--export", str(tmp_path / "no-such-export"), "--pool-db", str(tmp_path / "no.db")]
    argv = (
        ["select", *common, "--out-dir", str(out)]
        if cli == "select"
        else [*common, "--hours", str(tmp_path / "no-hours.txt"), "--out", str(out)]
    )
    with pytest.raises(paths.DataPathError):
        corpus.main(argv)
    assert not out.exists() and not (paths.CHECKOUT / "unwritten").exists()
