"""``python -m evaluation.score``: the score file and the near-miss queue.

Every plays file, store, and ``pool.db`` is synthetic and lives under a ``tmp_path`` data
directory. ``hour_addresses`` is replaced, so no test needs ffmpeg or an archive.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, cast

import pytest

from evaluation import score
from evaluation.clips import ClipAddress, grid
from evaluation.olaf_snapshot import RESULTS
from evaluation.pool import open_pool_db
from evaluation.results import Emission, ResultStore, recognizer_identity
from evaluation.score import attribute, plays_from
from stream_sleuth.recognizers.olaf import recognizer_identity as olaf_identity
from tests.stores import olaf_record, shazam_record
from tests.unit.test_score import (
    COCREDIT,
    HERMANOS,
    HOUR,
    MOLINA,
    OTHER,
    PRATT,
    RECORDS,
    SHAZAM,
    STAGE,
    UNLOGGED,
    _found,
    _hit,
    _hour,
    _local,
)

SNAPSHOT = "rotation"
SHAZAM12 = recognizer_identity(12)
OLAF = olaf_identity(SNAPSHOT)


@pytest.fixture
def data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A data directory with plays.jsonl, selection.json, and no archive: grids are synthetic."""
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setenv("STREAM_SLEUTH_DATA_DIR", str(data))
    monkeypatch.setattr(
        score,
        "hour_addresses",
        lambda keys, archive, length_s, profile: [
            a for k in keys for a in grid(k, length_s, profile)
        ],
    )
    labels = {HOUR: {"subset": True}, OTHER: {"subset": False}}
    (data / "selection.json").write_text(json.dumps({"hours": labels}))
    write_plays(data, RECORDS)
    return data


def write_plays(data: Path, records: list[dict[str, Any]]) -> Path:
    path = data / "plays.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    return path


def add_snapshot(data: Path, artist: str = COCREDIT, album_artist: str = "Juana Molina") -> None:
    """A snapshot whose one indexed file is the co-credited reference ``STAGE``."""
    db = open_pool_db(data / "olaf" / SNAPSHOT / "pool.db")
    db.execute(
        "INSERT INTO files (key, stage_id, prefix, format, size, artist, album_artist, album,"
        " title, status) VALUES ('k', ?, 'rotation/', 'mp3', 1, ?, ?, ?, ?, 'indexed')",
        (STAGE, artist, album_artist, MOLINA[2], MOLINA[1]),
    )
    db.commit()
    db.close()


def add_olaf(data: Path, address: ClipAddress, artist: str) -> None:
    olaf_record(
        ResultStore(data / "olaf" / SNAPSHOT / RESULTS),
        str(address),
        "matched",
        (artist, MOLINA[1], MOLINA[2]),
        identity=OLAF,
        ref_key=STAGE,
        ref_start_s=75.0,
    )


def add_shazam(data: Path, address: ClipAddress, track: tuple[str, str, str], **extra: Any) -> None:
    (data / "shazam").mkdir(exist_ok=True)
    shazam_record(
        ResultStore(data / "shazam" / "results.jsonl"), str(address), "matched", track, **extra
    )


def run_cli(data: Path, *args: str) -> int:
    """Score the 12 s leg, Shazam alone unless a snapshot is named."""
    only = [] if "--snapshot" in args else ["--only", "shazam"]
    return score.main(["--plays", str(data / "plays.jsonl"), "--legs", "12s", *only, *args])


def read_score(data: Path) -> dict[str, Any]:
    return json.loads((data / "score" / "score.json").read_text(encoding="utf-8"))


def test_an_olaf_match_on_a_co_credited_file_is_correct_in_the_score_file(data: Path) -> None:
    """The CLI hands the snapshot's own pool.db names to the scorer; without them this is wrong."""
    add_snapshot(data)
    add_olaf(data, ClipAddress(HOUR, 195, 12), COCREDIT)

    assert run_cli(data, "--snapshot", SNAPSHOT) == 0

    [emission] = read_score(data)["legs"]["12s/olaf"]["emissions"]
    assert (emission["play_id"], emission["neighbor"]) == (1, False)


def test_without_the_snapshots_pool_db_the_co_credit_would_be_wrong(data: Path) -> None:
    add_snapshot(data, artist=COCREDIT, album_artist="")  # no tag names the logged artist
    add_olaf(data, ClipAddress(HOUR, 195, 12), COCREDIT)

    run_cli(data, "--snapshot", SNAPSHOT)

    [emission] = read_score(data)["legs"]["12s/olaf"]["emissions"]
    assert emission["play_id"] is None


def test_the_score_file_has_the_shape_report_py_reads(data: Path) -> None:
    add_snapshot(data)
    add_olaf(data, ClipAddress(HOUR, 195, 12), COCREDIT)
    for a in grid(HOUR, 12):
        if a.offset_s != 195:
            olaf_record(
                ResultStore(data / "olaf" / SNAPSHOT / RESULTS), str(a), "no_match", identity=OLAF
            )

    run_cli(data, "--snapshot", SNAPSHOT)

    result = read_score(data)
    assert sorted(result) == ["legs", "version"]
    assert result["version"] == 1
    assert sorted(result["legs"]) == ["12s/olaf", "12s/shazam"]
    leg = result["legs"]["12s/olaf"]
    assert sorted(leg) == [
        "adjudicated_distinct_precision",
        "adjudicated_precision",
        "carryover_plays",
        "coverage",
        "distinct_precision",
        "emissions",
        "in_pool_recall",
        "leg",
        "length_s",
        "pads",
        "plays",
        "precision",
        "profile",
        "recall",
        "recognizer",
        "unjoinable_plays",
    ]
    assert (leg["leg"], leg["recognizer"], leg["length_s"], leg["profile"]) == (
        "12s",
        OLAF,
        12,
        "128k",
    )
    assert leg["coverage"] == {
        "grid_addresses": 480,
        "scored_addresses": 240,
        "uncovered_addresses": {"untried": 240},
        "hours_without_records": [OTHER],
        "uncovered_plays": {},
    }
    assert (leg["precision"], leg["distinct_precision"], leg["recall"]) == (1.0, 1.0, 0.2)
    assert leg["adjudicated_precision"] is None
    assert leg["pads"] == {
        "canonical": {"recommended_s": 180.0, "samples": 1, "p95_s": None, "window_pads_s": [180.0]}
    }
    assert leg["emissions"] == [
        {
            "address": f"{HOUR}#195+12@128k",
            "hour_key": HOUR,
            "source": "local",
            "artist": COCREDIT,
            "song": MOLINA[1],
            "album": MOLINA[2],
            "play_id": 1,
            "carryover": False,
            "neighbor": False,
        }
    ]
    assert leg["carryover_plays"] == []
    assert leg["plays"][0] == {
        "hour_key": HOUR,
        "play_id": 1,
        "carryover": False,
        "era": "canonical",
        "in_pool": True,
        "pool_match_tier": "exact",
        "pool_format": "mp3",
        "rotation": True,
        "group": "canonical-high",
        "band": "daytime",
        "subset": False,
        "reorder_flag": False,
        "play_order_status": "single_writer",
        "talk_rows": 0,
        "covered": True,
        "identified": True,
        "first_s": 195.0,
        "ttfi_s": 75.0,
        "lag_s": 0.0,
    }


def test_an_emission_row_joins_its_play_row_even_for_a_carryover_play(data: Path) -> None:
    """A carryover repeats a play_id: the join key is (hour_key, play_id, carryover)."""
    add_shazam(data, ClipAddress(HOUR, 30, 12), HERMANOS)
    add_shazam(data, ClipAddress(HOUR, 195, 12), MOLINA)

    run_cli(data)

    leg = read_score(data)["legs"]["12s/shazam"]
    plays = {(p["hour_key"], p["play_id"], p["carryover"]): p for p in leg["plays"]}
    carryovers = {(p["hour_key"], p["play_id"], p["carryover"]): p for p in leg["carryover_plays"]}
    joined = [
        (plays | carryovers)[e["hour_key"], e["play_id"], e["carryover"]] for e in leg["emissions"]
    ]
    assert [(p["play_id"], p["carryover"]) for p in joined] == [(100, True), (1, False)]
    assert list(carryovers) == [(HOUR, 100, True)]
    assert len(plays) == len(leg["plays"]) == 5  # a carryover is never a scored play


def test_a_leg_without_records_is_scored_with_everything_uncovered(data: Path) -> None:
    run_cli(data)

    leg = read_score(data)["legs"]["12s/shazam"]

    assert leg["recall"] is None
    assert leg["coverage"]["scored_addresses"] == 0
    assert set(read_score(data)["legs"]) == {"12s/shazam"}  # no --snapshot: no Olaf leg


@pytest.mark.parametrize(
    "flag",
    ["--plays", "--out", "--near-misses", "--false-positives", "--selection", "--archive-dir"]
    + ["--store"],
)
def test_a_path_inside_the_checkout_or_relative_is_refused_before_anything_is_read(
    data: Path, flag: str
) -> None:
    for bad in ("evaluation/x", str(Path(score.__file__).parent / "x")):
        with pytest.raises(SystemExit, match="checkout|absolute"):
            score.main(["--plays", str(data / "plays.jsonl"), "--only", "shazam", flag, bad])
    assert not (data / "score").exists()


def test_the_data_dir_default_is_refused_inside_the_checkout(
    data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STREAM_SLEUTH_DATA_DIR", str(Path(score.__file__).parent))
    with pytest.raises(SystemExit, match="checkout"):
        run_cli(data)


def test_a_missing_snapshot_pool_db_is_one_line(data: Path) -> None:
    with pytest.raises(SystemExit, match="no snapshot"):
        run_cli(data, "--snapshot", "absent")
    assert not (data / "olaf").exists()


def legs_of_plays(
    plays: list[score.Play],
    emissions: list[Emission],
    references: dict[str, tuple[str, ...]] | None = None,
) -> dict[str, score.LegScore]:
    """One leg, ``12s/shazam``, holding the verdicts of ``emissions`` against ``plays``."""
    verdicts = attribute(plays, emissions, references)
    return {"12s/shazam": score.LegScore(cast(Any, None), verdicts, [])}


def legs_of(
    emissions: list[Emission], references: dict[str, tuple[str, ...]] | None = None
) -> dict[str, score.LegScore]:
    return legs_of_plays(plays_from(RECORDS), emissions, references)


def near_misses(
    emissions: list[Emission],
    records: list[dict[str, Any]] = RECORDS,
    references: dict[str, tuple[str, ...]] | None = None,
) -> list[dict[str, Any]]:
    plays = plays_from(records)
    return score.near_miss_rows(plays, legs_of_plays(plays, emissions, references), references)


def one_play(artist: str, title: str, *more: tuple[int, float, str, str]) -> list[dict[str, Any]]:
    """Play 1 at 120 s, and any ``more`` as (play id, offset, artist, title)."""
    rows: list[Any] = [(1, 120.0, (artist, title, ""), {})]
    rows += [(i, t, (a, ti, ""), {}) for i, t, a, ti in more]
    return _hour(HOUR, "canonical", *rows)


NEAR_MISSES = [
    pytest.param(("Juana Molina", "Otra Cancion", "DOGA"), 195.0, "artist", id="artist only"),
    pytest.param(("Otra Artista", MOLINA[1], "DOGA"), 195.0, "title", id="title only"),
    pytest.param((COCREDIT, MOLINA[1], MOLINA[2]), 195.0, "title", id="a co-credit from Shazam"),
    pytest.param(
        ("Jessica Pratt", "Back, Baby (Live)", PRATT[2]),
        750.0,
        "artist",
        id="a version the play does not name",
    ),
    pytest.param(
        ("Jessica Pratt", "Back, Baby", PRATT[2]), 150.0, None, id="outside the play's window"
    ),
    pytest.param(("Stereolab", "Drive", ""), 750.0, None, id="nothing in common"),
    pytest.param(MOLINA, 195.0, None, id="a correct emission is not a near miss"),
]


@pytest.mark.parametrize(("track", "at", "matched_on"), NEAR_MISSES)
def test_near_misses_are_wrong_emissions_one_field_from_a_play_in_their_window(
    track: tuple[str, str, str], at: float, matched_on: str | None
) -> None:
    rows = near_misses([_hit(track, at)])

    assert [r["matched_on"] for r in rows] == ([matched_on] if matched_on else [])


def test_a_similar_token_set_is_a_near_miss_when_neither_field_matches() -> None:
    records = one_play("Chuquimamani-Condori", "Call Your Name Tonight Now Forever")
    near = ("Chuquimamani Condori Jr", "Call Your Name Tonight Now Forever Again", "Edits")
    far = ("Chuquimamani Condori Jr", "Call Your Name Tonight Now Forever Again Please", "Edits")

    [hit] = near_misses([_hit(near, 195.0)], records)
    assert hit["matched_on"] == "similarity"
    assert not near_misses([_hit(far, 195.0)], records)


def test_two_empty_artists_do_not_match() -> None:
    assert not near_misses([_hit(("", "Otra Cancion", ""), 195.0)], one_play("", "la paradoja"))


def test_a_run_is_judged_by_every_emission_and_the_nearest_play_wins() -> None:
    """The run starts outside every window of its song's play and enters one later."""
    chuqui = ("Chuquimamani-Condori", "Otra Cancion", "Edits")

    [row] = near_misses([_hit(chuqui, 1200.0), _hit(chuqui, 1335.0)])

    assert (row["address"], row["last_address"], row["play_id"]) == (
        f"{HOUR}#1200+12@128k",
        f"{HOUR}#1335+12@128k",
        4,
    )


def test_among_near_plays_the_most_similar_is_chosen() -> None:
    records = one_play(
        "Jessica Pratt",
        "Something Entirely Different Here",
        (2, 200.0, "Jessica Pratt", "Back, Baby (Live)"),
    )

    [row] = near_misses([_hit(("Jessica Pratt", "Back, Baby", ""), 195.0)], records)

    assert (row["play_id"], row["matched_on"]) == (2, "artist")


def test_a_near_miss_under_olaf_references_names_every_artist_tag_of_its_file() -> None:
    address = ClipAddress(HOUR, 195, 12)
    found = _found(("Mislabeled Tag", "Otra Cancion", "DOGA"), 195.0)
    found |= {"source": "local", "confidence": 40.0, "ref_key": STAGE}
    emission = Emission((address.key, OLAF), address, found)

    [row] = near_misses([emission], references={STAGE: ("Juana Molina",)})

    assert (row["play_id"], row["matched_on"]) == (1, "artist")
    assert not near_misses([emission])


def local_emission(track: tuple[str, str, str], at: float) -> Emission:
    """An Olaf emission of ``track`` whose reference file is ``STAGE``."""
    address = ClipAddress(HOUR, int(at), 12)
    found = _found(track, at) | {"source": "local", "confidence": 40.0, "ref_key": STAGE}
    return Emission((address.key, OLAF), address, found)


ARTISTS = "Chuquimamani Condori Jessica Pratt Hermanos Gutierrez Duo"  # seven tokens


def test_the_similarity_counts_the_artist_tokens_and_not_the_title_alone() -> None:
    """Titles alone overlap at 2/3; with the artists the token sets overlap at 9/11."""
    records = one_play(ARTISTS, "Call Name")

    [row] = near_misses([_hit((f"{ARTISTS} Jr", "Call Name Again", ""), 195.0)], records)

    assert row["matched_on"] == "similarity"


def test_the_similarity_of_an_olaf_match_is_taken_over_every_artist_tag_of_its_file() -> None:
    records = one_play(ARTISTS, "Call Name")
    emission = local_emission(("Mislabeled Tag", "Call Name Again", ""), 195.0)

    [row] = near_misses([emission], records, {STAGE: (f"{ARTISTS} Jr",)})

    assert (row["matched_on"], row["reference_artists"]) == ("similarity", f"{ARTISTS} Jr")
    assert not near_misses([emission], records)


def test_more_shared_fields_win_over_more_similar_tokens() -> None:
    """The first play shares no field but is the more similar; the second shares the artist."""
    records = one_play(ARTISTS, "Call Name", (2, 200.0, f"{ARTISTS} Jr", "Something Else"))

    [row] = near_misses([_hit((f"{ARTISTS} Jr", "Call Name Again", ""), 195.0)], records)

    assert (row["play_id"], row["matched_on"]) == (2, "artist")


def test_a_play_that_shares_both_fields_but_not_its_album_version_names_both() -> None:
    records = _hour(
        HOUR, "canonical", (1, 120.0, ("Juana Molina", "la paradoja", "DOGA (Live)"), {})
    )

    [row] = near_misses([_hit(MOLINA, 195.0)], records)

    assert row["matched_on"] == "artist+title"


def test_a_row_names_the_run_the_play_and_what_an_olaf_match_was_made_on() -> None:
    emission = local_emission(("Mislabeled Tag", "Otra Cancion", "DOGA"), 195.0)

    [row] = near_misses([emission], references={STAGE: ("Juana Molina", "Duo Tag")})

    assert (row["recognizer"], row["reference_artists"], row["play_carryover"]) == (
        OLAF,
        "Juana Molina | Duo Tag",
        False,
    )


def test_a_shazam_row_has_no_reference_artists_and_a_carryover_play_says_so() -> None:
    [row] = near_misses([_hit(("Hermanos Gutiérrez", "Otra Cancion", ""), 30.0)])

    assert (row["recognizer"], row["play_id"], row["play_carryover"], row["reference_artists"]) == (
        SHAZAM,
        100,
        True,
        "",
    )


def test_consecutive_wrong_emissions_of_one_song_are_one_row_with_archive_times() -> None:
    other = ("Juana Molina", "Otra Cancion", "DOGA")
    hits = [_hit(other, at) for at in (195.0, 210.0, 225.0)]
    hits += [_hit(MOLINA, 240.0), _hit(other, 255.0)]

    rows = near_misses(hits)

    assert [(r["address"], r["last_address"], r["emissions"]) for r in rows] == [
        (f"{HOUR}#195+12@128k", f"{HOUR}#225+12@128k", 3),
        (f"{HOUR}#255+12@128k", f"{HOUR}#255+12@128k", 1),
    ]
    assert (rows[0]["hour_key"], rows[0]["start"], rows[0]["end"]) == (HOUR, "0:03:15", "0:03:57")
    assert rows[0]["verdict"] == ""


def read_queue(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


QUEUE = "score/near_misses.csv"
HEADER = (
    "leg,recognizer,address,last_address,emissions,hour_key,start,end,source,artist,song,album,"
    "reference_artists,play_id,play_carryover,play_artist,play_title,play_album,matched_on,verdict"
)
OTHER_SONG = ("Juana Molina", "Otra Canción", "DOGA")


def test_the_cli_writes_the_near_miss_queue_with_an_empty_verdict_column(data: Path) -> None:
    add_shazam(data, ClipAddress(HOUR, 195, 12), OTHER_SONG)
    add_shazam(data, ClipAddress(HOUR, 210, 12), OTHER_SONG)
    add_shazam(data, ClipAddress(HOUR, 750, 12), ("Stereolab", "Drive", ""))

    run_cli(data)

    path = data / QUEUE
    assert path.read_text(encoding="utf-8-sig").splitlines()[0] == HEADER
    [row] = read_queue(path)
    assert (row["leg"], row["emissions"], row["song"], row["play_id"], row["verdict"]) == (
        "12s/shazam",
        "2",
        "Otra Canción",
        "1",
        "",
    )


def test_the_queue_is_utf_8_with_a_byte_order_mark_and_is_rewritten_by_the_next_run(
    data: Path,
) -> None:
    song = ("Hermanos Gutiérrez", "Otra Canción", "")
    add_shazam(data, ClipAddress(HOUR, 30, 12), song)
    run_cli(data)
    queue = data / QUEUE

    first = queue.read_bytes()
    [row] = read_queue(queue)
    run_cli(data)

    assert first.startswith(b"\xef\xbb\xbf")
    assert "Otra Canción".encode() in first and row["artist"] == "Hermanos Gutiérrez"
    assert queue.read_bytes() == first


def test_the_queue_defaults_to_the_directory_of_the_score_file(data: Path) -> None:
    add_shazam(data, ClipAddress(HOUR, 195, 12), OTHER_SONG)
    out = data / "elsewhere" / "shazam-only.json"

    run_cli(data, "--out", str(out))

    assert [r["leg"] for r in read_queue(out.parent / "near_misses.csv")] == ["12s/shazam"]
    assert not (data / "score").exists()


def fill_verdict(path: Path, verdict: str) -> bytes:
    """Put ``verdict`` in the queue's first row, as a person would; return the file's bytes."""
    rows = read_queue(path)
    rows[0]["verdict"] = verdict
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path.read_bytes()


def queued(data: Path) -> Path:
    """Run once over a store holding one near miss; return the queue it wrote."""
    add_shazam(data, ClipAddress(HOUR, 195, 12), OTHER_SONG)
    run_cli(data)
    (data / "score" / "score.json").unlink()
    return data / QUEUE


def test_a_queue_with_a_verdict_is_never_overwritten_and_nothing_else_is_written(
    data: Path,
) -> None:
    queue = queued(data)
    filled = fill_verdict(queue, "correct")

    with pytest.raises(SystemExit, match=r"near_misses\.csv.*verdict"):
        run_cli(data)

    assert queue.read_bytes() == filled
    assert not (data / "score" / "score.json").exists()


def test_a_queue_with_no_verdict_is_rewritten(data: Path) -> None:
    queue = queued(data)
    queue.write_text(queue.read_text(encoding="utf-8") + "stale\n", encoding="utf-8")

    run_cli(data)

    assert len(read_queue(queue)) == 1


ROW = "12s/shazam,rec,a,b,1,h,0:00:00,0:00:12,shazam,x,y,z,,1,False,p,q,r,artist"
BOM = b"\xef\xbb\xbf"
NOT_OURS = [
    pytest.param(
        f"{HEADER.replace('verdict', 'Verdict')}\n{ROW},correct\n".encode(), id="re-cased"
    ),
    pytest.param(BOM + f"{HEADER}\n{ROW},wrong\n".encode(), id="byte-order mark"),
    pytest.param(f"{HEADER}\n{ROW},caf".encode() + b"\xe9\n", id="not UTF-8"),
    pytest.param(b'{"hour_key": "h", "play_id": 1}\n', id="a JSONL file"),
    pytest.param(b"a,b\n1,2\n", id="another CSV"),
    pytest.param(b"", id="an empty file"),
    pytest.param(f"{HEADER.replace(',verdict', '')}\n".encode(), id="a header without verdict"),
]


@pytest.mark.parametrize("content", NOT_OURS)
def test_an_existing_file_that_is_not_an_unfilled_queue_of_ours_is_never_overwritten(
    data: Path, content: bytes
) -> None:
    queue = data / QUEUE
    queue.parent.mkdir()
    queue.write_bytes(content)

    with pytest.raises(SystemExit, match=r"near_misses\.csv"):
        run_cli(data)

    assert queue.read_bytes() == content
    assert not (data / "score" / "score.json").exists()


@pytest.mark.parametrize("prefix", ["", "\ufeff"], ids=["plain", "byte-order mark"])
def test_a_header_only_queue_of_ours_is_rewritten(data: Path, prefix: str) -> None:
    queue = data / QUEUE
    queue.parent.mkdir()
    queue.write_text(prefix + HEADER.replace("verdict", "Verdict") + "\n", encoding="utf-8")

    run_cli(data)

    assert queue.read_text(encoding="utf-8-sig").splitlines()[0] == HEADER


def inputs(data: Path) -> dict[str, Path]:
    """Every input a run reads, each existing, by the flag that names it (or its role)."""
    add_snapshot(data)
    home = data / "olaf" / SNAPSHOT
    add_olaf(data, ClipAddress(HOUR, 195, 12), COCREDIT)
    add_shazam(data, ClipAddress(HOUR, 195, 12), MOLINA)
    return {
        "plays": data / "plays.jsonl",
        "selection": data / "selection.json",
        "shazam store": data / "shazam" / "results.jsonl",
        "olaf store": home / RESULTS,
        "pool.db": home / "pool.db",
    }


@pytest.mark.parametrize("output", ["--out", "--near-misses", "--false-positives"])
@pytest.mark.parametrize("role", ["plays", "selection", "shazam store", "olaf store", "pool.db"])
@pytest.mark.parametrize("alias", ["path", "symlink", "hardlink"])
def test_an_output_that_is_an_input_is_refused_before_anything_is_written(
    data: Path, output: str, role: str, alias: str, tmp_path: Path
) -> None:
    target = inputs(data)[role]
    before = target.read_bytes()
    named = target
    if alias == "symlink":
        named = tmp_path / "alias"
        named.symlink_to(target)
    if alias == "hardlink":
        named = tmp_path / "alias"
        os.link(target, named)

    with pytest.raises(SystemExit, match="input"):
        run_cli(data, "--snapshot", SNAPSHOT, output, str(named))

    assert target.read_bytes() == before
    assert not (data / "score").exists()


def test_the_score_file_and_the_queue_may_not_share_a_path(data: Path) -> None:
    shared = data / "score" / "both"

    with pytest.raises(SystemExit, match="same"):
        run_cli(data, "--out", str(shared), "--near-misses", str(shared))

    assert not (data / "score").exists()


@pytest.mark.parametrize("legs", [["6s"], ["20s", "6s"], ["6s", "12s"]])
def test_an_olaf_leg_without_a_snapshot_is_refused_as_run_py_refuses_it(
    data: Path, legs: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as refusal:
        score.main(["--plays", str(data / "plays.jsonl"), "--legs", *legs])

    assert refusal.value.code == 2
    assert "--snapshot is required for Olaf legs" in capsys.readouterr().err
    assert not (data / "score").exists()


def test_only_shazam_skips_the_olaf_legs_of_a_mixed_selection(data: Path) -> None:
    score.main(["--plays", str(data / "plays.jsonl"), "--only", "shazam", "--legs", "6s", "12s"])

    assert set(read_score(data)["legs"]) == {"12s/shazam"}


def test_only_shazam_with_only_olaf_legs_selects_nothing(
    data: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        score.main(["--plays", str(data / "plays.jsonl"), "--only", "shazam", "--legs", "6s"])

    assert "no leg is selected" in capsys.readouterr().err


def test_a_play_keeps_its_rotation_flag_as_the_producer_wrote_it() -> None:
    records = _hour(
        HOUR, "canonical", (1, 120.0, MOLINA, {}), (2, 600.0, PRATT, {"rotation": False})
    )

    assert [p.rotation for p in plays_from(records)] == [True, False]


def previous_score_file(data: Path) -> bytes:
    """Run once and return the score file it wrote."""
    run_cli(data)
    return (data / "score" / "score.json").read_bytes()


NOT_A_SCORE_FILE = [
    pytest.param(b"STREAM_SLEUTH_POOL_PREFIXES=rotation/\nAWS_PROFILE=wxyc\n", id="an env file"),
    pytest.param(b'{"address": "h#0+12@128k", "kind": "no_match"}\n', id="a JSONL results file"),
    pytest.param(b'{"version": 1}', id="a JSON object missing legs"),
    pytest.param(b'{"version": 1, "legs": {}, "other": 1}', id="a JSON object with another key"),
    pytest.param(b"[]", id="a JSON array"),
    pytest.param(b"", id="an empty file"),
    pytest.param(b"\xff\xfe", id="not UTF-8"),
]


@pytest.mark.parametrize("content", NOT_A_SCORE_FILE)
def test_an_existing_out_that_is_not_a_score_file_is_never_overwritten(
    data: Path, content: bytes
) -> None:
    out = data / "olaf" / "other" / "results.jsonl"  # not an input of a Shazam-only run
    out.parent.mkdir(parents=True)
    out.write_bytes(content)

    with pytest.raises(SystemExit, match=r"results\.jsonl.*not overwriting"):
        run_cli(data, "--out", str(out))

    assert out.read_bytes() == content
    assert not (data / "score").exists()


def test_a_previous_score_file_is_rewritten(data: Path) -> None:
    previous = previous_score_file(data)
    add_shazam(data, ClipAddress(HOUR, 195, 12), MOLINA)

    run_cli(data)

    assert (data / "score" / "score.json").read_bytes() != previous


def test_only_shazam_ignores_snapshot_as_run_py_does(data: Path) -> None:
    score.main(
        ["--plays", str(data / "plays.jsonl"), "--only", "shazam", "--snapshot", "absent"]
        + ["--legs", "12s"]
    )

    assert set(read_score(data)["legs"]) == {"12s/shazam"}
    assert not (data / "olaf").exists()


def test_only_olaf_scores_the_olaf_side_of_a_mixed_leg(data: Path) -> None:
    add_snapshot(data)
    add_olaf(data, ClipAddress(HOUR, 195, 12), COCREDIT)

    score.main(
        ["--plays", str(data / "plays.jsonl"), "--only", "olaf", "--snapshot", SNAPSHOT]
        + ["--legs", "12s"]
    )

    assert set(read_score(data)["legs"]) == {"12s/olaf"}


@pytest.mark.parametrize(
    "content",
    [None, b"{not json\n", b'{"hour_key": "2026082315"}\n'],
    ids=["missing", "not JSON", "a record missing fields"],
)
def test_an_unreadable_plays_file_is_one_line(data: Path, content: bytes | None) -> None:
    plays = data / "plays.jsonl"
    if content is None:
        plays.unlink()
    else:
        plays.write_bytes(content)

    with pytest.raises(SystemExit, match=r"plays\.jsonl"):
        run_cli(data)

    assert not (data / "score").exists()


def test_a_failed_write_leaves_the_previous_score_file_whole(
    data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    previous = previous_score_file(data)
    add_shazam(data, ClipAddress(HOUR, 195, 12), MOLINA)

    def interrupted(src: object, dst: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(score.os, "replace", interrupted)
    with pytest.raises(OSError):
        run_cli(data)

    assert (data / "score" / "score.json").read_bytes() == previous


FP_QUEUE = "score/false_positives.csv"
FP_HEADER = (
    "leg,recognizer,address,last_address,emissions,hour_key,start,end,source,artist,song,album,"
    "reference_artists,preflag,verdict"
)
LIKELY = "likely-unlogged-correct"


def false_positives(
    shazam: list[Emission],
    olaf: list[Emission] | None = None,
    references: dict[str, tuple[str, ...]] | None = None,
) -> list[dict[str, Any]]:
    """The false-positive queue for one 12 s leg, Shazam and (when given) Olaf, against RECORDS."""
    plays = plays_from(RECORDS)
    legs = {"12s/shazam": score.LegScore(cast(Any, None), attribute(plays, shazam), [])}
    if olaf is not None:
        verdicts = attribute(plays, olaf, references)
        legs["12s/olaf"] = score.LegScore(cast(Any, None), verdicts, [])
    return score.false_positive_rows(plays, legs, references)


def unlogged(at: float, ref_start_s: float | None = None, hour: str = HOUR) -> Emission:
    """A Shazam answer for a song no play names; ``ref_start_s`` is its match offset, if any."""
    extra = {} if ref_start_s is None else {"query_offset_s": 0.0, "ref_start_s": ref_start_s}
    return _hit(UNLOGGED, at, hour, **extra)


def test_a_run_of_wrong_emissions_is_one_row_with_its_first_and_last_position() -> None:
    hits = [unlogged(at) for at in (750.0, 765.0, 780.0)] + [_hit(PRATT, 795.0), unlogged(810.0)]

    rows = false_positives(hits)

    assert [(r["address"], r["last_address"], r["emissions"]) for r in rows] == [
        (f"{HOUR}#750+12@128k", f"{HOUR}#780+12@128k", 3),
        (f"{HOUR}#810+12@128k", f"{HOUR}#810+12@128k", 1),
    ]
    assert (rows[0]["leg"], rows[0]["recognizer"], rows[0]["hour_key"]) == (
        "12s/shazam",
        SHAZAM,
        HOUR,
    )
    assert (rows[0]["start"], rows[0]["end"], rows[0]["verdict"]) == ("0:12:30", "0:13:12", "")
    assert (rows[0]["artist"], rows[0]["song"], rows[0]["album"]) == UNLOGGED


def test_a_run_the_near_miss_queue_holds_is_not_repeated() -> None:
    near = ("Juana Molina", "Otra Cancion", "DOGA")  # shares the artist with a play in its window

    rows = false_positives([_hit(near, 195.0), unlogged(750.0)])

    assert [r["song"] for r in rows] == [UNLOGGED[1]]


# Two emissions 15 s apart whose song-start estimates differ by ``drift``: a reference position that
# does not advance with the wall clock differs by the whole 15 s step.
STEADY = [
    pytest.param(0.0, LIKELY, id="identical estimates"),
    pytest.param(7.5, LIKELY, id="at the tolerance, half a grid step"),
    pytest.param(-7.5, LIKELY, id="at the tolerance, the other way"),
    pytest.param(7.6, "", id="just past the tolerance"),
    pytest.param(-7.6, "", id="just past it, the other way"),
    pytest.param(15.0, "", id="a reference position that stands still"),
]


@pytest.mark.parametrize(("drift", "preflag"), STEADY)
def test_a_steady_song_start_across_the_run_is_preflagged(drift: float, preflag: str) -> None:
    [row] = false_positives([unlogged(750.0, 100.0), unlogged(765.0, 115.0 - drift)])

    assert row["preflag"] == preflag


def test_the_estimate_is_judged_across_the_whole_run() -> None:
    steady = [unlogged(750.0, 100.0), unlogged(765.0, 115.0), unlogged(780.0, 130.0)]
    drifting = [*steady[:2], unlogged(780.0, 100.0)]

    assert false_positives(steady)[0]["preflag"] == LIKELY
    assert false_positives(drifting)[0]["preflag"] == ""


@pytest.mark.parametrize(
    "run",
    [
        pytest.param([unlogged(750.0, 100.0)], id="a single emission"),
        pytest.param([unlogged(750.0), unlogged(765.0)], id="Shazam answers with null offsets"),
        pytest.param([unlogged(750.0, 100.0), unlogged(765.0)], id="one with offsets, one without"),
    ],
)
def test_fewer_than_two_emissions_with_offsets_cannot_be_steady(run: list[Emission]) -> None:
    assert false_positives(run)[0]["preflag"] == ""


def other(track: tuple[str, str, str], at: float, hour: str = HOUR) -> Emission:
    """An Olaf answer at ``at`` that names its reference by ``STAGE``."""
    emission = _local(track[0], at, STAGE)
    found = {**emission.found, "song": track[1], "album": track[2]}
    address = ClipAddress(hour, int(at), 12)
    return Emission((address.key, OLAF), address, cast(Any, found))


UNLOGGED_REMASTER = (UNLOGGED[0], f"{UNLOGGED[1]} (Remastered)", UNLOGGED[2])
AGREEMENT = [
    pytest.param(UNLOGGED, 765.0, HOUR, LIKELY, id="the same song inside the run's span"),
    pytest.param(UNLOGGED_REMASTER, 750.0, HOUR, LIKELY, id="equal under names.fuzzy"),
    pytest.param(UNLOGGED, 735.0, HOUR, "", id="before the run"),
    pytest.param(UNLOGGED, 780.0, HOUR, "", id="after the run's last capture ends"),
    pytest.param(UNLOGGED, 765.0, OTHER, "", id="in another hour"),
    pytest.param((UNLOGGED[0], "Another Song", ""), 765.0, HOUR, "", id="another title"),
    pytest.param(("Hermanos Gutiérrez", UNLOGGED[1], ""), 765.0, HOUR, "", id="another artist"),
]


@pytest.mark.parametrize(("track", "at", "hour", "preflag"), AGREEMENT)
def test_the_other_recognizer_naming_the_same_song_in_the_runs_span_is_preflagged(
    track: tuple[str, str, str], at: float, hour: str, preflag: str
) -> None:
    rows = false_positives([unlogged(750.0), unlogged(765.0)], [other(track, at, hour)])

    assert [r["preflag"] for r in rows if r["recognizer"] == SHAZAM] == [preflag]


def test_each_recognizers_run_is_flagged_by_the_other() -> None:
    rows = false_positives([unlogged(750.0)], [other(UNLOGGED, 750.0)])

    assert {r["leg"]: r["preflag"] for r in rows} == {"12s/shazam": LIKELY, "12s/olaf": LIKELY}


def test_a_shazam_answer_naming_the_album_artist_agrees_with_an_olaf_reference_tag() -> None:
    credited = ("Stereolab & Duo Tag", UNLOGGED[1], UNLOGGED[2])
    references: dict[str, tuple[str, ...]] = {STAGE: ("Stereolab & Duo Tag", "Stereolab")}

    flagged = false_positives([unlogged(750.0)], [other(credited, 750.0)], references)
    bare = false_positives([unlogged(750.0)], [other(credited, 750.0)])

    assert [r["preflag"] for r in flagged if r["recognizer"] == SHAZAM] == [LIKELY]
    assert [r["preflag"] for r in bare if r["recognizer"] == SHAZAM] == [""]
    assert [r["reference_artists"] for r in flagged if r["recognizer"] == OLAF] == [
        "Stereolab & Duo Tag | Stereolab"
    ]


def test_a_run_is_not_flagged_by_a_different_capture_length() -> None:
    plays = plays_from(RECORDS)
    verdicts = attribute(plays, [unlogged(750.0)])
    short = ClipAddress(HOUR, 750, 6)
    answer = Emission((short.key, OLAF), short, cast(Any, _found(UNLOGGED, 750.0, source="local")))
    legs = {
        "12s/shazam": score.LegScore(cast(Any, None), verdicts, []),
        "6s/olaf": score.LegScore(cast(Any, None), attribute(plays, [answer]), []),
    }

    rows = score.false_positive_rows(plays, legs, None)

    assert {r["leg"]: r["preflag"] for r in rows} == {"12s/shazam": "", "6s/olaf": ""}


def test_the_cli_writes_the_false_positive_queue_beside_the_score_file(data: Path) -> None:
    add_shazam(data, ClipAddress(HOUR, 750, 12), UNLOGGED, offset_s=100.0)
    add_shazam(data, ClipAddress(HOUR, 765, 12), UNLOGGED, offset_s=115.0)
    add_shazam(data, ClipAddress(HOUR, 195, 12), OTHER_SONG)  # a near miss: the other queue's

    run_cli(data)

    path = data / FP_QUEUE
    assert path.read_text(encoding="utf-8-sig").splitlines()[0] == FP_HEADER
    [row] = read_queue(path)
    assert (row["emissions"], row["song"], row["preflag"], row["verdict"]) == (
        "2",
        "French Disko",
        LIKELY,
        "",
    )
    assert [r["song"] for r in read_queue(data / QUEUE)] == ["Otra Canción"]


def test_the_false_positive_queue_follows_out_and_its_own_flag(data: Path) -> None:
    add_shazam(data, ClipAddress(HOUR, 750, 12), UNLOGGED)
    out = data / "elsewhere" / "shazam-only.json"

    run_cli(data, "--out", str(out))
    run_cli(data, "--out", str(out), "--false-positives", str(data / "fp.csv"))

    assert [r["leg"] for r in read_queue(out.parent / "false_positives.csv")] == ["12s/shazam"]
    assert [r["leg"] for r in read_queue(data / "fp.csv")] == ["12s/shazam"]


def test_a_false_positive_queue_with_a_verdict_is_never_overwritten_and_nothing_is_written(
    data: Path,
) -> None:
    add_shazam(data, ClipAddress(HOUR, 750, 12), UNLOGGED)
    run_cli(data)
    queue = data / FP_QUEUE
    filled = fill_verdict(queue, "unlogged-correct")
    near = (data / QUEUE).read_bytes()
    (data / "score" / "score.json").unlink()

    with pytest.raises(SystemExit, match=r"false_positives\.csv.*verdict"):
        run_cli(data)

    assert queue.read_bytes() == filled
    assert (data / QUEUE).read_bytes() == near
    assert not (data / "score" / "score.json").exists()


def test_a_filled_false_positive_queue_stops_the_run_before_the_near_miss_queue_is_rewritten(
    data: Path,
) -> None:
    add_shazam(data, ClipAddress(HOUR, 195, 12), OTHER_SONG)
    add_shazam(data, ClipAddress(HOUR, 750, 12), UNLOGGED)
    run_cli(data)
    fill_verdict(data / FP_QUEUE, "talk")
    (data / QUEUE).unlink()

    with pytest.raises(SystemExit, match=r"false_positives\.csv"):
        run_cli(data)

    assert not (data / QUEUE).exists()


def test_the_two_queues_may_not_share_a_path(data: Path) -> None:
    shared = data / "score" / "both.csv"

    with pytest.raises(SystemExit, match="same"):
        run_cli(data, "--near-misses", str(shared), "--false-positives", str(shared))

    assert not (data / "score").exists()
