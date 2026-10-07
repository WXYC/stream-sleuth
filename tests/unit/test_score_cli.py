"""``python -m evaluation.score``: the score file and the near-miss queue.

Every plays file, store, and ``pool.db`` is synthetic and lives under a ``tmp_path`` data
directory. ``hour_addresses`` is replaced, so no test needs ffmpeg or an archive.
"""

from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path
from typing import Any

import pytest

from evaluation import queues, score
from evaluation.clips import ClipAddress, grid
from evaluation.olaf_snapshot import BUILDING, MARKER, POOL_DB, RESULTS, snapshot_lock
from evaluation.pool import open_pool_db
from evaluation.results import ResultStore, recognizer_identity
from evaluation.score import plays_from
from stream_sleuth.recognizers.olaf import OLAF_COMMIT
from stream_sleuth.recognizers.olaf import recognizer_identity as olaf_identity
from tests.stores import olaf_record, shazam_record
from tests.unit.test_queues import LIKELY, OLAF_PLAYBACK
from tests.unit.test_score import (
    COCREDIT,
    HERMANOS,
    HOUR,
    MOLINA,
    OTHER,
    PRATT,
    RECORDS,
    STAGE,
    UNLOGGED,
    _hit,
    _hour,
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


def add_snapshot(
    data: Path, artist: str = COCREDIT, album_artist: str = "Juana Molina", built: bool = True
) -> Path:
    """A snapshot whose one indexed file is the co-credited reference ``STAGE``, marked built."""
    home = data / "olaf" / SNAPSHOT
    db = open_pool_db(home / POOL_DB)
    db.execute(
        "INSERT INTO files (key, stage_id, prefix, format, size, artist, album_artist, album,"
        " title, status) VALUES ('k', ?, 'rotation/', 'mp3', 1, ?, ?, ?, ?, 'indexed')",
        (STAGE, artist, album_artist, MOLINA[2], MOLINA[1]),
    )
    db.commit()
    db.close()
    if built:
        marker = {"olaf_commit": OLAF_COMMIT, "indexed": 1, "failed": 0, "source": "build"}
        (home / MARKER).write_text(json.dumps(marker))
    return home


def add_olaf(
    data: Path,
    address: ClipAddress,
    artist: str,
    query_offset_s: float = 0.0,
    ref_start_s: float = 75.0,
    song: str = MOLINA[1],
    album: str = MOLINA[2],
) -> None:
    olaf_record(
        ResultStore(data / "olaf" / SNAPSHOT / RESULTS),
        str(address),
        "matched",
        (artist, song, album),
        identity=OLAF,
        ref_key=STAGE,
        query_offset_s=query_offset_s,
        ref_start_s=ref_start_s,
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
        "adjudicated_runs",
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
        "wrong_runs",
    ]
    assert (leg["wrong_runs"], leg["adjudicated_runs"]) == (0, 0)
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
    assert (leg["adjudicated_precision"], leg["adjudicated_distinct_precision"]) == (1.0, 1.0)
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


def test_a_malformed_snapshot_name_is_refused_in_one_line(data: Path) -> None:
    with pytest.raises(SystemExit, match="plain path component") as refusal:
        run_cli(data, "--snapshot", "../x")
    assert "\n" not in str(refusal.value)
    assert not (data / "score").exists()


def test_a_snapshot_that_is_not_built_is_refused_in_one_line_as_run_py_refuses_it(
    data: Path,
) -> None:
    home = add_snapshot(data, built=False)
    with pytest.raises(SystemExit, match="no completion marker") as refusal:
        run_cli(data, "--snapshot", SNAPSHOT)
    assert "\n" not in str(refusal.value)
    (home / MARKER).write_text("{}")  # a stale marker
    (home / BUILDING).write_text("")  # beside an interrupted build
    with pytest.raises(SystemExit, match="interrupted"):
        run_cli(data, "--snapshot", SNAPSHOT)
    assert not (data / "score").exists()
    assert not (home / RESULTS).exists()


def test_scoring_does_not_take_the_snapshot_lock_so_it_reads_a_running_query(data: Path) -> None:
    home = add_snapshot(data)
    add_olaf(data, ClipAddress(HOUR, 195, 12), COCREDIT)
    with snapshot_lock(home):  # as a running ``evaluation.run`` holds it
        assert run_cli(data, "--snapshot", SNAPSHOT) == 0
    assert (data / "score" / "score.json").exists()


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


def test_a_queue_with_a_verdict_is_kept_byte_for_byte_and_read_by_the_rescore(
    data: Path,
) -> None:
    queue = queued(data)
    filled = fill_verdict(queue, "correct")

    assert run_cli(data) == 0

    assert queue.read_bytes() == filled
    leg = read_score(data)["legs"]["12s/shazam"]
    assert (leg["precision"], leg["adjudicated_precision"]) == (0.0, 1.0)


def test_a_queue_with_no_verdict_is_rewritten(data: Path) -> None:
    queue = queued(data)
    queue.write_text(queue.read_text(encoding="utf-8") + "stale\n", encoding="utf-8")

    run_cli(data)

    assert len(read_queue(queue)) == 1


ROW = "12s/shazam,rec,a,b,1,h,0:00:00,0:00:12,shazam,x,y,z,,1,False,p,q,r,artist"
BOM = b"\xef\xbb\xbf"
NOT_OURS = [
    pytest.param(f"{HEADER}\n{ROW},caf".encode() + b"\xe9\n", id="not UTF-8"),
    pytest.param(b'{"hour_key": "h", "play_id": 1}\n', id="a JSONL file"),
    pytest.param(b"a,b\n1,2\n", id="another CSV"),
    pytest.param(b"", id="an empty file"),
    pytest.param(f"{HEADER.replace(',verdict', '')}\n".encode(), id="a header without verdict"),
]


FILLED = [
    pytest.param(
        f"{HEADER.replace('verdict', 'Verdict')}\n{ROW},correct\n".encode(), id="re-cased"
    ),
    pytest.param(BOM + f"{HEADER}\n{ROW},wrong\n".encode(), id="byte-order mark"),
]


@pytest.mark.parametrize("content", FILLED)
def test_a_filled_queue_of_ours_is_kept_whatever_its_rows_say(data: Path, content: bytes) -> None:
    """Its one row names a run nothing scored, so it is stale: ignored, never deleted."""
    queue = data / QUEUE
    queue.parent.mkdir()
    queue.write_bytes(content)

    assert run_cli(data) == 0

    assert queue.read_bytes() == content


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


def test_a_filled_false_positive_queue_is_kept_and_an_unfilled_near_miss_queue_rewritten(
    data: Path,
) -> None:
    add_shazam(data, ClipAddress(HOUR, 195, 12), OTHER_SONG)
    add_shazam(data, ClipAddress(HOUR, 750, 12), UNLOGGED)
    run_cli(data)
    filled = fill_verdict(data / FP_QUEUE, "talk")
    near = data / QUEUE
    near.write_text(near.read_text(encoding="utf-8") + "stale\n", encoding="utf-8")

    run_cli(data)

    assert (data / FP_QUEUE).read_bytes() == filled
    assert len(read_queue(near)) == 1


def test_the_two_queues_may_not_share_a_path(data: Path) -> None:
    shared = data / "score" / "both.csv"

    with pytest.raises(SystemExit, match="same"):
        run_cli(data, "--near-misses", str(shared), "--false-positives", str(shared))

    assert not (data / "score").exists()


def test_the_cli_preflags_a_stored_olaf_playback_through_the_read_side(data: Path) -> None:
    add_snapshot(data, UNLOGGED[0], UNLOGGED[0])  # tags that share nothing with a logged play
    for at, q, r in OLAF_PLAYBACK[:2]:
        add_olaf(data, ClipAddress(HOUR, int(at), 12), UNLOGGED[0], q, r, UNLOGGED[1], UNLOGGED[2])

    run_cli(data, "--snapshot", SNAPSHOT)

    [row] = [r for r in read_queue(data / FP_QUEUE) if r["leg"] == "12s/olaf"]
    assert (row["emissions"], row["song"], row["preflag"]) == ("2", UNLOGGED[1], LIKELY)


def mixed_store(data: Path) -> None:
    """Four emissions: one correct (a run), a near miss (a run), and a false-positive run of two."""
    add_shazam(data, ClipAddress(HOUR, 195, 12), OTHER_SONG)
    add_shazam(data, ClipAddress(HOUR, 210, 12), MOLINA)
    add_shazam(data, ClipAddress(HOUR, 750, 12), UNLOGGED)
    add_shazam(data, ClipAddress(HOUR, 765, 12), UNLOGGED)


def fill_queues(data: Path, near: str, fp: str) -> tuple[bytes, bytes]:
    """Fill the first row of each queue (a blank leaves it as written); return their bytes."""
    return fill_verdict(data / QUEUE, near), fill_verdict(data / FP_QUEUE, fp)


# (near-miss verdict, false-positive verdict) -> (adjudicated precision, adjudicated distinct).
# Flowsheet figures are 1/4 over emissions and 1/3 over runs (correct, near miss, false positive).
ADJUDICATED = [
    pytest.param("", "", 0.25, 1 / 3, id="blank verdicts stay wrong"),
    pytest.param("wrong", "wrong", 0.25, 1 / 3, id="wrong stays wrong"),
    pytest.param("correct", "", 0.5, 2 / 3, id="a correct near miss counts as correct"),
    pytest.param("", "unlogged-correct", 0.75, 2 / 3, id="the whole two-emission run is correct"),
    pytest.param("", "talk", 0.5, 0.5, id="talk leaves both the numerator and the denominator"),
    pytest.param("correct", "talk", 1.0, 1.0, id="both at once"),
    pytest.param(" Correct ", " Unlogged-Correct ", 1.0, 1.0, id="case and blanks"),
]


@pytest.mark.parametrize(("near", "fp", "emission_level", "distinct"), ADJUDICATED)
def test_a_rescore_reads_the_verdicts_and_moves_only_the_adjudicated_figures(
    data: Path, near: str, fp: str, emission_level: float, distinct: float
) -> None:
    mixed_store(data)
    run_cli(data)
    before = read_score(data)["legs"]["12s/shazam"]
    assert (before["adjudicated_precision"], before["adjudicated_distinct_precision"]) == (
        before["precision"],
        before["distinct_precision"],
    )
    filled = fill_queues(data, near, fp)

    assert run_cli(data) == 0

    for path, entry, written in zip((QUEUE, FP_QUEUE), (near, fp), filled, strict=True):
        assert (data / path).read_bytes() == written or not entry  # an unfilled queue is rewritten
    after = read_score(data)["legs"]["12s/shazam"]
    assert (after["precision"], after["distinct_precision"]) == (0.25, 1 / 3)
    assert after["adjudicated_precision"] == pytest.approx(emission_level)
    assert after["adjudicated_distinct_precision"] == pytest.approx(distinct)
    assert {k: v for k, v in after.items() if not k.startswith("adjudicated")} == {
        k: v for k, v in before.items() if not k.startswith("adjudicated")
    }


def test_a_rescore_leaves_both_queue_files_byte_identical(data: Path) -> None:
    mixed_store(data)
    run_cli(data)
    filled = fill_queues(data, "correct", "unlogged-correct")

    run_cli(data)
    run_cli(data)

    assert ((data / QUEUE).read_bytes(), (data / FP_QUEUE).read_bytes()) == filled


def test_a_row_naming_no_current_run_is_logged_once_and_ignored(
    data: Path, caplog: pytest.LogCaptureFixture
) -> None:
    queue = queued(data)  # one near miss at 195
    filled = fill_verdict(queue, "correct")
    (data / "shazam" / "results.jsonl").unlink()
    add_shazam(data, ClipAddress(HOUR, 210, 12), MOLINA)  # that run is gone
    caplog.set_level("WARNING")

    run_cli(data)

    stale = [r for r in caplog.records if "near_misses.csv" in r.getMessage()]
    assert len(stale) == 1 and "row(s) 2" in stale[0].getMessage()
    assert queue.read_bytes() == filled
    leg = read_score(data)["legs"]["12s/shazam"]
    assert leg["adjudicated_precision"] == leg["precision"] == 1.0


UNKNOWN = [
    pytest.param(QUEUE, "maybe", id="a near miss, unknown"),
    pytest.param(QUEUE, "talk", id="a near miss, a false-positive value"),
    pytest.param(QUEUE, "unlogged-correct", id="a near miss, the other queue's value"),
    pytest.param(FP_QUEUE, "correct", id="a false positive, a near-miss value"),
    pytest.param(FP_QUEUE, "yes", id="a false positive, unknown"),
]


@pytest.mark.parametrize(("queue", "verdict"), UNKNOWN)
def test_an_unknown_verdict_is_refused_in_one_line_naming_the_file_row_and_value(
    data: Path, queue: str, verdict: str
) -> None:
    mixed_store(data)
    run_cli(data)
    (data / "score" / "score.json").unlink()
    filled = fill_verdict(data / queue, verdict)

    with pytest.raises(SystemExit) as refusal:
        run_cli(data)

    message = str(refusal.value)
    assert "\n" not in message
    assert Path(queue).name in message and "row 2" in message and repr(verdict) in message
    assert (data / queue).read_bytes() == filled
    assert not (data / "score" / "score.json").exists()


def save_like_a_spreadsheet(path: Path) -> bytes:
    """Rewrite a queue as Excel or Sheets saves it: a byte-order mark, CRLF line endings, every
    field quoted, and booleans upper-cased."""
    rows = read_queue(path)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, list(rows[0]), quoting=csv.QUOTE_ALL, lineterminator="\r\n")
        writer.writeheader()
        writer.writerows(
            {k: v.upper() if v in ("True", "False") else v for k, v in r.items()} for r in rows
        )
    return path.read_bytes()


def test_verdicts_survive_a_save_in_a_spreadsheet(data: Path) -> None:
    mixed_store(data)
    run_cli(data)
    fill_queues(data, "correct", "unlogged-correct")
    saved = (save_like_a_spreadsheet(data / QUEUE), save_like_a_spreadsheet(data / FP_QUEUE))
    assert b"FALSE" in saved[0] and saved[0].startswith(BOM) and b"\r\n" in saved[0]

    run_cli(data)

    assert ((data / QUEUE).read_bytes(), (data / FP_QUEUE).read_bytes()) == saved
    leg = read_score(data)["legs"]["12s/shazam"]
    assert (leg["adjudicated_precision"], leg["adjudicated_runs"]) == (1.0, 2)


def test_the_score_file_says_how_many_wrong_runs_there_are_and_how_many_were_judged(
    data: Path,
) -> None:
    mixed_store(data)
    run_cli(data)
    leg = read_score(data)["legs"]["12s/shazam"]
    assert (leg["wrong_runs"], leg["adjudicated_runs"]) == (2, 0)

    fill_queues(
        data, "", "wrong"
    )  # a recognized verdict counts, even a wrong one; a blank does not
    run_cli(data)

    leg = read_score(data)["legs"]["12s/shazam"]
    assert (leg["wrong_runs"], leg["adjudicated_runs"], leg["adjudicated_precision"]) == (
        2,
        1,
        0.25,
    )


def set_cells(path: Path, leg: str, **cells: str) -> bytes:
    """Set cells in the rows of ``leg``, as a person would; return the file's bytes."""
    rows = read_queue(path)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows({**r, **cells} if r["leg"] == leg else r for r in rows)
    return path.read_bytes()


def two_legs(data: Path) -> None:
    """One false-positive run of one emission at one address, for both recognizers of the leg."""
    add_snapshot(data, UNLOGGED[0], UNLOGGED[0])
    add_shazam(data, ClipAddress(HOUR, 750, 12), UNLOGGED)
    add_olaf(data, ClipAddress(HOUR, 750, 12), UNLOGGED[0], song=UNLOGGED[1], album=UNLOGGED[2])


def adjudicated(data: Path) -> dict[str, Any]:
    legs = read_score(data)["legs"]
    return {k: (v["adjudicated_precision"], v["adjudicated_runs"]) for k, v in legs.items()}


@pytest.mark.parametrize(
    ("shazam", "olaf", "expected"),
    [
        pytest.param("unlogged-correct", "wrong", {"shazam": (1.0, 1), "olaf": (0.0, 1)}, id="a"),
        pytest.param("wrong", "unlogged-correct", {"shazam": (0.0, 1), "olaf": (1.0, 1)}, id="b"),
        pytest.param("talk", "unlogged-correct", {"shazam": (None, 1), "olaf": (1.0, 1)}, id="c"),
        pytest.param("unlogged-correct", "", {"shazam": (1.0, 1), "olaf": (0.0, 0)}, id="d"),
    ],
)
def test_a_verdict_reaches_only_its_own_leg_at_a_shared_address(
    data: Path, shazam: str, olaf: str, expected: dict[str, tuple[float | None, int]]
) -> None:
    two_legs(data)
    run_cli(data, "--snapshot", SNAPSHOT)
    set_cells(data / FP_QUEUE, "12s/shazam", verdict=shazam)
    set_cells(data / FP_QUEUE, "12s/olaf", verdict=olaf)

    run_cli(data, "--snapshot", SNAPSHOT)

    assert adjudicated(data) == {f"12s/{k}": v for k, v in expected.items()}


def test_a_row_judged_under_another_identity_of_the_same_leg_is_stale(
    data: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The same leg and address, but another match floor: a different run, not this one."""
    two_legs(data)
    run_cli(data, "--snapshot", SNAPSHOT)
    other_floor = olaf_identity(SNAPSHOT, 7)
    set_cells(data / FP_QUEUE, "12s/olaf", recognizer=other_floor, verdict="unlogged-correct")
    caplog.set_level("WARNING")

    run_cli(data, "--snapshot", SNAPSHOT)

    assert adjudicated(data)["12s/olaf"] == (0.0, 0)
    assert any("no current run matches" in r.getMessage() for r in caplog.records)


def test_two_rows_for_one_run_are_refused_naming_both(data: Path) -> None:
    mixed_store(data)
    run_cli(data)
    queue = data / FP_QUEUE
    lines = queue.read_text(encoding="utf-8-sig").splitlines()
    queue.write_text("\n".join([*lines, lines[1]]) + "\n", encoding="utf-8")
    fill_verdict(queue, "talk")
    (data / "score" / "score.json").unlink()

    with pytest.raises(SystemExit, match=r"false_positives\.csv.*rows 2 and 3") as refusal:
        run_cli(data)

    assert "\n" not in str(refusal.value)
    assert not (data / "score" / "score.json").exists()


def test_a_run_with_no_row_in_a_kept_queue_is_logged_by_address(
    data: Path, caplog: pytest.LogCaptureFixture
) -> None:
    queued(data)
    fill_verdict(data / QUEUE, "correct")
    add_shazam(data, ClipAddress(HOUR, 210, 12), MOLINA)  # ends the first run
    add_shazam(data, ClipAddress(HOUR, 300, 12), OTHER_SONG)  # a new near-miss run
    caplog.set_level("WARNING")

    run_cli(data)

    [message] = [m for r in caplog.records if "have no row" in (m := r.getMessage())]
    assert f"{HOUR}#300+12@128k" in message and f"{HOUR}#195+12@128k" not in message


def test_a_bad_false_positive_queue_stops_the_run_before_the_near_miss_queue_is_rewritten(
    data: Path,
) -> None:
    mixed_store(data)
    run_cli(data)
    near = data / QUEUE
    near.write_text(near.read_text(encoding="utf-8") + "stale\n", encoding="utf-8")
    kept = near.read_bytes()
    fill_verdict(data / FP_QUEUE, "maybe")

    with pytest.raises(SystemExit, match="maybe"):
        run_cli(data)

    assert near.read_bytes() == kept


def test_write_queue_refuses_a_file_with_a_verdict_in_it(data: Path) -> None:
    mixed_store(data)
    run_cli(data)
    filled = fill_verdict(data / QUEUE, "correct")

    with pytest.raises(SystemExit, match="verdicts filled in"):
        queues.write_queue(data / QUEUE, queues.NEAR_COLUMNS, [])

    assert (data / QUEUE).read_bytes() == filled


def test_a_distinct_run_is_dropped_only_when_every_emission_in_it_is_talk() -> None:
    """One song across a wrong run marked talk, a correct emission, and a wrong run, then a run
    that is nothing but talk, then a wrong one."""
    play = plays_from(RECORDS)[1]
    first, correct, last = (_hit(UNLOGGED, at) for at in (750.0, 765.0, 780.0))
    talk, wrong = _hit(MOLINA, 900.0), _hit(PRATT, 915.0)
    verdicts = [
        score.Verdict(e, p, False)
        for e, p in ((first, None), (correct, play), (last, None), (talk, None), (wrong, None))
    ]
    calls = {first.address.key: "talk", talk.address.key: "talk"}

    assert (
        score.distinct_precision(verdicts, calls) == 0.5
    )  # the first run is correct, the third not
    assert score.precision(verdicts, calls) == 1 / 3


def test_a_run_that_grew_after_it_was_labeled_is_logged_and_not_applied(
    data: Path, caplog: pytest.LogCaptureFixture
) -> None:
    mixed_store(data)
    run_cli(data)
    filled = fill_verdict(data / FP_QUEUE, "unlogged-correct")  # the run of two at 750 and 765
    add_shazam(data, ClipAddress(HOUR, 780, 12), UNLOGGED)  # the run now ends later and holds three
    caplog.set_level("WARNING")

    run_cli(data)

    [grown] = [r.getMessage() for r in caplog.records if "grown" in r.getMessage()]
    assert "false_positives.csv" in grown and "row 2" in grown
    assert re.search(r"2 emissions; now \S+, 3\); ignored", grown)
    assert (data / FP_QUEUE).read_bytes() == filled
    leg = read_score(data)["legs"]["12s/shazam"]
    assert leg["adjudicated_runs"] == 0
    assert leg["adjudicated_precision"] == leg["precision"]


@pytest.mark.parametrize("emissions", ["2", "2.0", " 2 ", "02"])
def test_an_unchanged_run_still_applies_whatever_a_spreadsheet_made_of_its_extent(
    data: Path, emissions: str
) -> None:
    mixed_store(data)
    run_cli(data)
    fill_verdict(data / FP_QUEUE, "unlogged-correct")
    [row] = read_queue(data / FP_QUEUE)
    set_cells(
        data / FP_QUEUE, "12s/shazam", emissions=emissions, last_address=f" {row['last_address']} "
    )

    run_cli(data)

    assert read_score(data)["legs"]["12s/shazam"]["adjudicated_runs"] == 1


def test_a_run_that_filled_a_gap_in_its_middle_is_not_applied(
    data: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The first and last addresses are unchanged; only the count tells the run has grown."""
    add_shazam(data, ClipAddress(HOUR, 750, 12), UNLOGGED)
    add_shazam(data, ClipAddress(HOUR, 780, 12), UNLOGGED)
    run_cli(data)
    filled = fill_verdict(data / FP_QUEUE, "unlogged-correct")
    add_shazam(data, ClipAddress(HOUR, 765, 12), UNLOGGED)
    caplog.set_level("WARNING")

    run_cli(data)

    [grown] = [r.getMessage() for r in caplog.records if "grown" in r.getMessage()]
    assert re.search(r"2 emissions; now \S+, 3\); ignored", grown)
    assert (data / FP_QUEUE).read_bytes() == filled
    assert read_score(data)["legs"]["12s/shazam"]["adjudicated_runs"] == 0


@pytest.mark.parametrize("emissions", ["2.9", "2.5", "1.99", "two", ""])
def test_an_emissions_cell_that_is_not_the_whole_count_is_a_mismatch_never_truncated(
    data: Path, caplog: pytest.LogCaptureFixture, emissions: str
) -> None:
    mixed_store(data)
    run_cli(data)
    fill_verdict(data / FP_QUEUE, "unlogged-correct")
    set_cells(data / FP_QUEUE, "12s/shazam", emissions=emissions)
    caplog.set_level("WARNING")

    run_cli(data)

    assert any("grown or shrunk" in r.getMessage() for r in caplog.records)
    assert read_score(data)["legs"]["12s/shazam"]["adjudicated_runs"] == 0
