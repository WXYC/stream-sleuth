"""``python -m evaluation.score``: the score file.

Every plays file, store, and ``pool.db`` is synthetic and lives under a ``tmp_path`` data
directory. ``hour_addresses`` is replaced, so no test needs ffmpeg or an archive.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from evaluation import score
from evaluation.clips import ClipAddress, grid
from evaluation.olaf_snapshot import RESULTS
from evaluation.pool import open_pool_db
from evaluation.run import OlafOutcome
from evaluation.score import plays_from
from evaluation.shazam_eval import ResultStore, ShazamOutcome, recognizer_identity
from stream_sleuth.recognizers.olaf import recognizer_identity as olaf_identity
from tests.unit.test_score import (
    COCREDIT,
    HERMANOS,
    HOUR,
    MOLINA,
    OTHER,
    PRATT,
    RECORDS,
    STAGE,
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
    outcome = OlafOutcome(
        0,
        "matched",
        artist,
        MOLINA[1],
        MOLINA[2],
        "",
        confidence=40.0,
        ref_key=STAGE,
        query_offset_s=0.0,
        ref_start_s=75.0,
    )
    ResultStore(data / "olaf" / SNAPSHOT / RESULTS).append(str(address), OLAF, outcome)


def add_shazam(data: Path, address: ClipAddress, track: tuple[str, str, str], **extra: Any) -> None:
    artist, song, album = track
    outcome = ShazamOutcome(200, "matched", artist, song, album, "", **extra)
    (data / "shazam").mkdir(exist_ok=True)
    ResultStore(data / "shazam" / "results.jsonl").append(str(address), SHAZAM12, outcome)


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
            ResultStore(data / "olaf" / SNAPSHOT / RESULTS).append(
                str(a), OLAF, OlafOutcome(0, "no_match")
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


@pytest.mark.parametrize("flag", ["--plays", "--out", "--selection", "--archive-dir", "--store"])
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


@pytest.mark.parametrize("role", ["plays", "selection", "shazam store", "olaf store", "pool.db"])
@pytest.mark.parametrize("alias", ["path", "symlink", "hardlink"])
def test_an_output_that_is_an_input_is_refused_before_anything_is_written(
    data: Path, role: str, alias: str, tmp_path: Path
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
        run_cli(data, "--snapshot", SNAPSHOT, "--out", str(named))

    assert target.read_bytes() == before
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
