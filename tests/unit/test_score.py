"""``evaluation.score``: attribution, coverage, recall, precision, time to first identification, lag.

Every play, store, and emissions file is synthetic. No test reads research data, decodes audio,
or contacts a recognizer: grids come from ``clips.grid`` with an explicit hour length.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import pytest

from evaluation import score
from evaluation.clips import ClipAddress, grid
from evaluation.run import Emission, Leg, read_results
from evaluation.score import Play, attribute, load_emissions, plays_from, score_plays
from evaluation.shazam_eval import ResultStore
from stream_sleuth.recognizers.base import EvalIdentification

HOUR = "2026/08/12/202608121600.mp3"
EMPTY = "2026/08/12/202608121700.mp3"  # gridded, never queried
SHORT = "2026/08/12/202608121800.mp3"  # decodes to 1,800 s
SHAZAM = "shazam@0.8.1, segment=12"
LEG = Leg("12s", 12, "128k", "all")
PADS = {"canonical": 180.0, "etl": 220.0}

MOLINA = ("Juana Molina", "la paradoja", "DOGA")
PRATT = ("Jessica Pratt", "Back, Baby", "On Your Own Love Again")
CHUQUI = ("Chuquimamani-Condori", "Call Your Name", "Edits")
HERMANOS = ("Hermanos Gutiérrez", "El Bueno y el Malo", "El Bueno y el Malo")
ELLINGTON = ("Duke Ellington & John Coltrane", "In a Sentimental Mood", "Duke Ellington & John Coltrane")  # fmt: skip
UNLOGGED = ("Stereolab", "French Disko", "Jenny Ondioline")


def _hour(
    hour_key: str, era: str, *rows: tuple[int, float, tuple[str, str, str], dict]
) -> list[dict]:
    """``plays.jsonl`` records for one hour, windowed as ``corpus.write_plays`` windows them.

    The first row is the carryover when its offset is negative.
    """
    records = []
    for i, (play_id, t, (artist, title, album), extra) in enumerate(rows):
        t_next = rows[i + 1][1] if i + 1 < len(rows) else 3600.0
        records.append(
            {
                "hour_key": hour_key,
                "play_id": play_id,
                "t_offset_s": t,
                "window_start_s": max(0.0, t - PADS[era]),
                "window_end_s": min(3600.0, t_next + PADS[era]),
                "artist": artist,
                "title": title,
                "album": album,
                "era": era,
                "pad_s": PADS[era],
                "reorder_flag": False,
                "play_order_status": "single_writer",
                "carryover": t < 0,
                **extra,
            }  # fmt: skip
        )
    return records


# Windows: carryover [0, 300], 1 [0, 780], 2 [420, 1080], 3 [720, 1680], 4 [1320, 2580],
# 5 [2220, 3600]. Play 4's show is reorder-flagged and play 5's play order is unreliable: neither
# changes attribution, which reads windows and logged intervals only. Play 4's interval runs
# through a talk break (2,100 s to 2,400 s) in which nothing is logged.
RECORDS = _hour(
    HOUR,
    "canonical",
    (100, -200.0, HERMANOS, {}),
    (1, 120.0, MOLINA, {}),
    (2, 600.0, PRATT, {}),
    (3, 900.0, PRATT, {}),  # the same song twice in a row
    (4, 1500.0, CHUQUI, {"reorder_flag": True}),
    (5, 2400.0, ELLINGTON, {"play_order_status": "unreliable", "reorder_flag": None}),
)


def _found(track: tuple[str, str, str], at: float, **extra: Any) -> EvalIdentification:
    artist, song, album = track
    found: EvalIdentification = {
        "artist": artist, "song": song, "album": album, "label": "", "at": at, "source": "shazam",
    }  # fmt: skip
    return {**found, **extra}  # type: ignore[typeddict-item]


def _hit(track: tuple[str, str, str], at: float, **extra: Any) -> Emission:
    """A 12 s Shazam emission at grid offset ``at``, as ``read_results`` would return it."""
    address = ClipAddress(HOUR, int(at), 12)
    return Emission((address.key, SHAZAM), address, _found(track, at, **extra))


@pytest.mark.parametrize(
    ("track", "at", "play_id", "neighbor"),
    [
        pytest.param(HERMANOS, 30.0, 100, False, id="carryover song, inside its logged interval"),
        pytest.param(
            PRATT, 450.0, 2, True, id="early identification: correct before its logged start"
        ),
        pytest.param(CHUQUI, 1380.0, 4, True, id="late log in a reorder-flagged show"),
        pytest.param(
            PRATT, 750.0, 2, False, id="same song twice: the play whose interval holds at"
        ),
        pytest.param(PRATT, 990.0, 3, False, id="same song twice: the second play"),
        pytest.param(ELLINGTON, 2295.0, 5, True, id="play-order-unreliable show, in its pad"),
        pytest.param(UNLOGGED, 2205.0, None, False, id="unlogged play in a talk gap: no candidate"),
        pytest.param(MOLINA, 990.0, None, False, id="right song, outside every window it has"),
        pytest.param(
            ("Juana Molina", "la paradoja (feat. Jessica Pratt)", "DOGA (Remastered 2011 Version)"),
            195.0,
            1,
            False,
            id="featuring and edition clauses drop out under the join's rules",
        ),
        pytest.param(
            ("Jessica Pratt", "Back, Baby (Live)", "On Your Own Love Again"),
            750.0,
            None,
            False,
            id="a version the play does not name is wrong",
        ),
        pytest.param(("REM", "Drive", ""), 750.0, None, False, id="not this hour's song"),
    ],
)
def test_attribution(
    track: tuple[str, str, str], at: float, play_id: int | None, neighbor: bool
) -> None:
    [verdict] = attribute(plays_from(RECORDS), [_hit(track, at)])
    assert (verdict.play.play_id if verdict.play else None, verdict.neighbor) == (play_id, neighbor)


def test_a_same_song_play_in_neither_logged_interval_goes_to_the_earlier() -> None:
    records = _hour(HOUR, "canonical", (1, 600.0, PRATT, {}), (2, 900.0, MOLINA, {}), (3, 1200.0, PRATT, {}))  # fmt: skip
    # 1,050 s is in both Pratt plays' windows ([420, 1080] and [1020, 3600]), and in neither's interval.
    [verdict] = attribute(plays_from(records), [_hit(PRATT, 1050.0)])
    assert verdict.play is not None and (verdict.play.play_id, verdict.neighbor) == (1, True)


def test_logged_intervals_run_to_the_next_play_and_the_last_to_the_hour_end() -> None:
    plays = plays_from(list(reversed(RECORDS)))  # file order does not matter
    assert [(p.play_id, p.t_offset_s, p.logged_end_s) for p in plays] == [
        (100, -200.0, 120.0), (1, 120.0, 600.0), (2, 600.0, 900.0), (3, 900.0, 1500.0),
        (4, 1500.0, 2400.0), (5, 2400.0, 3600.0),
    ]  # fmt: skip


def test_read_plays_reads_utf8_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "plays.jsonl"
    lines = [json.dumps(r, ensure_ascii=False) for r in RECORDS]
    path.write_text("\n".join(lines) + "\n\n", encoding="utf-8")
    assert score.read_plays(path) == plays_from(RECORDS)


HITS = [
    _hit(HERMANOS, 30.0),
    _hit(MOLINA, 195.0),
    _hit(PRATT, 450.0, query_offset_s=0.0, ref_start_s=15.0),
    _hit(PRATT, 750.0, query_offset_s=0.0, ref_start_s=315.0),
    _hit(PRATT, 990.0),
    _hit(CHUQUI, 1380.0, query_offset_s=2.0, ref_start_s=32.0),
    _hit(UNLOGGED, 2205.0),
]


def test_precision_and_distinct_song_precision() -> None:
    verdicts = attribute(plays_from(RECORDS), list(reversed(HITS)))  # scored in at order
    assert score.precision(verdicts) == pytest.approx(6 / 7)
    # Runs: Hermanos, Molina, Pratt x3 (two plays, one song the loop would emit once), Chuquimamani,
    # the unlogged Stereolab.
    assert score.distinct_precision(verdicts) == pytest.approx(4 / 5)


def test_precision_of_nothing_is_none() -> None:
    assert (score.precision([]), score.distinct_precision([]), score.recall([])) == (
        None,
        None,
        None,
    )


def test_recall_time_to_first_identification_and_lag() -> None:
    plays = plays_from(RECORDS)
    rows = score_plays(plays, attribute(plays, HITS), covered=set(plays))
    by_id = {r.play.play_id: r for r in rows}
    assert 100 not in by_id  # a carryover is attributed to, never scored
    assert {i: r.identified for i, r in by_id.items()} == {
        1: True,
        2: True,
        3: True,
        4: True,
        5: False,
    }
    assert score.recall(rows) == pytest.approx(4 / 5)
    # Play 2 started at 435 s by both offset-bearing hits; it was first heard at 450 s, logged at 600 s.
    assert (by_id[2].ttfi_s, by_id[2].lag_s) == (15.0, 165.0)
    # Play 4 started at 1,380 + 2 - 32 = 1,350 s, and was logged at 1,500 s.
    assert (by_id[4].ttfi_s, by_id[4].lag_s) == (30.0, 150.0)
    # No hit on play 1 carries offsets (a Shazam answer with a null offset_s): no lag, no time.
    assert (by_id[1].ttfi_s, by_id[1].lag_s) == (None, None)


def test_recall_counts_covered_plays_only() -> None:
    plays = plays_from(RECORDS)
    covered = {p for p in plays if p.play_id in (1, 5)}
    rows = score_plays(plays, attribute(plays, HITS), covered=covered)
    assert score.recall(rows) == pytest.approx(1 / 2)
    assert {r.play.play_id: r.ttfi_s for r in rows}[2] is None  # timed only when covered


def _rows(era: str, lags: list[float]) -> list[score.PlayScore]:
    play = Play(HOUR, 1, 0.0, 600.0, 0.0, 780.0, "", "", "", era, False)
    return [score.PlayScore(play, True, True, None, lag) for lag in lags]


@pytest.mark.parametrize(
    ("lags", "pad"),
    [
        pytest.param([10.0] * 19, None, id="19 samples: insufficient data"),
        pytest.param(
            [float(n) for n in range(-10, 10)], 9.0, id="20 samples: nearest-rank p95 of |lag|"
        ),
        pytest.param([float(n) for n in range(1, 101)], 95.0, id="nearest rank covers 95 of 100"),
        pytest.param([700.0] * 25, 600.0, id="capped at 600 s"),
    ],
)
def test_pad_per_era(lags: list[float], pad: float | None) -> None:
    rows = _rows("canonical", lags) + _rows("etl", [5.0] * 3) + _rows("canonical", [])
    rows.append(score.PlayScore(rows[0].play, True, True, None, None))  # no offsets: not a sample
    assert score.pads(rows) == {"canonical": pad, "etl": None}


def _store_line(address: str, kind: str, track: tuple[str, str, str] = MOLINA, **extra: Any) -> str:
    artist, song, album = track
    record = {
        "address": address, "recognizer": SHAZAM, "recorded_at": "2026-10-06T20:00:00+00:00",
        "status": 200, "kind": kind, "artist": artist, "song": song, "album": album, "label": "",
        "offset_s": None, **extra,
    }  # fmt: skip
    return json.dumps(record, ensure_ascii=False) + "\n"


def test_coverage_and_scores_of_a_partial_leg(tmp_path: Path) -> None:
    short = _hour(SHORT, "etl", (7, 300.0, MOLINA, {}), (8, 1200.0, PRATT, {}))
    records = RECORDS + _hour(EMPTY, "canonical", (6, 60.0, CHUQUI, {})) + short
    addresses = grid(HOUR, 12) + grid(EMPTY, 12) + grid(SHORT, 12, hour_s=1800.0)
    lines = [
        _store_line(a.key, "no_match")
        for a in addresses
        if a.hour_key == SHORT or (a.hour_key == HOUR and a.offset_s not in (195, 2700))
    ]
    lines += [
        _store_line(f"{HOUR}#2700+12@128k", "server_error", status=503),  # in play 5's window only
        _store_line(f"{HOUR}#195+12@128k", "matched"),
        _store_line(f"{HOUR}#195+12@128k", "matched"),  # a repeat: counted once
        _store_line(f"{HOUR}#195+12@320k", "matched"),  # another leg
    ]
    path = tmp_path / "results.jsonl"
    path.write_text("".join(lines), encoding="utf-8")
    results = read_results([ResultStore(path)], {SHAZAM})

    leg = score.score_leg(
        plays_from(records), results, SHAZAM, LEG, [HOUR, EMPTY, SHORT], addresses
    )

    assert leg.coverage == score.Coverage(
        grid_addresses=240 + 240 + 120,
        scored_addresses=240 - 1 + 120,
        uncovered_addresses={"server_error": 1, "untried": 240},
        hours_without_records=[EMPTY],
        uncovered_plays={HOUR: 1, EMPTY: 1, SHORT: 1},
    )
    covered = {r.play.play_id for r in leg.plays if r.covered}
    # Play 5's window holds the error-only address; play 6's hour was never queried; play 8's
    # window ([980, 3600]) runs past the short hour's last grid address (1,785 s).
    assert covered == {1, 2, 3, 4, 7}
    assert [(v.emission.found["at"], v.play.play_id if v.play else None) for v in leg.verdicts] == [
        (195.0, 1)
    ]
    assert score.recall(leg.plays) == pytest.approx(1 / 5)  # the uncovered plays are not misses


def test_a_leg_with_no_grid_leaves_every_play_uncovered(tmp_path: Path) -> None:
    results = read_results([ResultStore(tmp_path / "none.jsonl")], {SHAZAM})
    leg = score.score_leg(plays_from(RECORDS), results, SHAZAM, LEG, [HOUR], [])
    assert leg.coverage.uncovered_plays == {HOUR: 5}
    assert (leg.coverage.hours_without_records, score.recall(leg.plays)) == ([HOUR], None)


def test_a_leg_scores_only_its_own_hours(tmp_path: Path) -> None:
    results = read_results([ResultStore(tmp_path / "none.jsonl")], {SHAZAM})
    leg = score.score_leg(plays_from(RECORDS), results, SHAZAM, LEG, [EMPTY], grid(EMPTY, 12))
    assert (leg.plays, leg.coverage.uncovered_plays) == ([], {})


@pytest.fixture
def outputs(fresh_recognizer):
    fresh_recognizer(WXDU_SHAZAM_SECRET="not-a-real-secret")
    return importlib.import_module("stream_sleuth.outputs")


def test_an_emissions_file_scores_as_the_same_emissions_in_memory(tmp_path: Path, outputs) -> None:
    path = tmp_path / "emissions.jsonl"
    out = outputs.JsonlOutput(path)
    for hit in HITS:
        out.emit({**hit.found, "address": hit.address.key})
    text = path.read_text(encoding="utf-8")
    assert "Gutiérrez" in text and '"emitted_at"' in text  # UTF-8, unescaped, stamped

    loaded = load_emissions(path, SHAZAM)

    assert loaded == HITS  # address and emitted_at are dropped from found
    plays = plays_from(RECORDS)
    assert attribute(plays, loaded) == attribute(plays, HITS)


def _emissions_file(tmp_path: Path, *records: dict) -> Path:
    path = tmp_path / "emissions.jsonl"
    lines = [json.dumps(r, ensure_ascii=False) for r in records]
    path.write_text("\n".join(lines[:1] + [""] + lines[1:]) + "\n", encoding="utf-8")
    return path


def _line(offset: float, /, **changes: Any) -> dict:
    """An emissions-file record at ``offset``; a change to None removes that field."""
    record = {**_found(MOLINA, offset), "address": f"{HOUR}#{int(offset)}+12@128k", **changes}
    return {k: v for k, v in record.items() if v is not None}


@pytest.mark.parametrize(
    ("changes", "problem"),
    [
        ({"at": None}, r"no at\b"),
        ({"source": None}, r"no source\b"),
        ({"address": None}, r"no address\b"),
        ({"address": None, "at": None}, r"no address or at\b"),
        ({"address": f"{HOUR}#200+12@128k"}, "not on the 15 s grid"),
        ({"address": f"{HOUR}#195+12"}, "not a clip address"),
        ({"at": 180.0}, "not its address's offset"),
    ],
)
def test_a_record_without_a_valid_address_at_or_source_raises_naming_its_line(
    tmp_path: Path, changes: dict, problem: str
) -> None:
    path = _emissions_file(tmp_path, _line(150.0), _line(195.0, **changes))
    with pytest.raises(ValueError, match=rf"emissions\.jsonl:3: .*{problem}"):
        load_emissions(path, SHAZAM)


def test_an_emissions_file_spans_hours(tmp_path: Path) -> None:
    path = _emissions_file(tmp_path, _line(150.0), _line(195.0, address=f"{EMPTY}#195+12@128k"))
    assert [e.address.hour_key for e in load_emissions(path, SHAZAM)] == [HOUR, EMPTY]
