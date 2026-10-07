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
from evaluation.results import Emission, Leg, Results, ResultStore, read_results
from evaluation.score import attribute, load_emissions, plays_from, score_plays
from stream_sleuth.recognizers.base import EvalIdentification
from stream_sleuth.recognizers.olaf import recognizer_identity as olaf_identity
from tests.stores import olaf_record, shazam_record

HOUR = "2026/08/12/202608121600.mp3"
EMPTY = "2026/08/12/202608121700.mp3"  # gridded, never queried
SHORT = "2026/08/12/202608121800.mp3"  # decodes to 1,800 s
OTHER = "2026/08/12/202608121900.mp3"  # outside the leg's hours
SHAZAM = "shazam@0.8.1, segment=12"
OLAF = olaf_identity("rotation")
LEG = Leg("12s", 12, "128k", "all", ("shazam", "olaf"))
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
                "in_pool": True,
                "pool_match_tier": "exact",
                "pool_format": "mp3",
                "rotation": True,
                "talk_rows": 0,
                "group": "canonical-high",
                "band": "daytime",
                "subset": False,
                **extra,
            }  # fmt: skip
        )
    return records


# Windows: carryover [0, 300], 1 [0, 780], 2 [420, 1080], 3 [720, 1680], 4 [1320, 2580],
# 5 [2220, 3600]. Play 4's show is reorder-flagged and play 5's play order is unreliable: neither
# changes attribution, which reads windows and logged intervals only. Play 4's interval runs
# through a talk break (2,100 s to 2,400 s) in which nothing is logged. Play 3 is out of the pool.
RECORDS = _hour(
    HOUR,
    "canonical",
    (100, -200.0, HERMANOS, {}),
    (1, 120.0, MOLINA, {}),
    (2, 600.0, PRATT, {}),
    (3, 900.0, PRATT, {"in_pool": False, "pool_match_tier": None}),  # the same song twice
    (4, 1500.0, CHUQUI, {"reorder_flag": True}),
    (5, 2400.0, ELLINGTON, {"play_order_status": "unreliable", "reorder_flag": None}),
)


def _found(track: tuple[str, str, str], at: float, **extra: Any) -> EvalIdentification:
    artist, song, album = track
    found: EvalIdentification = {
        "artist": artist, "song": song, "album": album, "label": "", "at": at, "source": "shazam",
    }  # fmt: skip
    return {**found, **extra}  # type: ignore[typeddict-item]


def _hit(track: tuple[str, str, str], at: float, hour: str = HOUR, **extra: Any) -> Emission:
    """A 12 s Shazam emission at grid offset ``at``, as ``read_results`` would return it."""
    address = ClipAddress(hour, int(at), 12)
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
        pytest.param(("Stereolab", "Drive", ""), 750.0, None, False, id="not this hour's song"),
        pytest.param(PRATT, 420.0, 2, True, id="at the window's start, which is inside it"),
        pytest.param(CHUQUI, 1320.0, 4, True, id="at the other window's start"),
        pytest.param(CHUQUI, 2580.0, 4, True, id="at the window's end, which is inside it"),
        pytest.param(CHUQUI, 1500.0, 4, False, id="at the logged start, inside the interval"),
        pytest.param(CHUQUI, 2400.0, 4, True, id="at the logged end, outside the interval"),
    ],
)
def test_attribution(
    track: tuple[str, str, str], at: float, play_id: int | None, neighbor: bool
) -> None:
    [verdict] = attribute(plays_from(RECORDS), [_hit(track, at)])
    assert (verdict.play.play_id if verdict.play else None, verdict.neighbor) == (play_id, neighbor)


def test_a_dotted_initialism_joins_the_same_letters_undotted() -> None:
    """The join's rule since WXYC/stream-sleuth#96, so "A.R. Kane" logged is "AR Kane" heard."""
    plays = plays_from(_hour(HOUR, "canonical", (1, 120.0, ("A.R. Kane", "Baby Milk Snatcher", "69"), {})))  # fmt: skip
    [verdict] = attribute(plays, [_hit(("AR Kane", "Baby Milk Snatcher", "69"), 195.0)])
    assert verdict.play is not None and verdict.play.play_id == 1


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


@pytest.mark.parametrize(
    ("hits", "plain", "distinct"),
    [
        pytest.param(
            [_hit(MOLINA, 195.0), _hit(("JUANA MOLINA", "La Paradoja", "DOGA"), 210.0), _hit(UNLOGGED, 2205.0)],
            2 / 3, 1 / 2, id="the loop's key: artist and song, case-folded with lower()",
        ),
        pytest.param(
            [_hit(MOLINA, 195.0), _hit(MOLINA, 990.0)], 1 / 2, 1.0,
            id="a run is correct when any of it is, though one emission is past the window",
        ),
        pytest.param(
            [_hit(CHUQUI, 1305.0), _hit(CHUQUI, 1320.0)], 1 / 2, 1.0,
            id="a run whose first emission is before the window is still correct",
        ),
        pytest.param(
            [_hit(ELLINGTON, 3585.0), _hit(ELLINGTON, 0.0, hour=EMPTY)], 1 / 2, 1 / 2,
            id="a run ends at the hour's end",
        ),
        pytest.param(
            [_hit(PRATT, 450.0), _hit(UNLOGGED, 2205.0), _hit(PRATT, 750.0)], 2 / 3, 1 / 2,
            id="runs are read in at order, not input order",
        ),
    ],
)  # fmt: skip
def test_distinct_song_runs(hits: list[Emission], plain: float, distinct: float) -> None:
    verdicts = attribute(plays_from(RECORDS), hits)
    assert (score.precision(verdicts), score.distinct_precision(verdicts)) == pytest.approx(
        (plain, distinct)
    )


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
    # Play 2 started at 435 s by its earliest offset-bearing hit; first heard at 450 s, logged at 600 s.
    assert (by_id[2].first_s, by_id[2].ttfi_s, by_id[2].lag_s) == (450.0, 15.0, 165.0)
    # Play 4 started at 1,380 + 2 - 32 = 1,350 s, and was logged at 1,500 s.
    assert (by_id[4].first_s, by_id[4].ttfi_s, by_id[4].lag_s) == (1380.0, 30.0, 150.0)
    # No hit on play 1 carries offsets (a Shazam answer with a null offset_s): no lag, no time.
    assert (by_id[1].first_s, by_id[1].ttfi_s, by_id[1].lag_s) == (195.0, None, None)


@pytest.mark.parametrize(
    ("hits", "first", "ttfi", "lag"),
    [
        pytest.param(
            [_hit(MOLINA, 150.0, query_offset_s=0.0, ref_start_s=30.0),
             _hit(MOLINA, 300.0, query_offset_s=0.0, ref_start_s=60.0),
             _hit(MOLINA, 450.0, query_offset_s=0.0, ref_start_s=90.0)],
            150.0, 30.0, 80.0, id="the earliest clip with offsets sets the start, not the median",
        ),
        pytest.param(
            [_hit(MOLINA, 300.0, query_offset_s=0.0, ref_start_s=180.0), _hit(MOLINA, 135.0)],
            135.0, 15.0, 80.0, id="an earlier offset-less answer is the first identification only",
        ),
        pytest.param(
            [_hit(MOLINA, 150.0, query_offset_s=10.0, ref_start_s=0.0)], 150.0, 0.0, 40.0,
            id="a song that starts inside the first clip is identified at once, never before",
        ),
    ],
)  # fmt: skip
def test_the_song_start_is_the_earliest_offset_bearing_clips(
    hits: list[Emission], first: float, ttfi: float, lag: float
) -> None:
    plays = plays_from(_hour(HOUR, "canonical", (1, 200.0, MOLINA, {})))
    verdicts = list(reversed(attribute(plays, hits)))  # score_plays orders them itself
    [row] = score_plays(plays, verdicts, covered=set(plays))
    assert (row.first_s, row.ttfi_s, row.lag_s) == (first, ttfi, lag)


def test_recall_counts_covered_plays_only() -> None:
    plays = plays_from(RECORDS)
    covered = {p for p in plays if p.play_id in (1, 5)}
    rows = score_plays(plays, attribute(plays, HITS), covered=covered)
    assert score.recall(rows) == pytest.approx(1 / 2)
    play_2 = {r.play.play_id: r for r in rows}[2]  # it has a timed hit, but is uncovered
    assert (play_2.identified, play_2.ttfi_s, play_2.lag_s) == (True, None, None)


def _rows(era: str, lags: list[float | None]) -> list[score.PlayScore]:
    [play] = plays_from(_hour(HOUR, era, (1, 0.0, MOLINA, {})))
    return [score.PlayScore(play, True, True, None, None, lag) for lag in lags]


@pytest.mark.parametrize(
    ("lags", "recommended", "p95"),
    [
        pytest.param([10.0] * 19, 180.0, None, id="19 samples: insufficient data, 180 s"),
        pytest.param(
            [float(n) for n in range(-10, 10)], 180.0, 9.0,
            id="20 samples under the default: 180 s stands",
        ),
        pytest.param(
            [3.0 * n for n in range(1, 101)], 285.0, 285.0,
            id="over the default: the pad covers p95 of |lag|",
        ),
        pytest.param(
            [float(n) for n in range(190, 211)], 209.0, 209.0,
            id="nearest rank: the 20th of 21, not the 19th",
        ),
        pytest.param([700.0] * 25, 600.0, 700.0, id="capped at 600 s"),
    ],
)  # fmt: skip
def test_pad_per_era(lags: list[float], recommended: float, p95: float | None) -> None:
    rows = _rows("canonical", [*lags, None]) + _rows("etl", [200.0] * 20)  # None: no offsets
    assert score.pads(rows) == {
        "canonical": score.Pad(recommended, len(lags), p95, (180.0,)),
        # Against the plan's 180 s default, not the 220 s this era's windows were padded with.
        "etl": score.Pad(200.0, 20, 200.0, (220.0,)),
    }


def _no_matches(
    store: ResultStore, hour: str, skip: tuple[int, ...] = (), **grid_args: Any
) -> None:
    for a in grid(hour, 12, **grid_args):
        if a.offset_s not in skip:
            shazam_record(store, a.key, "no_match")


def test_coverage_and_scores_of_a_partial_leg(tmp_path: Path) -> None:
    short = _hour(SHORT, "etl", (7, 300.0, MOLINA, {}), (8, 1200.0, PRATT, {}))
    records = RECORDS + _hour(EMPTY, "canonical", (6, 60.0, CHUQUI, {})) + short
    addresses = grid(HOUR, 12) + grid(EMPTY, 12) + grid(SHORT, 12, hour_s=1800.0)
    # SHORT's records run to 3,585 s, but past 1,785 s they are not in its decoded grid.
    store = ResultStore(tmp_path / "results.jsonl")
    _no_matches(store, SHORT, hour_s=1800.0)
    _no_matches(store, HOUR, skip=(195, 2700))
    shazam_record(store, f"{HOUR}#2700+12@128k", "server_error")  # in play 5's window only
    shazam_record(store, f"{HOUR}#195+12@128k")
    shazam_record(store, f"{HOUR}#195+12@128k")  # a repeat: counted once
    shazam_record(store, f"{HOUR}#195+12@320k")  # another leg
    olaf_record(
        store, f"{HOUR}#195+12@128k", "matched", CHUQUI, identity=OLAF
    )  # another recognizer
    shazam_record(store, f"{OTHER}#195+12@128k")  # an hour outside the leg
    results = read_results([store], {SHAZAM, OLAF})

    leg = score.score_leg(
        plays_from(records + _hour(OTHER, "canonical", (9, 60.0, MOLINA, {}))),
        results,
        SHAZAM,
        LEG,
        [HOUR, EMPTY, SHORT],
        addresses,
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
    # The uncovered plays are not misses, and play 3 is out of the pool.
    assert (leg.recall, leg.in_pool_recall) == pytest.approx((1 / 5, 1 / 4))


@pytest.mark.parametrize(
    ("offset", "uncovered"),
    [(2205, {4}), (2220, {4, 5}), (2580, {4, 5}), (2595, {5})],
)
def test_an_address_on_a_window_edge_gates_that_play(
    tmp_path: Path, offset: int, uncovered: set[int]
) -> None:
    store = ResultStore(tmp_path / "results.jsonl")
    _no_matches(store, HOUR, skip=(offset,))
    results = read_results([store], {SHAZAM})
    leg = score.score_leg(plays_from(RECORDS), results, SHAZAM, LEG, [HOUR], grid(HOUR, 12))
    assert {r.play.play_id for r in leg.plays if not r.covered} == uncovered


LABELS = "2026/08/12/202608122000.mp3"


def test_plays_no_emission_can_join_stay_in_recall_and_are_counted(tmp_path: Path) -> None:
    records = _hour(
        LABELS,
        "canonical",
        (1, 120.0, ("", "Untitled", ""), {"in_pool": None, "pool_match_tier": None}),
        (2, 600.0, ("Jessica Pratt", "", "On Your Own Love Again"), {}),  # in the pool by album
        (3, 1200.0, MOLINA, {}),
        (4, 2400.0, ("", "", ""), {"in_pool": None, "pool_match_tier": None}),  # uncovered
    )
    store = ResultStore(tmp_path / "results.jsonl")
    _no_matches(store, LABELS, skip=(1215, 3000))
    shazam_record(store, f"{LABELS}#1215+12@128k")
    results = read_results([store], {SHAZAM})
    leg = score.score_leg(plays_from(records), results, SHAZAM, LEG, [LABELS], grid(LABELS, 12))
    # Play 1 (no artist, in_pool null) is never in the in-pool denominator; plays 1 and 2 have
    # no title key, so title_tier can never join them: misses all the same, and counted. Play 4
    # has none either, but is uncovered, so it is neither a miss nor counted.
    assert (leg.recall, leg.in_pool_recall, leg.unjoinable_plays) == pytest.approx(
        (1 / 3, 1 / 2, 2)
    )


def test_an_hour_whose_records_are_all_errors_is_uncovered_but_has_records(tmp_path: Path) -> None:
    errors = "2026/08/12/202608122100.mp3"
    store = ResultStore(tmp_path / "results.jsonl")
    for offset in (0, 15):
        shazam_record(store, f"{errors}#{offset}+12@128k", "server_error")
    results = read_results([store], {SHAZAM})
    plays = plays_from(_hour(errors, "canonical", (1, 120.0, MOLINA, {})))
    leg = score.score_leg(plays, results, SHAZAM, LEG, [errors], grid(errors, 12))
    assert leg.coverage == score.Coverage(
        grid_addresses=240,
        scored_addresses=0,
        uncovered_addresses={"server_error": 2, "untried": 238},
        hours_without_records=[],
        uncovered_plays={errors: 1},
    )


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


def _emissions_file(tmp_path: Path, *records: dict | str) -> Path:
    """An emissions file of ``records`` after a blank line; a str is written as the line itself."""
    path = tmp_path / "emissions.jsonl"
    lines = [r if isinstance(r, str) else json.dumps(r, ensure_ascii=False) for r in records]
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
        ({"at": False}, "at False is not a number"),
        ({"address": 195}, "address 195 is not a string"),
        ({"artist": None}, r"no artist\b"),
        ('{"address": "' + HOUR, "not JSON"),  # a line torn mid-append
        ("null", "not a JSON object"),
        ("[1, 2]", "not a JSON object"),
    ],
)
def test_a_malformed_record_raises_naming_its_line(
    tmp_path: Path, changes: dict | str, problem: str
) -> None:
    second = changes if isinstance(changes, str) else _line(195.0, **changes)
    path = _emissions_file(tmp_path, _line(150.0), second)
    with pytest.raises(ValueError, match=rf"emissions\.jsonl:3: .*{problem}"):
        load_emissions(path, SHAZAM)


def test_an_emissions_file_spans_hours_and_capture_lengths(tmp_path: Path) -> None:
    """A replay of the 6 s / 12 s cadence mixes capture lengths in one file; ``attribute`` keeps
    every emission, since only ``score_leg`` (the grid path) selects a capture length."""
    six = _line(150.0, address=f"{HOUR}#150+6@128k")
    path = _emissions_file(tmp_path, six, _line(195.0, address=f"{EMPTY}#195+12@128k"))
    loaded = load_emissions(path, SHAZAM)
    assert [str(e.address) for e in loaded] == [f"{HOUR}#150+6@128k", f"{EMPTY}#195+12@128k"]
    assert len(attribute(plays_from(RECORDS), loaded)) == 2


COCREDIT = "Juana Molina & Chuquimamani-Condori"
STAGE = "a" * 40
REFERENCES = {STAGE: (COCREDIT, "Juana Molina")}


def _local(artist: str, at: float, ref_key: str = STAGE, **extra: Any) -> Emission:
    """A 12 s Olaf emission as ``read_results`` would return it: it names its reference by stage id."""
    track = (artist, MOLINA[1], MOLINA[2])
    found = _found(track, at, source="local", confidence=40.0, ref_key=ref_key, **extra)
    address = ClipAddress(HOUR, int(at), 12)
    return Emission((address.key, OLAF), address, found)


def _molina_plays(artist: str = "Juana Molina") -> list[score.Play]:
    return plays_from(_hour(HOUR, "canonical", (1, 120.0, (artist, MOLINA[1], MOLINA[2]), {})))


@pytest.mark.parametrize(
    ("play_artist", "ref_key", "references", "scored"),
    [
        pytest.param("Juana Molina", STAGE, REFERENCES, True, id="joins through the album artist"),
        pytest.param(COCREDIT, STAGE, REFERENCES, True, id="joins through the artist"),
        pytest.param("Jessica Pratt", STAGE, REFERENCES, False, id="another artist is still wrong"),
        pytest.param("Juana Molina", STAGE, None, False, id="no names mapping: as before"),
        pytest.param("Juana Molina", "b" * 40, REFERENCES, False, id="a ref_key not in it"),
        pytest.param(COCREDIT, STAGE, None, True, id="no mapping: the emission's own artist joins"),
        pytest.param(
            COCREDIT, "b" * 40, REFERENCES, True, id="a ref_key not in it: own artist joins"
        ),
        pytest.param(
            COCREDIT,
            STAGE,
            {STAGE: ("Jessica Pratt",)},
            True,
            id="a mismatched mapping never loses the emission's own artist",
        ),  # fmt: skip
        pytest.param(
            "Jessica Pratt", STAGE, {STAGE: ("Jessica Pratt",)}, True, id="the mapping's name joins"
        ),  # fmt: skip
    ],
)
def test_an_olaf_match_on_a_co_credited_file_is_correct_by_either_artist_tag(
    play_artist: str,
    ref_key: str,
    references: dict[str, tuple[str, ...]] | None,
    scored: bool,
) -> None:
    emission = _local(COCREDIT, 195.0, ref_key)
    [verdict] = attribute(_molina_plays(play_artist), [emission], references)
    assert (verdict.play is not None) is scored


def test_a_shazam_co_credit_stays_wrong_whatever_the_names_mapping_holds() -> None:
    """Shazam names one artist string and no reference: its co-credit is a near miss (#93)."""
    shazam = _hit((COCREDIT, MOLINA[1], MOLINA[2]), 195.0, ref_key=STAGE)
    [verdict] = attribute(_molina_plays(), [shazam], REFERENCES)
    assert verdict.play is None


def test_a_shazam_emission_still_joins_by_its_own_artist_whatever_its_ref_key_maps_to() -> None:
    shazam = _hit((COCREDIT, MOLINA[1], MOLINA[2]), 195.0, ref_key=STAGE)
    [verdict] = attribute(_molina_plays(COCREDIT), [shazam], {STAGE: ("Jessica Pratt",)})
    assert verdict.play is not None and verdict.play.play_id == 1


@pytest.mark.parametrize(("references", "play_id"), [(REFERENCES, 1), (None, None)])
def test_a_leg_scores_an_olaf_co_credit_with_the_names_mapping(
    references: dict[str, tuple[str, ...]] | None, play_id: int | None
) -> None:
    emission = _local(COCREDIT, 195.0)
    results = Results([emission], {emission.key}, {})

    leg = score.score_leg(
        _molina_plays(),
        results,
        OLAF,
        LEG,
        [HOUR],
        grid(HOUR, 12),
        references,  # fmt: skip
    )

    assert [v.play.play_id if v.play else None for v in leg.verdicts] == [play_id]
