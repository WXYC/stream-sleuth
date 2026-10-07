"""``evaluation.queues``: the near-miss and flowsheet-false-positive queues' run selection.

Every play and emission is synthetic. The functions are called directly; the CLI tests that write
and read the queue files through ``python -m evaluation.score`` are in ``test_score_cli.py``.
"""

from __future__ import annotations

from typing import Any

import pytest

from evaluation import queues
from evaluation.clips import ClipAddress
from evaluation.results import Emission
from evaluation.score import plays_from
from tests.emissions import OLAF, SHAZAM, emission, leg_scores
from tests.plays import (
    COCREDIT,
    HOUR,
    MOLINA,
    OTHER,
    PRATT,
    RECORDS,
    STAGE,
    UNLOGGED,
    hour_records,
)


def near_misses(
    emissions: list[Emission],
    records: list[dict[str, Any]] = RECORDS,
    references: dict[str, tuple[str, ...]] | None = None,
) -> list[dict[str, Any]]:
    plays = plays_from(records)
    legs = leg_scores(plays, {"12s/shazam": emissions}, references)
    return [row for row, _ in queues.near_miss_runs(plays, legs, references)]


def one_play(artist: str, title: str, *more: tuple[int, float, str, str]) -> list[dict[str, Any]]:
    """Play 1 at 120 s, and any ``more`` as (play id, offset, artist, title)."""
    rows: list[Any] = [(1, 120.0, (artist, title, ""), {})]
    rows += [(i, t, (a, ti, ""), {}) for i, t, a, ti in more]
    return hour_records(HOUR, "canonical", *rows)


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
    rows = near_misses([emission(track, at)])

    assert [r["matched_on"] for r in rows] == ([matched_on] if matched_on else [])


def test_a_similar_token_set_is_a_near_miss_when_neither_field_matches() -> None:
    records = one_play("Chuquimamani-Condori", "Call Your Name Tonight Now Forever")
    near = ("Chuquimamani Condori Jr", "Call Your Name Tonight Now Forever Again", "Edits")
    far = ("Chuquimamani Condori Jr", "Call Your Name Tonight Now Forever Again Please", "Edits")

    [hit] = near_misses([emission(near, 195.0)], records)
    assert hit["matched_on"] == "similarity"
    assert not near_misses([emission(far, 195.0)], records)


def test_two_empty_artists_do_not_match() -> None:
    assert not near_misses([emission(("", "Otra Cancion", ""), 195.0)], one_play("", "la paradoja"))


def test_a_run_is_judged_by_every_emission_and_the_nearest_play_wins() -> None:
    """The run starts outside every window of its song's play and enters one later."""
    chuqui = ("Chuquimamani-Condori", "Otra Cancion", "Edits")

    [row] = near_misses([emission(chuqui, 1200.0), emission(chuqui, 1335.0)])

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

    [row] = near_misses([emission(("Jessica Pratt", "Back, Baby", ""), 195.0)], records)

    assert (row["play_id"], row["matched_on"]) == (2, "artist")


def test_an_artist_credit_does_not_swallow_the_title_in_the_similarity() -> None:
    """Each field is keyed alone: an inline credit in the artist never reaches the title."""
    records = one_play(
        "Carré feat. Zeta",
        "Totally Other Thing",
        (2, 200.0, "Carré", "Hibiscus Pt. 2 (Edit)"),
    )

    [row] = near_misses([emission(("Carré feat. Bbyafricka", "Hibiscus Pt 2", ""), 195.0)], records)

    assert (row["play_id"], row["matched_on"]) == (2, "artist")


def test_a_near_miss_under_olaf_references_names_every_artist_tag_of_its_file() -> None:
    olaf = emission(("Mislabeled Tag", "Otra Cancion", "DOGA"), 195.0, source="local")

    [row] = near_misses([olaf], references={STAGE: ("Juana Molina",)})

    assert (row["play_id"], row["matched_on"]) == (1, "artist")
    assert not near_misses([olaf])


ARTISTS = "Chuquimamani Condori Jessica Pratt Hermanos Gutierrez Duo"  # seven tokens


def test_the_similarity_counts_the_artist_tokens_and_not_the_title_alone() -> None:
    """Titles alone overlap at 2/3; with the artists the token sets overlap at 9/11."""
    records = one_play(ARTISTS, "Call Name")

    [row] = near_misses([emission((f"{ARTISTS} Jr", "Call Name Again", ""), 195.0)], records)

    assert row["matched_on"] == "similarity"


def test_the_similarity_of_an_olaf_match_is_taken_over_every_artist_tag_of_its_file() -> None:
    records = one_play(ARTISTS, "Call Name")
    olaf = emission(("Mislabeled Tag", "Call Name Again", ""), 195.0, source="local")

    [row] = near_misses([olaf], records, {STAGE: (f"{ARTISTS} Jr",)})

    assert (row["matched_on"], row["reference_artists"]) == ("similarity", f"{ARTISTS} Jr")
    assert not near_misses([olaf], records)


def test_more_shared_fields_win_over_more_similar_tokens() -> None:
    """The first play shares no field but is the more similar; the second shares the artist."""
    records = one_play(ARTISTS, "Call Name", (2, 200.0, f"{ARTISTS} Jr", "Something Else"))

    [row] = near_misses([emission((f"{ARTISTS} Jr", "Call Name Again", ""), 195.0)], records)

    assert (row["play_id"], row["matched_on"]) == (2, "artist")


def test_a_play_that_shares_both_fields_but_not_its_album_version_names_both() -> None:
    records = hour_records(
        HOUR, "canonical", (1, 120.0, ("Juana Molina", "la paradoja", "DOGA (Live)"), {})
    )

    [row] = near_misses([emission(MOLINA, 195.0)], records)

    assert row["matched_on"] == "artist+title"


def test_a_row_names_the_run_the_play_and_what_an_olaf_match_was_made_on() -> None:
    olaf = emission(("Mislabeled Tag", "Otra Cancion", "DOGA"), 195.0, source="local")

    [row] = near_misses([olaf], references={STAGE: ("Juana Molina", "Duo Tag")})

    assert (row["recognizer"], row["reference_artists"], row["play_carryover"]) == (
        OLAF,
        "Juana Molina | Duo Tag",
        False,
    )


def test_a_shazam_row_has_no_reference_artists_and_a_carryover_play_says_so() -> None:
    [row] = near_misses([emission(("Hermanos Gutiérrez", "Otra Cancion", ""), 30.0)])

    assert (row["recognizer"], row["play_id"], row["play_carryover"], row["reference_artists"]) == (
        SHAZAM,
        100,
        True,
        "",
    )


def test_consecutive_wrong_emissions_of_one_song_are_one_row_with_archive_times() -> None:
    other = ("Juana Molina", "Otra Cancion", "DOGA")
    hits = [emission(other, at) for at in (195.0, 210.0, 225.0)]
    hits += [emission(MOLINA, 240.0), emission(other, 255.0)]

    rows = near_misses(hits)

    assert [(r["address"], r["last_address"], r["emissions"]) for r in rows] == [
        (f"{HOUR}#195+12@128k", f"{HOUR}#225+12@128k", 3),
        (f"{HOUR}#255+12@128k", f"{HOUR}#255+12@128k", 1),
    ]
    assert (rows[0]["hour_key"], rows[0]["start"], rows[0]["end"]) == (HOUR, "0:03:15", "0:03:57")
    assert rows[0]["verdict"] == ""


LIKELY = "likely-unlogged-correct"


def false_positives(
    shazam: list[Emission],
    olaf: list[Emission] | None = None,
    references: dict[str, tuple[str, ...]] | None = None,
) -> list[dict[str, Any]]:
    """The false-positive queue for one 12 s leg, Shazam and (when given) Olaf, against RECORDS."""
    plays = plays_from(RECORDS)
    answers = {"12s/shazam": shazam} | ({} if olaf is None else {"12s/olaf": olaf})
    legs = leg_scores(plays, answers, references)
    return [row for row, _ in queues.false_positive_runs(plays, legs, references)]


def unlogged(at: float, ref_start_s: float | None = None, hour: str = HOUR) -> Emission:
    """A Shazam answer for a song no play names; ``ref_start_s`` is its match offset, if any."""
    query_offset_s = None if ref_start_s is None else 0.0
    return emission(UNLOGGED, at, hour=hour, query_offset_s=query_offset_s, ref_start_s=ref_start_s)


def test_a_run_of_wrong_emissions_is_one_row_with_its_first_and_last_position() -> None:
    hits = [unlogged(at) for at in (750.0, 765.0, 780.0)] + [
        emission(PRATT, 795.0),
        unlogged(810.0),
    ]

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

    rows = false_positives([emission(near, 195.0), unlogged(750.0)])

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
    rows = false_positives(
        [unlogged(750.0), unlogged(765.0)], [emission(track, at, hour=hour, source="local")]
    )

    assert [r["preflag"] for r in rows if r["recognizer"] == SHAZAM] == [preflag]


def test_each_recognizers_run_is_flagged_by_the_other() -> None:
    rows = false_positives([unlogged(750.0)], [emission(UNLOGGED, 750.0, source="local")])

    assert {r["leg"]: r["preflag"] for r in rows} == {"12s/shazam": LIKELY, "12s/olaf": LIKELY}


def test_a_shazam_answer_naming_the_album_artist_agrees_with_an_olaf_reference_tag() -> None:
    credited = ("Stereolab & Duo Tag", UNLOGGED[1], UNLOGGED[2])
    references: dict[str, tuple[str, ...]] = {STAGE: ("Stereolab & Duo Tag", "Stereolab")}

    flagged = false_positives(
        [unlogged(750.0)], [emission(credited, 750.0, source="local")], references
    )
    bare = false_positives([unlogged(750.0)], [emission(credited, 750.0, source="local")])

    assert [r["preflag"] for r in flagged if r["recognizer"] == SHAZAM] == [LIKELY]
    assert [r["preflag"] for r in bare if r["recognizer"] == SHAZAM] == [""]
    assert [r["reference_artists"] for r in flagged if r["recognizer"] == OLAF] == [
        "Stereolab & Duo Tag | Stereolab"
    ]


def test_a_run_is_not_flagged_by_a_different_capture_length() -> None:
    plays = plays_from(RECORDS)
    answers = {
        "12s/shazam": [unlogged(750.0)],
        "6s/olaf": [emission(UNLOGGED, 750.0, source="local", length_s=6)],
    }
    legs = leg_scores(plays, answers)

    rows = [row for row, _ in queues.false_positive_runs(plays, legs, None)]

    assert {r["leg"]: r["preflag"] for r in rows} == {"12s/shazam": "", "6s/olaf": ""}


def olaf_run(*clips: tuple[float, float, float]) -> list[Emission]:
    """Olaf answers for one song at (grid offset, query_offset_s, ref_start_s) each."""
    return [
        emission(UNLOGGED, at, source="local", query_offset_s=q, ref_start_s=r)
        for at, q, r in clips
    ]


# As ``run_olaf`` stores them for consecutive grid clips of one playback that started at 650 s: a
# clip's matched query window starts ``query_offset_s`` into it, and ``ref_start_s`` is where in
# the reference that window began, so the estimate is ``at + query_offset_s - ref_start_s``.
OLAF_PLAYBACK = [(750.0, 0.5, 100.5), (765.0, 11.0, 126.0), (780.0, 2.25, 132.25)]
OLAF_STEADY = [
    pytest.param(OLAF_PLAYBACK, LIKELY, id="one playback, estimate 650 throughout"),
    pytest.param(
        [(750.0, 0.5, 100.5), (765.0, 11.0, 126.0 - 7.5)], LIKELY, id="7.5 s off, at the tolerance"
    ),
    pytest.param(
        [(750.0, 0.5, 100.5), (765.0, 11.0, 126.0 - 7.6)], "", id="7.6 s off, past the tolerance"
    ),
    pytest.param(
        [(750.0, 0.5, 100.5), (765.0, 11.0, 115.5)], "", id="a reference position standing still"
    ),
    pytest.param([(750.0, 0.5, 100.5)], "", id="a single emission"),
]


@pytest.mark.parametrize(("clips", "preflag"), OLAF_STEADY)
def test_an_olaf_playbacks_song_start_is_steady_only_with_its_query_offset(
    clips: list[tuple[float, float, float]], preflag: str
) -> None:
    """Dropping the query offset or flipping its sign moves this playback's estimates 10.5 s and
    21 s apart, so the run would stop being preflagged."""
    [row] = false_positives([], olaf_run(*clips))

    assert row["preflag"] == preflag


def test_an_olaf_run_is_flagged_through_its_reference_tags_by_a_shazam_album_artist() -> None:
    credited = ("Stereolab & Duo Tag", UNLOGGED[1], UNLOGGED[2])
    references: dict[str, tuple[str, ...]] = {STAGE: ("Stereolab & Duo Tag", "Stereolab")}

    flagged = false_positives(
        [unlogged(750.0)], [emission(credited, 750.0, source="local")], references
    )

    assert {r["recognizer"]: r["preflag"] for r in flagged} == {SHAZAM: LIKELY, OLAF: LIKELY}


def test_the_span_follows_the_capture_length_of_the_runs_last_emission() -> None:
    """A 20 s clip runs past the next 15 s grid offset, so an agreeing answer there counts; at 12 s
    the span ends before it. The span's end never lands on a grid offset (6, 12, 20 s from a
    multiple of 15), so its exclusive edge is not reachable."""

    def flagged(length_s: int) -> str:
        address = ClipAddress(HOUR, 750, length_s)
        answer = Emission((address.key, SHAZAM), address, unlogged(750.0).found)
        rows = false_positives([answer], [emission(UNLOGGED, 765.0, source="local")])
        return next(r["preflag"] for r in rows if r["recognizer"] == SHAZAM)

    assert (flagged(20), flagged(12)) == (LIKELY, "")
