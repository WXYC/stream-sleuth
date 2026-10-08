"""``tests.emissions`` and ``tests.queues``: the in-memory fixtures hold what the writers write.

``emission`` must equal what ``results.to_identification`` makes of a record filed through
``tests.stores``, so a fixture cannot drift from the store format it stands in for.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from evaluation.results import ResultStore, to_identification
from evaluation.score import attribute, plays_from
from tests.emissions import emission, leg_scores
from tests.plays import HOUR, MOLINA, PRATT, RECORDS, STAGE
from tests.stores import olaf_record, shazam_record

ADDRESS = f"{HOUR}#195+12@128k"
ELSEWHERE = "2026/08/12/202608121900.mp3"


def _stored(tmp_path: Path) -> ResultStore:
    return ResultStore(tmp_path / "results.jsonl")


def _read(store: ResultStore):  # type: ignore[no-untyped-def]
    [record] = store.records()
    return to_identification(record)


@pytest.mark.parametrize(
    ("offsets", "ref_key"),
    [
        pytest.param({}, None, id="defaults"),
        pytest.param({"query_offset_s": 2.0, "ref_start_s": 32.0}, None, id="both offsets"),
        pytest.param({}, "b" * 40, id="another reference"),
    ],
)
def test_a_local_emission_is_what_an_olaf_record_reads_back_as(
    tmp_path: Path, offsets: dict[str, Any], ref_key: str | None
) -> None:
    store = _stored(tmp_path)
    olaf_record(store, ADDRESS, "matched", PRATT, ref_key=ref_key or STAGE, **offsets)

    assert emission(PRATT, 195.0, source="local", ref_key=ref_key, **offsets) == _read(store)


@pytest.mark.parametrize(
    "offset_s", [pytest.param(None, id="no stored offset"), pytest.param(15.0, id="an offset")]
)
def test_a_shazam_emission_is_what_a_shazam_record_reads_back_as(
    tmp_path: Path, offset_s: float | None
) -> None:
    store = _stored(tmp_path)
    shazam_record(store, ADDRESS, "matched", MOLINA, offset_s=offset_s)
    offsets: dict[str, Any] = (
        {} if offset_s is None else {"query_offset_s": 0.0, "ref_start_s": offset_s}
    )

    assert emission(MOLINA, 195.0, **offsets) == _read(store)


def test_the_address_follows_the_hour_and_length() -> None:
    made = emission(MOLINA, 195.0, hour=ELSEWHERE, length_s=6)

    assert made.key == (f"{ELSEWHERE}#195+6@128k", made.key[1])
    assert made.key[1].endswith("segment=6")


@pytest.mark.parametrize("given", ["query_offset_s", "ref_start_s"])
@pytest.mark.parametrize("source", ["shazam", "local"])
def test_one_offset_without_the_other_is_refused(source: str, given: str) -> None:
    given_offset: dict[str, Any] = {given: 1.0}
    with pytest.raises(ValueError, match="both offsets or neither"):
        emission(MOLINA, 195.0, source=source, **given_offset)


def test_leg_scores_holds_each_tags_verdicts_against_the_plays() -> None:
    plays = plays_from(RECORDS)
    shazam, olaf = [emission(MOLINA, 195.0)], [emission(PRATT, 450.0, source="local")]

    legs = leg_scores(plays, {"12s/shazam": shazam, "12s/olaf": olaf})

    assert list(legs) == ["12s/shazam", "12s/olaf"]
    assert legs["12s/shazam"].verdicts == attribute(plays, shazam)
    assert legs["12s/olaf"].verdicts == attribute(plays, olaf)
