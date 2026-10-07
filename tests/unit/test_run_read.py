"""The read side of the result stores: matched records to ``EvalIdentification``, and what is uncovered.

No test contacts Shazam or runs Olaf; every store is a JSONL file written by hand or through
``ResultStore.append``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from evaluation.clips import CAPTURE_LENGTHS_S
from evaluation.run import read_results, study_identities, to_identification
from evaluation.shazam_eval import ResultStore, recognizer_identity
from stream_sleuth.recognizers.olaf import recognizer_identity as olaf_identity

HOUR = "2026/08/12/202608121600.mp3"
OTHER = "2026/08/12/202608121700.mp3"
SHAZAM = "shazam@0.8.1, segment=12"

# A line in the merged Shazam store format, as the first legs wrote it.
MERGED_LINE = (
    f'{{"address": "{HOUR}#45+12@128k", "recognizer": "{SHAZAM}", '
    '"recorded_at": "2026-10-06T20:00:00+00:00", "status": 200, "kind": "matched", '
    '"artist": "Juana Molina", "song": "la paradoja", "album": "DOGA", "label": "Sonamos", '
    '"offset_s": 41.2}\n'
)


def _record(kind: str, **fields: object) -> dict:
    return {**json.loads(MERGED_LINE), "kind": kind, **fields}


def _store(tmp_path: Path, *records: dict) -> ResultStore:
    path = tmp_path / "results.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return ResultStore(path)


def test_a_line_in_the_merged_store_format_reads_back_and_converts(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    path.write_text(MERGED_LINE)
    emissions, uncovered = read_results([ResultStore(path)], {SHAZAM})
    assert uncovered == {}
    assert emissions == [
        (
            HOUR,
            {
                "artist": "Juana Molina",
                "song": "la paradoja",
                "album": "DOGA",
                "label": "Sonamos",
                "at": 45.0,
                "source": "shazam",
                "query_offset_s": 0.0,
                "ref_start_s": 41.2,
            },
        )
    ]


def test_a_shazam_answer_with_no_offset_carries_neither_offset() -> None:
    converted = to_identification(_record("matched", offset_s=None))
    assert converted is not None
    ident = converted[1]
    assert "query_offset_s" not in ident and "ref_start_s" not in ident
    assert (ident["at"], ident["source"]) == (45.0, "shazam")


def test_an_olaf_answer_is_local_with_its_confidence_offsets_and_ref_key() -> None:
    record = _record(
        "matched", recognizer=olaf_identity("rotation"), offset_s=None, confidence=42.0,
        ref_key="ab" * 20, query_offset_s=3.5, ref_start_s=61.0,
    )  # fmt: skip
    assert to_identification(record) == (
        HOUR,
        {
            "artist": "Juana Molina",
            "song": "la paradoja",
            "album": "DOGA",
            "label": "Sonamos",
            "at": 45.0,
            "source": "local",
            "confidence": 42.0,
            "query_offset_s": 3.5,
            "ref_start_s": 61.0,
            "ref_key": "ab" * 20,
        },
    )


@pytest.mark.parametrize("kind", ["no_match", "rate_limited", "server_error", "decode_error"])
def test_only_a_matched_record_converts(kind: str) -> None:
    assert to_identification(_record(kind)) is None


def test_an_address_with_only_error_records_is_uncovered_with_its_latest_kind(
    tmp_path: Path,
) -> None:
    store = _store(
        tmp_path,
        _record("rate_limited", status=429),
        _record("server_error", status=403),
        _record("no_match", address=f"{OTHER}#0+12@128k"),
    )
    emissions, uncovered = read_results([store], {SHAZAM})
    assert emissions == []  # no emission, and the no_match address is not an emission either
    assert uncovered == {(f"{HOUR}#45+12@128k", SHAZAM): "server_error"}


def test_an_address_scored_after_a_failure_is_covered(tmp_path: Path) -> None:
    store = _store(tmp_path, _record("server_error", status=503), _record("no_match"))
    assert read_results([store], {SHAZAM}) == ([], {})


def test_the_uncovered_set_is_the_union_over_stores(tmp_path: Path) -> None:
    olaf = olaf_identity("rotation")
    first = _store(tmp_path, _record("decode_error"))
    second = ResultStore(tmp_path / "olaf.jsonl")
    second.path.write_text(json.dumps(_record("server_error", recognizer=olaf)) + "\n")
    _, uncovered = read_results([first, second], {SHAZAM, olaf})
    assert uncovered == {
        (f"{HOUR}#45+12@128k", SHAZAM): "decode_error",
        (f"{HOUR}#45+12@128k", olaf): "server_error",
    }


def test_a_result_filed_under_one_floor_is_not_returned_for_another(tmp_path: Path) -> None:
    loose, strict = olaf_identity("rotation", 5), olaf_identity("rotation", 12)
    olaf = {"confidence": 9.0, "ref_key": "ab" * 20, "query_offset_s": 0.0, "ref_start_s": 1.0}
    store = _store(tmp_path, *(_record("matched", recognizer=loose, **olaf) for _ in range(2)))
    assert read_results([store], {strict}) == ([], {})
    emissions, _ = read_results([store], {loose})
    assert len(emissions) == 2


def test_records_under_other_identities_are_counted_in_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    other = "shazam@9.9.9, segment=12"
    store = _store(tmp_path, _record("matched"), _record("no_match", recognizer=other))
    with caplog.at_level(logging.WARNING, logger="evaluation.run"):
        emissions, _ = read_results([store], {SHAZAM})
    assert len(emissions) == 1
    assert [r.levelno for r in caplog.records] == [logging.WARNING]
    assert other in caplog.text and "1 record" in caplog.text


def test_no_warning_when_every_record_is_under_a_requested_identity(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="evaluation.run"):
        read_results([_store(tmp_path, _record("matched"))], {SHAZAM})
    assert caplog.records == []


def test_the_stores_are_not_rewritten(tmp_path: Path) -> None:
    store = _store(tmp_path, _record("matched"), _record("server_error"))
    before = store.path.read_bytes()
    read_results([store], {SHAZAM})
    assert store.path.read_bytes() == before


def test_a_store_that_does_not_exist_yet_reads_as_nothing(tmp_path: Path) -> None:
    assert read_results([ResultStore(tmp_path / "none.jsonl")], {SHAZAM}) == ([], {})


def test_study_identities_name_every_shazam_segment_and_the_snapshot() -> None:
    assert study_identities("rotation", 12) == {
        *(recognizer_identity(n) for n in CAPTURE_LENGTHS_S),
        olaf_identity("rotation", 12),
    }
