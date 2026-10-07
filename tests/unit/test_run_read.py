"""The read side of the result stores: matched records to ``EvalIdentification``, and what is uncovered.

No test contacts Shazam or runs Olaf; every store is a JSONL file written by hand or through
``ResultStore.append``.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
from pathlib import Path

import pytest

from evaluation.clips import CAPTURE_LENGTHS_S, ClipAddress
from evaluation.results import (
    Emission,
    ResultStore,
    read_results,
    recognizer_identity,
    study_identities,
    to_identification,
)
from stream_sleuth.recognizers.olaf import recognizer_identity as olaf_identity

HOUR = "2026/08/12/202608121600.mp3"
OTHER = "2026/08/12/202608121700.mp3"
SHAZAM = "shazam@0.8.1, segment=12"
KEY = (f"{HOUR}#45+12@128k", SHAZAM)

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
    result = read_results([ResultStore(path)], {SHAZAM})
    assert result.uncovered == {}
    assert result.scored == {KEY}
    assert result.emissions == [
        Emission(
            KEY,
            ClipAddress(HOUR, 45, 12, "128k"),
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
    ident = converted.found
    assert "query_offset_s" not in ident and "ref_start_s" not in ident
    assert (ident["at"], ident["source"]) == (45.0, "shazam")


def test_an_olaf_answer_is_local_with_its_confidence_offsets_and_ref_key() -> None:
    record = _record(
        "matched", recognizer=olaf_identity("rotation"), offset_s=None, confidence=42.0,
        ref_key="ab" * 20, query_offset_s=3.5, ref_start_s=61.0,
    )  # fmt: skip
    assert to_identification(record) == Emission(
        (f"{HOUR}#45+12@128k", olaf_identity("rotation")),
        ClipAddress(HOUR, 45, 12, "128k"),
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
    result = read_results([store], {SHAZAM})
    assert result.emissions == []  # no emission, and the no_match address is not one either
    assert result.uncovered == {KEY: "server_error"}
    assert result.scored == {(f"{OTHER}#0+12@128k", SHAZAM)}  # the error-only address is not


def test_a_no_match_address_is_scored_and_a_never_tried_one_is_in_neither_set(
    tmp_path: Path,
) -> None:
    result = read_results([_store(tmp_path, _record("no_match"))], {SHAZAM})
    assert (result.emissions, result.scored, result.uncovered) == ([], {KEY}, {})
    empty = read_results([_store(tmp_path)], {SHAZAM})
    assert (empty.emissions, empty.scored, empty.uncovered) == ([], set(), {})


def test_an_address_scored_after_a_failure_is_covered(tmp_path: Path) -> None:
    store = _store(tmp_path, _record("server_error", status=503), _record("no_match"))
    result = read_results([store], {SHAZAM})
    assert (result.scored, result.uncovered) == ({KEY}, {})


def test_legs_at_one_offset_keep_their_own_keys(tmp_path: Path) -> None:
    six = "shazam@0.8.1, segment=6"
    store = _store(
        tmp_path,
        _record("matched"),
        _record("matched", address=f"{HOUR}#45+12@320k"),
        _record("matched", address=f"{HOUR}#45+6@128k", recognizer=six),
    )
    keys = {e.key for e in read_results([store], {SHAZAM, six}).emissions}
    assert keys == {
        (f"{HOUR}#45+12@128k", SHAZAM),
        (f"{HOUR}#45+12@320k", SHAZAM),
        (f"{HOUR}#45+6@128k", six),
    }


@pytest.mark.parametrize(
    ("kinds", "emitted"),
    [
        (["matched", "matched"], 1),  # a repeat counts once
        (["no_match", "matched"], 0),  # the first scoring record is the answer
        (["matched", "no_match"], 1),
        (["server_error", "matched", "matched"], 1),  # an error before it is not scoring
    ],
)
def test_the_first_scoring_record_per_key_wins(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, kinds: list[str], emitted: int
) -> None:
    store = _store(tmp_path, *(_record(k, song=f"take {i}") for i, k in enumerate(kinds)))
    with caplog.at_level(logging.WARNING, logger="evaluation.results"):
        result = read_results([store], {SHAZAM})
    assert len(result.emissions) == emitted
    if emitted:
        assert result.emissions[0].found["song"] == f"take {kinds.index('matched')}"
    assert result.scored == {KEY}
    ignored = sum(k in ("matched", "no_match") for k in kinds) - 1
    assert caplog.text.count("ignored") == ignored


def test_a_key_scored_in_one_store_is_not_uncovered_for_another_store_error(
    tmp_path: Path,
) -> None:
    first = _store(tmp_path, _record("matched"))
    second = ResultStore(tmp_path / "copy.jsonl")
    second.path.write_text(json.dumps(_record("server_error", status=503)) + "\n")
    result = read_results([first, second], {SHAZAM})
    assert (len(result.emissions), result.scored, result.uncovered) == (1, {KEY}, {})


@pytest.mark.parametrize("second", ["matched", "no_match"])
def test_the_first_scoring_record_wins_across_stores(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, second: str
) -> None:
    first = _store(tmp_path, _record("matched", song="first store"))
    copy = ResultStore(tmp_path / "copy.jsonl")
    copy.path.write_text(json.dumps(_record(second, song="second store")) + "\n")
    with caplog.at_level(logging.WARNING, logger="evaluation.results"):
        result = read_results([first, copy], {SHAZAM})
    assert [e.found["song"] for e in result.emissions] == ["first store"]
    assert caplog.text.count("ignored") == 1


def test_the_union_over_stores_has_every_stores_keys(tmp_path: Path) -> None:
    olaf = olaf_identity("rotation")
    first = _store(tmp_path, _record("decode_error"))
    second = ResultStore(tmp_path / "olaf.jsonl")
    second.path.write_text(json.dumps(_record("server_error", recognizer=olaf)) + "\n")
    result = read_results([first, second], {SHAZAM, olaf})
    assert result.uncovered == {KEY: "decode_error", (KEY[0], olaf): "server_error"}


def test_a_result_filed_under_one_floor_is_not_returned_for_another(tmp_path: Path) -> None:
    loose, strict = olaf_identity("rotation", 5), olaf_identity("rotation", 12)
    olaf = {"confidence": 9.0, "ref_key": "ab" * 20, "query_offset_s": 0.0, "ref_start_s": 1.0}
    records = [
        _record("matched", recognizer=loose, **olaf),
        _record("matched", recognizer=loose, address=f"{HOUR}#60+12@128k", **olaf),
        _record("server_error", recognizer=loose, address=f"{OTHER}#0+12@128k"),
    ]
    store = _store(tmp_path, *records)
    nothing = read_results([store], {strict})
    assert (nothing.emissions, nothing.scored, nothing.uncovered) == ([], set(), {})  # all three
    result = read_results([store], {loose})
    assert len(result.emissions) == 2
    assert result.uncovered == {(f"{OTHER}#0+12@128k", loose): "server_error"}


def test_a_bare_string_is_not_a_set_of_identities(tmp_path: Path) -> None:
    # A str is a Collection[str], and `in` on it is a substring test: "min=1" is inside "min=12".
    loose, strict = olaf_identity("rotation", 1), olaf_identity("rotation", 12)
    store = _store(tmp_path, _record("no_match", recognizer=loose))
    with pytest.raises(TypeError, match="set of identities"):
        read_results([store], strict)  # type: ignore[arg-type]
    assert read_results([store], {strict}).scored == set()  # the same request, as a set


def test_records_under_other_identities_are_counted_in_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    other = "shazam@9.9.9, segment=12"
    store = _store(tmp_path, _record("matched"), _record("no_match", recognizer=other))
    with caplog.at_level(logging.WARNING, logger="evaluation.results"):
        result = read_results([store], {SHAZAM})
    assert len(result.emissions) == 1
    assert [r.levelno for r in caplog.records] == [logging.WARNING]
    assert other in caplog.text and "1 record" in caplog.text


def test_the_warning_counts_every_record_and_each_identity(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    a, b = "shazam@9.9.9, segment=12", "shazam@9.9.9, segment=6"
    others = [
        _record("no_match", recognizer=a, address=f"{HOUR}#{15 * i}+12@128k") for i in range(3)
    ]
    store = _store(tmp_path, _record("matched"), *others, _record("no_match", recognizer=b))
    with caplog.at_level(logging.WARNING, logger="evaluation.results"):
        read_results([store], {SHAZAM})
    assert "4 record(s)" in caplog.text
    assert f"{a!r}: 3" in caplog.text and f"{b!r}: 1" in caplog.text


def test_no_warning_when_every_record_is_under_a_requested_identity(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="evaluation.results"):
        read_results([_store(tmp_path, _record("matched"))], {SHAZAM})
    assert caplog.records == []


@pytest.mark.parametrize("kind", ["matched", "no_match", "server_error"])
def test_an_unparseable_address_raises_with_the_store_path_and_line(
    tmp_path: Path, kind: str
) -> None:
    bad = _record(kind, address="not-an-address")
    store = _store(tmp_path, _record("no_match", address=f"{OTHER}#0+12@128k"), bad)
    with pytest.raises(ValueError, match=rf"{re.escape(str(store.path))}:2: .*not-an-address"):
        read_results([store], {SHAZAM})


def test_an_unparseable_address_under_another_identity_is_not_read(tmp_path: Path) -> None:
    store = _store(tmp_path, _record("matched", address="not-an-address", recognizer="x@1"))
    assert read_results([store], {SHAZAM}).emissions == []


def test_the_stores_are_not_rewritten(tmp_path: Path) -> None:
    store = _store(tmp_path, _record("matched"), _record("server_error"))
    before = store.path.read_bytes()
    read_results([store], {SHAZAM})
    assert store.path.read_bytes() == before


def test_a_store_that_does_not_exist_yet_reads_as_nothing(tmp_path: Path) -> None:
    result = read_results([ResultStore(tmp_path / "none.jsonl")], {SHAZAM})
    assert (result.emissions, result.scored, result.uncovered) == ([], set(), {})


def test_one_read_per_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = _store(tmp_path, _record("matched"))
    reads = 0
    original = ResultStore.records

    def counting(self: ResultStore) -> list[dict]:
        nonlocal reads
        reads += 1
        return original(self)

    monkeypatch.setattr(ResultStore, "records", counting)
    read_results([store], {SHAZAM})
    assert reads == 1  # scored, last, and the emissions all come from one snapshot


def test_study_identities_name_every_shazam_segment_and_the_snapshot() -> None:
    assert study_identities("rotation", 12) == {
        *(recognizer_identity(n) for n in CAPTURE_LENGTHS_S),
        olaf_identity("rotation", 12),
    }


def test_importing_the_read_side_loads_neither_the_runner_nor_shazam():
    code = (
        "import sys, evaluation.results, evaluation.score; "
        "print(sorted(m for m in ('evaluation.run', 'evaluation.shazam_eval', 'shazamio', 'aiohttp') "
        "if m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)

    assert out.stdout.strip() == "[]"
