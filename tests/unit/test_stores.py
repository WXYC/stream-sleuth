"""``tests.stores``: the store fixtures hold what the writers write and nothing else."""

from __future__ import annotations

from pathlib import Path

import pytest

from evaluation.results import ResultStore
from tests.stores import olaf_record, shazam_record

ADDRESS = "2026/08/12/202608121600.mp3#45+12@128k"
OLAF_KEYS = {
    "address", "recognizer", "recorded_at", "status", "kind", "artist", "song", "album", "label",
    "offset_s", "confidence", "ref_key", "query_offset_s", "ref_start_s",
}  # fmt: skip


def _only(store: ResultStore) -> dict:
    [record] = store.records()
    return record


@pytest.mark.parametrize(
    ("kind", "status", "artist"),
    [
        ("matched", 200, "Juana Molina"),
        ("no_match", 200, ""),
        ("decode_error", 200, ""),
        ("rate_limited", 429, ""),
        ("server_error", 503, ""),
    ],
)
def test_a_shazam_record_has_the_status_and_fields_outcome_from_gives_its_kind(
    tmp_path: Path, kind: str, status: int, artist: str
) -> None:
    store = ResultStore(tmp_path / "results.jsonl")
    shazam_record(store, ADDRESS, kind)
    record = _only(store)
    assert (record["kind"], record["status"], record["artist"]) == (kind, status, artist)
    assert record["recognizer"] == "shazam@0.8.1, segment=12"


def test_an_olaf_match_has_status_0_a_null_offset_and_olafs_fields(tmp_path: Path) -> None:
    store = ResultStore(tmp_path / "results.jsonl")
    olaf_record(store, ADDRESS)
    record = _only(store)
    assert set(record) == OLAF_KEYS
    assert (record["status"], record["offset_s"], record["kind"]) == (0, None, "matched")
    assert record["recognizer"].startswith("olaf@")


def test_an_olaf_no_match_has_status_0_and_no_answer(tmp_path: Path) -> None:
    store = ResultStore(tmp_path / "results.jsonl")
    olaf_record(store, ADDRESS, "no_match")
    record = _only(store)
    assert (record["status"], record["artist"], record["confidence"]) == (0, "", None)


@pytest.mark.parametrize("kind", ["server_error", "rate_limited", "decode_error"])
def test_an_olaf_record_cannot_be_an_error_run_olaf_never_files(tmp_path: Path, kind: str) -> None:
    with pytest.raises(ValueError, match="run_olaf files only"):
        olaf_record(ResultStore(tmp_path / "results.jsonl"), ADDRESS, kind)
