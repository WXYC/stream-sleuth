"""``JsonlOutput``: one JSON object per emission, appended to a local file."""

from __future__ import annotations

import importlib
import json
from datetime import datetime, timezone

import pytest

MOLINA = {"artist": "Juana Molina", "song": "la paradoja", "album": "DOGA", "label": "Sonamos"}
PRATT = {
    "artist": "Jessica Pratt",
    "song": "Back, Baby",
    "album": "On Your Own Love Again",
    "label": "Drag City",
}
NOW = datetime(2026, 10, 6, 23, 15, 0, tzinfo=timezone.utc)


@pytest.fixture
def outputs(fresh_recognizer):
    fresh_recognizer(WXDU_SHAZAM_SECRET="not-a-real-secret")
    return importlib.import_module("stream_sleuth.outputs")


def lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_is_an_output(outputs, tmp_path):
    assert isinstance(outputs.JsonlOutput(tmp_path / "e.jsonl"), outputs.Output)


def test_appends_one_record_per_emission_in_order_with_a_timestamp(outputs, tmp_path):
    path = tmp_path / "e.jsonl"
    out = outputs.JsonlOutput(path, clock=lambda: NOW)
    out.emit(MOLINA)
    out.emit(PRATT)
    assert lines(path) == [
        {**MOLINA, "emitted_at": "2026-10-06T23:15:00+00:00"},
        {**PRATT, "emitted_at": "2026-10-06T23:15:00+00:00"},
    ]


def test_never_truncates_existing_content(outputs, tmp_path):
    path = tmp_path / "e.jsonl"
    path.write_text('{"earlier": "record"}\n', encoding="utf-8")
    outputs.JsonlOutput(path, clock=lambda: NOW).emit(MOLINA)
    outputs.JsonlOutput(path, clock=lambda: NOW).emit(PRATT)
    assert [r.get("artist", r.get("earlier")) for r in lines(path)] == [
        "record",
        "Juana Molina",
        "Jessica Pratt",
    ]


def test_writes_non_ascii_as_utf8_and_round_trips(outputs, tmp_path):
    path = tmp_path / "e.jsonl"
    track = {"artist": "Hermanos Gutiérrez", "song": "El Bueno y el Malo", "album": "", "label": ""}
    outputs.JsonlOutput(path, clock=lambda: NOW).emit(track)
    assert "Gutiérrez" in path.read_text(encoding="utf-8")
    assert lines(path)[0]["artist"] == "Hermanos Gutiérrez"


def test_record_is_on_disk_when_emit_returns(outputs, tmp_path):
    path = tmp_path / "e.jsonl"
    out = outputs.JsonlOutput(path, clock=lambda: NOW)
    out.emit(MOLINA)
    assert lines(path) == [{**MOLINA, "emitted_at": NOW.isoformat()}]


def test_does_not_mutate_the_track(outputs, tmp_path):
    track = dict(MOLINA)
    outputs.JsonlOutput(tmp_path / "e.jsonl", clock=lambda: NOW).emit(track)
    assert track == MOLINA


def test_default_clock_is_utc_now(outputs, tmp_path):
    path = tmp_path / "e.jsonl"
    before = datetime.now(timezone.utc)
    outputs.JsonlOutput(path).emit(MOLINA)
    stamp = datetime.fromisoformat(lines(path)[0]["emitted_at"])
    assert stamp.tzinfo is not None
    assert before <= stamp <= datetime.now(timezone.utc)
