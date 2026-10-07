"""The Shazam CLI end to end against a localhost server, cutting real clips with ffmpeg."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from evaluation.shazam_eval import main
from tests.characterization.shazam_responses import JESSICA_PRATT, NO_MATCH
from tests.unit.test_shazam_eval import HTML_429, FakeShazam, _json

pytestmark = pytest.mark.ffmpeg

HOUR = "2026/08/12/202608121600.mp3"
NEXT_HOUR = "2026/08/12/202608121700.mp3"


@pytest.fixture(autouse=True)
def _no_interval(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STREAM_SLEUTH_SHAZAM_MIN_INTERVAL_S", "0")


def _make_hour(archive: Path, key: str, seconds: int) -> None:
    (archive / key).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={seconds}",
            "-c:a",
            "libmp3lame",
            "-b:a",
            "128k",
            str(archive / key),
        ],
        check=True,
    )


def _records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _flags(tmp_path: Path, fake: FakeShazam, *hours: str) -> list[str]:
    (tmp_path / "hours.txt").write_text("".join(h + "\n" for h in hours))
    return [
        "--hours",
        str(tmp_path / "hours.txt"),
        "--archive-dir",
        str(tmp_path / "archive"),
        "--base-url",
        fake.url,
    ]


def _explicit_paths(tmp_path: Path) -> list[str]:
    return [
        "--work-dir",
        str(tmp_path / "work"),
        "--store",
        str(tmp_path / "shazam.jsonl"),
        "--state",
        str(tmp_path / "throttle.json"),
    ]


def test_a_short_hour_is_queried_only_at_the_addresses_that_fit(tmp_path: Path) -> None:
    _make_hour(tmp_path / "archive", HOUR, 40)  # 12 s clips fit at 0 and 15 s, not at 30 s
    fake = FakeShazam([_json(200, JESSICA_PRATT), _json(200, NO_MATCH)])
    try:
        assert main([*_flags(tmp_path, fake, HOUR), *_explicit_paths(tmp_path)]) == 0
    finally:
        fake.close()
    records = _records(tmp_path / "shazam.jsonl")
    assert [(r["address"], r["kind"]) for r in records] == [
        (f"{HOUR}#0+12@128k", "matched"),
        (f"{HOUR}#15+12@128k", "no_match"),
    ]
    assert records[0]["artist"] == "Jessica Pratt"
    assert len(fake.requests) == 2
    assert list((tmp_path / "work").iterdir()) == []


def test_a_429_stops_the_cli_for_the_day(tmp_path: Path) -> None:
    _make_hour(tmp_path / "archive", HOUR, 60)
    fake = FakeShazam([_json(200, JESSICA_PRATT), HTML_429])
    try:
        assert main([*_flags(tmp_path, fake, HOUR), *_explicit_paths(tmp_path)]) == 0
    finally:
        fake.close()
    assert [r["kind"] for r in _records(tmp_path / "shazam.jsonl")] == ["matched", "rate_limited"]
    assert len(fake.requests) == 2
    assert json.loads((tmp_path / "throttle.json").read_text())["stopped"] is True


def test_missing_and_unreadable_hours_are_logged_and_skipped_without_a_request(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    missing, unreadable = "2026/08/12/202608121400.mp3", "2026/08/12/202608121500.mp3"
    (tmp_path / "archive" / unreadable).parent.mkdir(parents=True)
    (tmp_path / "archive" / unreadable).write_bytes(b"not audio")
    _make_hour(tmp_path / "archive", NEXT_HOUR, 20)  # one clip: 0 s
    fake = FakeShazam([_json(200, NO_MATCH)])
    try:
        with caplog.at_level("WARNING"):
            assert (
                main(
                    [
                        *_flags(tmp_path, fake, missing, unreadable, NEXT_HOUR),
                        *_explicit_paths(tmp_path),
                    ]
                )
                == 0
            )
    finally:
        fake.close()
    assert [r["address"] for r in _records(tmp_path / "shazam.jsonl")] == [f"{NEXT_HOUR}#0+12@128k"]
    assert len(fake.requests) == 1
    assert missing in caplog.text and unreadable in caplog.text


def test_paths_default_from_the_data_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "data"
    monkeypatch.setenv("STREAM_SLEUTH_DATA_DIR", str(data))
    _make_hour(tmp_path / "archive", HOUR, 20)
    fake = FakeShazam([_json(200, NO_MATCH)])
    try:
        assert main(_flags(tmp_path, fake, HOUR)) == 0
    finally:
        fake.close()
    assert len(_records(data / "shazam" / "results.jsonl")) == 1
    assert (data / "shazam" / "throttle.json").exists()
    assert list((data / "clips").iterdir()) == []
