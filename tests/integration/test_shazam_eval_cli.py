"""The Shazam CLI end to end against a localhost server, cutting real clips with ffmpeg."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

from evaluation.clips import hour_addresses
from evaluation.shazam_eval import MAX_RETRIES, Throttle, ThrottleBusyError, main
from stream_sleuth.paths import CHECKOUT, DataPathError
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


def test_a_repeated_hour_key_in_the_hours_file_is_queried_once(tmp_path: Path) -> None:
    _make_hour(tmp_path / "archive", HOUR, 40)  # two clips: 0 and 15 s
    fake = FakeShazam([_json(200, JESSICA_PRATT), _json(200, NO_MATCH)])
    try:
        assert main([*_flags(tmp_path, fake, HOUR, HOUR), *_explicit_paths(tmp_path)]) == 0
    finally:
        fake.close()
    assert len(fake.requests) == 2
    assert len(_records(tmp_path / "shazam.jsonl")) == 2


def test_a_429_stops_the_cli_for_the_day(tmp_path: Path) -> None:
    _make_hour(tmp_path / "archive", HOUR, 60)
    fake = FakeShazam([_json(200, JESSICA_PRATT), HTML_429])
    try:
        assert main([*_flags(tmp_path, fake, HOUR), *_explicit_paths(tmp_path)]) == 0
    finally:
        fake.close()
    assert [r["kind"] for r in _records(tmp_path / "shazam.jsonl")] == ["matched", "rate_limited"]
    assert len(fake.requests) == 2
    assert json.loads((tmp_path / "throttle.json").read_text())["stopped"] == "rate_limited"


def test_the_end_summary_lists_addresses_that_are_out_of_retries(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _make_hour(tmp_path / "archive", HOUR, 20)  # one clip: 0 s
    fake = FakeShazam([_json(503, {})] * (1 + MAX_RETRIES))
    argv = [*_flags(tmp_path, fake, HOUR), *_explicit_paths(tmp_path)]
    try:
        for _ in range(MAX_RETRIES):
            assert main(argv) == 0
        assert "out of retries" not in caplog.text
        with caplog.at_level("WARNING"):
            assert main(argv) == 0  # the last allowed retry
            caplog.clear()
            assert main(argv) == 0  # nothing left to query
    finally:
        fake.close()
    assert len(fake.requests) == 1 + MAX_RETRIES
    messages = [r.getMessage() for r in caplog.records]
    summary = [m for m in messages if "out of retries and not queried" in m]
    assert len(summary) == 1 and f"{HOUR}#0+12@128k" in summary[0]


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


@pytest.mark.parametrize("flag", ["--store", "--state", "--work-dir"])
@pytest.mark.parametrize("where", ["inside", "relative"])
def test_the_cli_refuses_a_path_in_the_checkout_before_any_request(
    tmp_path: Path, flag: str, where: str
) -> None:
    _make_hour(tmp_path / "archive", HOUR, 20)
    # Without the guard the run would query: the hour has an address.
    assert hour_addresses([HOUR], tmp_path / "archive", 12, "128k")
    bad = CHECKOUT / "shazam-guard-test" / "x" if where == "inside" else Path("x")
    paths = dict(zip(_explicit_paths(tmp_path)[::2], _explicit_paths(tmp_path)[1::2], strict=True))
    paths[flag] = str(bad)
    fake = FakeShazam([_json(200, NO_MATCH)])
    try:
        with pytest.raises(DataPathError):
            main([*_flags(tmp_path, fake, HOUR), *(p for pair in paths.items() for p in pair)])
    finally:
        fake.close()
    assert fake.requests == []
    assert not (CHECKOUT / "shazam-guard-test").exists()
    assert not Path("x").exists()
    for created in ("work", "shazam.jsonl", "throttle.json", "throttle.json.lock"):
        assert not (tmp_path / created).exists()


def test_a_second_run_on_a_held_state_file_is_refused_before_any_request(tmp_path: Path) -> None:
    _make_hour(tmp_path / "archive", HOUR, 20)
    fake = FakeShazam([_json(200, NO_MATCH)])
    try:
        with Throttle(tmp_path / "throttle.json", 500, 20.0):  # the first run, still going
            with pytest.raises(ThrottleBusyError, match=re.escape(str(tmp_path / "throttle.json"))):
                main([*_flags(tmp_path, fake, HOUR), *_explicit_paths(tmp_path)])
            assert fake.requests == []
        assert main([*_flags(tmp_path, fake, HOUR), *_explicit_paths(tmp_path)]) == 0
    finally:
        fake.close()
    assert len(fake.requests) == 1


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
