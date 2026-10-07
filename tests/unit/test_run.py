"""The leg runner: the legs table, Shazam legs under one throttle, and the free-space preflight.

No test contacts Shazam (a localhost ``FakeShazam``); ``hour_addresses`` and ``cut`` are
replaced, so no test needs ffmpeg either.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections import namedtuple
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from evaluation import run as run_mod
from evaluation.clips import ClipAddress
from evaluation.run import LEGS, main, preflight, read_hours, run_legs
from evaluation.shazam_eval import (
    MAX_RETRIES,
    CountingClient,
    ResultStore,
    ShazamOutcome,
    Throttle,
    budget_from_env,
    recognizer_identity,
)
from tests.characterization.shazam_responses import NO_MATCH
from tests.shazam_fake import HTML_429, FakeShazam, clock_at, json_response, write_tone

HOUR = "2026/08/12/202608121600.mp3"
OTHER = "2026/08/12/202608121700.mp3"
HOURS = {"all": [HOUR, OTHER], "subset": [HOUR]}

Setup = namedtuple("Setup", "store client fake")


@pytest.fixture(autouse=True)
def fake_audio(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Two clips per hour, 0 s and 15 s, each 'cut' to a synthetic tone."""
    tone = tmp_path / "tone.wav"
    write_tone(tone)

    def addresses(keys, archive_dir, length_s, profile):  # noqa: ANN001, ANN202
        return [ClipAddress(k, 15 * i, length_s, profile) for k in keys for i in range(2)]

    @contextmanager
    def cut(address, hour_path, work_dir) -> Iterator[Path]:  # noqa: ANN001
        yield tone

    monkeypatch.setattr(run_mod, "hour_addresses", addresses)
    monkeypatch.setattr(run_mod, "cut", cut)


@pytest.fixture
def shazam(tmp_path: Path) -> Iterator[Callable[[list[tuple]], Setup]]:
    fakes: list[FakeShazam] = []

    def make(responses: list[tuple]) -> Setup:
        fake = FakeShazam(responses)
        fakes.append(fake)
        clock = clock_at("2026-10-06T20:00:00")
        throttle = Throttle(tmp_path / "throttle.json", 500, 0.0, clock=clock, sleep=clock.sleep)
        client = CountingClient(throttle, base_url=fake.url)
        return Setup(ResultStore(tmp_path / "shazam.jsonl"), client, fake)

    yield make
    for fake in fakes:
        fake.close()


def legs(*names: str):
    return [leg for leg in LEGS if leg.name in names]


def test_the_legs_table_is_the_shazam_query_budget() -> None:
    assert [(leg.length_s, leg.profile, leg.hours) for leg in LEGS] == [
        (12, "128k", "all"),
        (6, "128k", "subset"),
        (20, "128k", "subset"),
        (12, "320k", "subset"),
    ]
    assert len({leg.name for leg in LEGS}) == len(LEGS)


def test_every_leg_reads_the_hours_of_selection_json(tmp_path: Path) -> None:
    path = tmp_path / "selection.json"
    hours = {
        HOUR: {"group": "contrast", "band": "evening", "subset": True},
        OTHER: {"group": "contrast", "band": "evening", "subset": False},
    }
    path.write_text(json.dumps({"hours": hours}))
    assert read_hours(path) == {"all": [HOUR, OTHER], "subset": [HOUR]}


@pytest.mark.parametrize(
    ("hours", "names"),
    [
        ({HOUR: {"subset": True}, OTHER: {"subset": "false"}}, OTHER),  # a hand edit, truthy
        ({HOUR: {"subset": 1}}, HOUR),
        ({HOUR: {"group": "contrast"}}, HOUR),  # no subset label at all
        ({HOUR: True}, HOUR),
        ([HOUR, OTHER], "hours"),  # a list of keys, not an object keyed by hour
    ],
)
def test_a_selection_without_a_boolean_subset_per_hour_is_refused_naming_it(
    tmp_path: Path, hours: object, names: str
) -> None:
    path = tmp_path / "selection.json"
    path.write_text(json.dumps({"hours": hours}))
    with pytest.raises(SystemExit, match=re.escape(names)) as refusal:
        read_hours(path)
    assert str(path) in str(refusal.value)


def test_two_shazam_legs_share_one_throttle(tmp_path: Path, shazam) -> None:
    setup = shazam([json_response(200, NO_MATCH)] * 6)
    report = run_legs(legs("12s", "6s-subset"), HOURS, tmp_path, tmp_path, *setup[:2])
    assert report == {"12s": "done", "6s-subset": "done"}
    assert len(setup.fake.requests) == 6  # 2 hours x 2 clips, then 1 hour x 2 clips
    assert json.loads((tmp_path / "throttle.json").read_text())["count"] == 6
    assert {r["recognizer"] for r in setup.store.records()} == {
        recognizer_identity(12),
        recognizer_identity(6),
    }


def test_a_leg_that_is_stopped_stops_the_next_leg_before_it_sends_anything(
    tmp_path: Path, shazam, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup = shazam([HTML_429, json_response(200, NO_MATCH)])
    grids: list[int] = []
    real = run_mod.hour_addresses

    def spy(keys, archive_dir, length_s, profile):  # noqa: ANN001, ANN202
        grids.append(length_s)
        return real(keys, archive_dir, length_s, profile)

    monkeypatch.setattr(run_mod, "hour_addresses", spy)
    report = run_legs(
        legs("12s", "6s-subset", "12s-320k-subset"), HOURS, tmp_path, tmp_path, *setup[:2]
    )
    assert report == {
        "12s": "rate_limited",
        "6s-subset": "not started",
        "12s-320k-subset": "not started",
    }
    assert len(setup.fake.requests) == 1
    assert grids == [12]  # a leg that is not started does not even decode its hours


def test_a_daily_cap_also_stops_the_later_legs(tmp_path: Path, shazam) -> None:
    setup = shazam([json_response(200, NO_MATCH)] * 4)
    clock = clock_at("2026-10-06T20:00:00")
    setup.client.throttle.close()
    with Throttle(tmp_path / "throttle.json", 3, 0.0, clock=clock, sleep=clock.sleep) as throttle:
        client = CountingClient(throttle, base_url=setup.fake.url)
        report = run_legs(legs("12s", "6s-subset"), HOURS, tmp_path, tmp_path, setup.store, client)
    assert report == {"12s": "daily_cap", "6s-subset": "not started"}
    assert len(setup.fake.requests) == 3


def test_a_future_dated_throttle_state_is_reported_as_the_leg_refused(
    tmp_path: Path, shazam
) -> None:
    setup = shazam([json_response(200, NO_MATCH)])
    ahead = clock_at("2026-10-06T21:00:00").now  # an hour ahead of the throttle's clock
    state = {"day": "2026-10-06", "count": 0, "last": ahead, "stopped": False}
    (tmp_path / "throttle.json").write_text(json.dumps(state))
    report = run_legs(legs("12s", "6s-subset"), HOURS, tmp_path, tmp_path, *setup[:2])
    assert report == {"12s": "refused", "6s-subset": "not started"}
    assert setup.fake.requests == []


@pytest.fixture
def data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A data directory whose selection.json labels HOUR subset and OTHER not."""
    data = tmp_path / "data"
    (data / "shazam").mkdir(parents=True)
    monkeypatch.setenv("STREAM_SLEUTH_DATA_DIR", str(data))
    monkeypatch.setenv("STREAM_SLEUTH_SHAZAM_MIN_INTERVAL_S", "0")
    labels = {HOUR: {"subset": True}, OTHER: {"subset": False}}
    (data / "selection.json").write_text(json.dumps({"hours": labels}))
    return data


def test_the_cli_reports_every_leg_and_exits_non_zero_on_a_refused_state(
    data: Path, caplog: pytest.LogCaptureFixture
) -> None:
    state = {"day": "2026-10-06", "count": 0, "last": time.time() + 3600, "stopped": False}
    (data / "shazam" / "throttle.json").write_text(json.dumps(state))
    fake = FakeShazam([])
    try:
        with caplog.at_level(logging.INFO):
            assert main(["--legs", "12s", "6s-subset", "--base-url", fake.url]) == 1
    finally:
        fake.close()
    assert "is in the future" in caplog.text
    assert "12s: refused" in caplog.text
    assert "6s-subset: not started" in caplog.text
    assert fake.requests == []


def test_the_cli_takes_its_daily_cap_from_the_environment(
    data: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("STREAM_SLEUTH_SHAZAM_RATE_PER_DAY", "1")
    fake = FakeShazam([json_response(200, NO_MATCH)] * 2)
    try:
        with caplog.at_level(logging.INFO):
            assert main(["--legs", "12s", "6s-subset", "--base-url", fake.url]) == 0
    finally:
        fake.close()
    assert len(fake.requests) == 1
    assert "12s: daily_cap" in caplog.text
    assert "6s-subset: not started" in caplog.text
    assert "1 requests" in caplog.text


def test_the_cli_ends_with_the_out_of_retries_summary(
    data: Path, caplog: pytest.LogCaptureFixture
) -> None:
    store = ResultStore(data / "shazam" / "results.jsonl")
    tired = f"{HOUR}#0+6@128k"
    for _ in range(MAX_RETRIES + 1):
        store.append(tired, recognizer_identity(6), ShazamOutcome(503, "server_error"))
    fake = FakeShazam([json_response(200, NO_MATCH)])
    try:
        with caplog.at_level(logging.INFO):
            assert main(["--legs", "6s-subset", "--base-url", fake.url]) == 0
    finally:
        fake.close()
    assert len(fake.requests) == 1  # the other clip; the tired one is not queried
    assert "out of retries and not queried" in caplog.text
    assert tired in caplog.text


def test_preflight_refuses_below_five_gib(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(run_mod.shutil, "disk_usage", lambda p: usage(0, 0, (5 << 30) - 1))
    with pytest.raises(SystemExit, match="free"):
        preflight(tmp_path)
    monkeypatch.setattr(run_mod.shutil, "disk_usage", lambda p: usage(0, 0, 5 << 30))
    preflight(tmp_path)


def test_the_budget_variables_are_read_in_one_place(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("STREAM_SLEUTH_SHAZAM_RATE_PER_DAY", raising=False)
    monkeypatch.delenv("STREAM_SLEUTH_SHAZAM_MIN_INTERVAL_S", raising=False)
    assert budget_from_env() == (500, 20.0)
    monkeypatch.setenv("STREAM_SLEUTH_SHAZAM_RATE_PER_DAY", "7")
    monkeypatch.setenv("STREAM_SLEUTH_SHAZAM_MIN_INTERVAL_S", "1.5")
    assert budget_from_env() == (7, 1.5)
